import Foundation

/// URLProtocol that:
/// 1. Serves ExpTech tiles out of the Dart-owned cache when present
/// 2. On a miss, fetches once through a shared URLSession and reports the body
///    back to Dart so it lands in SQLite
/// 3. Injects custom headers on all live requests (tiles / style / glyphs / …)
///
/// Forwarding uses ONE shared ephemeral URLSession with `protocolClasses = []`
/// (no recursion, no per-request session).
///
/// **Nothing here blocks.** A cache lookup that has to reach Dart returns
/// through a callback, so a URL-loading thread is never parked — that parking
/// was what turned a timeline scrub into a request storm: the lookup timed out,
/// the timeout read as a miss, and MapLibre re-downloaded tiles that were
/// already on disk.
final class MapLibreHeadersProtocol: URLProtocol {
    private static let handledKey = "MapLibreHeadersProtocolHandled"
    private static let maxPutBytes = 2 * 1024 * 1024

    /// Freshness advertised for ExpTech tiles.
    ///
    /// Their URLs are content-addressed — a new radar frame is a new path — so
    /// a body can never go stale under its own URL. Saying so lets MapLibre's
    /// own ambient database answer repeat requests without entering the URL
    /// loading system at all. Without this header MapLibre treats every tile as
    /// already expired and re-requests it on every reveal, which is what made a
    /// timeline scrub generate traffic for tiles it had just drawn.
    private static let tileCacheControl = "public, max-age=604800, immutable"

    /// Shared forwarder — never invalidate; cancel tasks only.
    private static let forwardSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = []
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.urlCache = nil
        // Scrub storms: fail fast instead of holding 30 s slots for abandoned frames.
        config.timeoutIntervalForRequest = 8
        config.timeoutIntervalForResource = 12
        config.httpMaximumConnectionsPerHost = 16
        return URLSession(configuration: config)
    }()

    /// Cancels in-flight forwards whose URL contains one of [needles]; an empty
    /// list cancels everything.
    ///
    /// Scoping matters: a scrub abandons the frames it swept past, but the
    /// basemap and the frame the finger landed on must keep loading. [completion]
    /// runs once the cancels have been applied, so a caller can await that
    /// without any thread blocking to provide the guarantee.
    static func cancelForwardTasks(
        matching needles: [String],
        completion: (() -> Void)? = nil
    ) {
        forwardSession.getAllTasks { tasks in
            for task in tasks {
                guard !needles.isEmpty else {
                    task.cancel()
                    continue
                }
                let url = task.originalRequest?.url?.absoluteString
                    ?? task.currentRequest?.url?.absoluteString
                    ?? ""
                if needles.contains(where: { url.contains($0) }) {
                    task.cancel()
                }
            }
            completion?()
        }
    }

    private let stateLock = NSLock()
    private var activeTask: URLSessionDataTask?
    private var stopped = false
    private var forwardingTile = false

    private var isStopped: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return stopped
    }

    override class func canInit(with request: URLRequest) -> Bool {
        guard URLProtocol.property(forKey: handledKey, in: request) == nil else {
            return false
        }
        let scheme = request.url?.scheme?.lowercased() ?? ""
        return scheme == "http" || scheme == "https"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        return request
    }

    override func startLoading() {
        let urlString = request.url?.absoluteString ?? ""
        guard MapLibreDartTileBridge.isCacheableUrl(urlString) else {
            forward()
            return
        }
        forwardingTile = true

        // Session-hot tiles resolve with no IPC at all.
        if let hit = MapLibreDartTileBridge.cached(url: urlString) {
            serve(hit)
            return
        }
        MapLibreDartTileBridge.lookup(url: urlString) { [weak self] hit in
            guard let self, !self.isStopped else { return }
            if let hit {
                self.serve(hit)
            } else {
                self.forward()
            }
        }
    }

    override func stopLoading() {
        stateLock.lock()
        stopped = true
        let task = activeTask
        activeTask = nil
        stateLock.unlock()
        task?.cancel()
    }

    /// Restamps a live tile response for MapLibre consumption.
    ///
    /// URLSession transparently inflates gzip, but often **leaves**
    /// `Content-Encoding: gzip` on the response. Forwarding that header with an
    /// already-plain body makes MapLibre gunzip again and drop the tile —
    /// silent blank for MVT feeds like DPM AED (`application/vnd.mapbox-vector-tile`).
    /// Basemap PBF is more forgiving, which is why the map looked fine without
    /// AED. Strip encoding, rewrite length, and (when the origin said nothing)
    /// advertise [tileCacheControl].
    private static func clientTileResponse(
        _ response: URLResponse,
        bodyLength: Int
    ) -> URLResponse {
        guard let http = response as? HTTPURLResponse,
              let url = http.url
        else { return response }

        var headers: [String: String] = [:]
        for (key, value) in http.allHeaderFields {
            guard let name = key as? String else { continue }
            let lower = name.lowercased()
            if lower == "content-encoding" || lower == "content-length" {
                continue
            }
            headers[name] = "\(value)"
        }
        headers["Content-Length"] = "\(bodyLength)"
        // Origin `no-store` (CDN) fights MapLibre's ambient DB and forces every
        // reveal back through this bridge. DPM / radar / satellite URLs are
        // content-addressed by z/x/y — restamp so a decoded tile sticks.
        if http.value(forHTTPHeaderField: "Cache-Control") == nil
            || http.value(forHTTPHeaderField: "Cache-Control")?
            .localizedCaseInsensitiveContains("no-store") == true
        {
            headers["Cache-Control"] = tileCacheControl
        }
        return HTTPURLResponse(
            url: url,
            statusCode: http.statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: headers
        ) ?? response
    }

    /// Replays a cached body as a synthetic 200 — no network.
    private func serve(_ hit: MapLibreTileEntry) {
        guard !isStopped, let url = request.url else { return }
        var headers: [String: String] = [
            "Content-Type": hit.contentType ?? "application/octet-stream",
            "Content-Length": "\(hit.data.count)",
            "Cache-Control": Self.tileCacheControl,
        ]
        if let etag = hit.etag {
            headers["ETag"] = etag
        }
        let response = HTTPURLResponse(
            url: url,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: headers
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: hit.data)
        client?.urlProtocolDidFinishLoading(self)
    }

    /// Sends the request upstream with custom headers applied.
    private func forward() {
        guard !isStopped else { return }

        let mutable = (request as NSURLRequest).mutableCopy() as! NSMutableURLRequest
        URLProtocol.setProperty(true, forKey: Self.handledKey, in: mutable)

        let (headers, shouldApply) = MapLibreCustomHeaders.headersIfApplicable(
            to: mutable.url?.absoluteString ?? ""
        )
        if shouldApply {
            for (key, value) in headers {
                mutable.setValue(value, forHTTPHeaderField: key)
            }
        }

        let task = Self.forwardSession.dataTask(with: mutable as URLRequest) {
            [weak self] data, response, error in
            self?.finishForward(data: data, response: response, error: error)
        }

        stateLock.lock()
        if stopped {
            stateLock.unlock()
            task.cancel()
            return
        }
        activeTask = task
        stateLock.unlock()
        task.resume()
    }

    private func finishForward(data: Data?, response: URLResponse?, error: Error?) {
        guard !isStopped else { return }
        stateLock.lock()
        activeTask = nil
        stateLock.unlock()

        if let error {
            if (error as NSError).code == NSURLErrorCancelled { return }
            client?.urlProtocol(self, didFailWithError: error)
            return
        }

        guard let response else {
            client?.urlProtocol(
                self,
                didFailWithError: URLError(.badServerResponse)
            )
            return
        }

        let body = data ?? Data()
        // Empty 200s make ImageIO probe garbage ("GIF" decode errors); treat as fail.
        if forwardingTile,
           let http = response as? HTTPURLResponse,
           http.statusCode == 200,
           body.isEmpty
        {
            client?.urlProtocol(
                self,
                didFailWithError: URLError(.zeroByteResource)
            )
            return
        }

        let clientResponse = forwardingTile
            ? Self.clientTileResponse(response, bodyLength: body.count)
            : response
        client?.urlProtocol(
            self,
            didReceive: clientResponse,
            cacheStoragePolicy: .notAllowed
        )
        if !body.isEmpty {
            client?.urlProtocol(self, didLoad: body)
        }

        if forwardingTile,
           let http = response as? HTTPURLResponse,
           http.statusCode == 200 || http.statusCode == 404,
           !body.isEmpty || http.statusCode == 404,
           body.count <= Self.maxPutBytes,
           let url = http.url?.absoluteString ?? request.url?.absoluteString
        {
            // 200 body or basemap-style 404 hole — both land in Dart SQLite.
            // Body is already inflated (URLSession); never store gzip framing.
            MapLibreDartTileBridge.put(
                url: url,
                data: body,
                contentType: http.value(forHTTPHeaderField: "Content-Type"),
                etag: http.value(forHTTPHeaderField: "ETag")
            )
        }

        client?.urlProtocolDidFinishLoading(self)
    }
}
