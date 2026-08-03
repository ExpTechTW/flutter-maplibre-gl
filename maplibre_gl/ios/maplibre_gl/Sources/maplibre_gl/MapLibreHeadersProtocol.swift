import Foundation

// URLProtocol that:
// 1. Serves ExpTech tiles from Dart SQLite when present
// 2. On miss, fetches once via a shared URLSession and puts the body into Dart
// 3. Injects custom headers on all live requests (tiles / style / glyphs / …)
//
// Forwarding uses ONE shared ephemeral URLSession with `protocolClasses = []`
// (no recursion, no per-request session). AmbientPrefetch is off — MapLibre is
// the sole tile network path, so there is no Dio+native double fetch.
final class MapLibreHeadersProtocol: URLProtocol {
    private static let handledKey = "MapLibreHeadersProtocolHandled"
    private static let maxPutBytes = 2 * 1024 * 1024

    /// Shared forwarder — never invalidate; cancel tasks only.
    private static let forwardSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = []
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.urlCache = nil
        // Scrub storms: fail fast instead of holding 30s slots for abandoned frames.
        config.timeoutIntervalForRequest = 8
        config.timeoutIntervalForResource = 12
        config.httpMaximumConnectionsPerHost = 16
        return URLSession(configuration: config)
    }()

    /// Drop every in-flight forward (abandoned radar/sat frames after scrub).
    /// Blocks until the cancel list is applied so the next fetch cannot race.
    static func cancelAllForwardTasks() {
        let sem = DispatchSemaphore(value: 0)
        forwardSession.getAllTasks { tasks in
            for task in tasks {
                task.cancel()
            }
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 0.5)
    }

    private var activeTask: URLSessionDataTask?
    private var stopped = false
    private var forwardingTile = false

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

        // ExpTech tile hit → Dart only (no network).
        if MapLibreDartTileBridge.isTileUrl(urlString) {
            if let hit = MapLibreDartTileBridge.get(url: urlString), let url = request.url {
                guard !stopped else { return }
                var headers: [String: String] = [
                    "Content-Type": hit.contentType ?? "application/octet-stream",
                    "Content-Length": "\(hit.data.count)",
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
                return
            }
            // stopLoading may have fired while get() blocked — don't start a fetch.
            guard !stopped else { return }
        }

        forwardingTile = MapLibreDartTileBridge.isTileUrl(urlString)

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

        guard !stopped else { return }
        let task = Self.forwardSession.dataTask(with: mutable as URLRequest) {
            [weak self] data, response, error in
            self?.finishForward(data: data, response: response, error: error)
        }
        activeTask = task
        if stopped {
            task.cancel()
            activeTask = nil
            return
        }
        task.resume()
    }

    override func stopLoading() {
        stopped = true
        activeTask?.cancel()
        activeTask = nil
    }

    private func finishForward(data: Data?, response: URLResponse?, error: Error?) {
        guard !stopped else { return }
        activeTask = nil

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
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if !body.isEmpty {
            client?.urlProtocol(self, didLoad: body)
        }

        if forwardingTile,
           let http = response as? HTTPURLResponse,
           (http.statusCode == 200 || http.statusCode == 404),
           body.count <= Self.maxPutBytes,
           let url = http.url?.absoluteString ?? request.url?.absoluteString
        {
            // 200 body or basemap-style 404 hole — both land in Dart SQLite.
            MapLibreDartTileBridge.putAsync(
                url: url,
                data: body,
                contentType: http.value(forHTTPHeaderField: "Content-Type"),
                etag: http.value(forHTTPHeaderField: "ETag")
            )
        }

        client?.urlProtocolDidFinishLoading(self)
    }
}
