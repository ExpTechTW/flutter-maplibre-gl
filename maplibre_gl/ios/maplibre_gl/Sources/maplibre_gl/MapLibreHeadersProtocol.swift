import Foundation

// URLProtocol that:
// 1. Asks Dart for tile bodies (Dart owns cache + metering)
// 2. Injects custom headers on live requests
// 3. On a live tile miss, forwards the body to Dart for persist/meter
//
// Forwarding uses ONE shared ephemeral URLSession with `protocolClasses = []`
// (no recursion, no per-request session). Creating + invalidate'ing a session
// per tile previously UAF'd on `com.apple.network.connections` / objc_retain.
final class MapLibreHeadersProtocol: URLProtocol {
    private static let handledKey = "MapLibreHeadersProtocolHandled"
    private static let maxPutBytes = 2 * 1024 * 1024

    /// Shared forwarder — never invalidate; cancel tasks only.
    private static let forwardSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = []
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.urlCache = nil
        config.timeoutIntervalForRequest = 30
        return URLSession(configuration: config)
    }()

    private var activeTask: URLSessionDataTask?
    private var stopped = false

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
        if let hit = MapLibreDartTileBridge.get(url: urlString), let url = request.url {
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
        activeTask = task
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
            // Cancellation after stopLoading — don't touch the client.
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

        if let http = response as? HTTPURLResponse,
           http.statusCode == 200,
           !body.isEmpty,
           body.count <= Self.maxPutBytes,
           let url = http.url?.absoluteString ?? request.url?.absoluteString
        {
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
