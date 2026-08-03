import Flutter
import Foundation

/// Thin bridge: Dart owns SQLite + metering. Native asks `get` before fetch and
/// `put` after a tile miss download.
enum MapLibreDartTileBridge {
    private static let lock = NSLock()
    private static var channel: FlutterMethodChannel?
    private static let getTimeout: TimeInterval = 0.1

    static func attach(messenger: FlutterBinaryMessenger) {
        let ch = FlutterMethodChannel(
            name: "plugins.flutter.io/maplibre_gl/tile_cache",
            binaryMessenger: messenger
        )
        lock.lock()
        channel = ch
        lock.unlock()
    }

    /// ExpTech immutable tiles only (basemap / radar / sat / DPM).
    static func isTileUrl(_ url: String) -> Bool {
        if url.contains("/api/v1/map/tiles/") { return true }
        if url.contains("/api/v2/tiles/") { return true }
        return false
    }

    /// Synchronous lookup on a background thread (never call from main).
    static func get(url: String) -> (data: Data, contentType: String?, etag: String?)? {
        guard isTileUrl(url) else { return nil }
        lock.lock()
        let ch = channel
        lock.unlock()
        guard let ch else { return nil }

        var payload: [String: Any]?
        let sem = DispatchSemaphore(value: 0)
        DispatchQueue.main.async {
            ch.invokeMethod("get", arguments: ["url": url]) { result in
                payload = result as? [String: Any]
                sem.signal()
            }
        }
        if sem.wait(timeout: .now() + getTimeout) == .timedOut {
            return nil
        }
        guard let payload,
              let typed = payload["data"] as? FlutterStandardTypedData
        else {
            return nil
        }
        return (
            Data(typed.data),
            payload["contentType"] as? String,
            payload["etag"] as? String
        )
    }

    /// Fire-and-forget: persist a network-fetched tile into Dart SQLite.
    static func putAsync(
        url: String,
        data: Data,
        contentType: String?,
        etag: String?
    ) {
        guard isTileUrl(url) else { return }
        lock.lock()
        let ch = channel
        lock.unlock()
        guard let ch else { return }
        var args: [String: Any] = [
            "url": url,
            "data": FlutterStandardTypedData(bytes: data),
        ]
        if let contentType { args["contentType"] = contentType }
        if let etag { args["etag"] = etag }
        DispatchQueue.main.async {
            ch.invokeMethod("put", arguments: args)
        }
    }
}
