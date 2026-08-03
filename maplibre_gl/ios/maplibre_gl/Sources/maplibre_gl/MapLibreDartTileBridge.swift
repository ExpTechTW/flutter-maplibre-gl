import Flutter
import Foundation

/// Thin bridge: Dart owns SQLite + metering. Native asks `get` before fetch and
/// `put` after a tile miss download.
///
/// A process-local [NSCache] sits in front of the MethodChannel so a radar
/// settle (dozens of tiles) does not serialize every hit through Flutter's
/// main isolate — that jam was causing false misses → connection pile-up →
/// timeouts, and ImageIO "GIF" decode failures on empty/error bodies.
enum MapLibreDartTileBridge {
    private static let lock = NSLock()
    private static var channel: FlutterMethodChannel?
    private static let getTimeout: TimeInterval = 0.1
    private static let mem = NSCache<NSString, MemEntry>()

    private final class MemEntry: NSObject {
        let data: Data
        let contentType: String?
        let etag: String?
        init(data: Data, contentType: String?, etag: String?) {
            self.data = data
            self.contentType = contentType
            self.etag = etag
        }
    }

    static func attach(messenger: FlutterBinaryMessenger) {
        mem.totalCostLimit = 32 * 1024 * 1024
        let ch = FlutterMethodChannel(
            name: "plugins.flutter.io/maplibre_gl/tile_cache",
            binaryMessenger: messenger
        )
        ch.setMethodCallHandler { call, result in
            switch call.method {
            case "cancelPendingFetches":
                MapLibreHeadersProtocol.cancelAllForwardTasks()
                result(nil)
            default:
                result(FlutterMethodNotImplemented)
            }
        }
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

    private static func memPut(
        url: String,
        data: Data,
        contentType: String?,
        etag: String?
    ) {
        mem.setObject(
            MemEntry(data: data, contentType: contentType, etag: etag),
            forKey: url as NSString,
            cost: data.count
        )
    }

    /// Synchronous lookup on a background thread (never call from main).
    static func get(url: String) -> (data: Data, contentType: String?, etag: String?)? {
        guard isTileUrl(url) else { return nil }
        if let hit = mem.object(forKey: url as NSString) {
            return (hit.data, hit.contentType, hit.etag)
        }

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
        let data = Data(typed.data)
        let contentType = payload["contentType"] as? String
        let etag = payload["etag"] as? String
        memPut(url: url, data: data, contentType: contentType, etag: etag)
        return (data, contentType, etag)
    }

    /// Fire-and-forget: persist a network-fetched tile into Dart SQLite.
    static func putAsync(
        url: String,
        data: Data,
        contentType: String?,
        etag: String?
    ) {
        guard isTileUrl(url) else { return }
        memPut(url: url, data: data, contentType: contentType, etag: etag)
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
