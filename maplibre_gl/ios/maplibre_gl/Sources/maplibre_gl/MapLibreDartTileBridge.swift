import Flutter
import Foundation

/// Thin bridge: ask Dart for an ExpTech tile body. No native cache / put —
/// Dart Dio + SQLite own the network and persist path.
enum MapLibreDartTileBridge {
    private static let lock = NSLock()
    private static var channel: FlutterMethodChannel?
    /// SQLite-backed get can be slower than the old Dart LRU — give it room.
    private static let getTimeout: TimeInterval = 0.5

    static func attach(messenger: FlutterBinaryMessenger) {
        let ch = FlutterMethodChannel(
            name: "plugins.flutter.io/maplibre_gl/tile_cache",
            binaryMessenger: messenger
        )
        lock.lock()
        channel = ch
        lock.unlock()
    }

    /// ExpTech immutable tiles only (basemap / radar / sat / DPM). Glyphs and
    /// other `*.pbf` must NOT match — those still go through native HTTP.
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
}
