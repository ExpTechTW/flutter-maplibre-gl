import Flutter
import Foundation

/// One cached tile body plus the response metadata needed to replay it.
final class MapLibreTileEntry: NSObject {
    let data: Data
    let contentType: String?
    let etag: String?
    let sourceSize: Int?
    let bodyCost: Int

    init(
        data: Data,
        contentType: String?,
        etag: String?,
        sourceSize: Int? = nil,
        bodyCost: Int? = nil
    ) {
        self.data = data
        self.contentType = contentType
        self.etag = etag
        self.sourceSize = sourceSize
        self.bodyCost = bodyCost ?? data.count
    }
}

/// Bounded in-process tile mirror — an LRU by byte cost.
///
/// Deliberately not `NSCache`: this one must be *enumerable* so Dart can evict a
/// frame's tiles by URL substring, and its eviction must be deterministic so
/// `filterMissing` is trustworthy enough to skip re-injecting bytes.
private final class MapLibreTileMemStore {
    private struct Slot {
        let entry: MapLibreTileEntry
        var usedAt: UInt64
    }

    private let lock = NSLock()
    private var slots: [String: Slot] = [:]
    private var clock: UInt64 = 0
    private var bytes = 0
    private var limit: Int

    /// Once over budget, trim down to this share of it. Recency is a per-entry
    /// stamp, so choosing what to drop costs a sort — doing that in batches
    /// keeps it off the per-tile path, where it would run on every insert once
    /// the store is full.
    private static let trimTarget = 0.85

    init(limit: Int) {
        self.limit = limit
    }

    func setLimit(_ newLimit: Int) {
        lock.lock()
        defer { lock.unlock() }
        limit = max(0, newLimit)
        trimLocked()
    }

    func get(_ url: String) -> MapLibreTileEntry? {
        lock.lock()
        defer { lock.unlock() }
        guard var slot = slots[url] else { return nil }
        clock += 1
        slot.usedAt = clock
        slots[url] = slot
        return slot.entry
    }

    func contains(_ url: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return slots[url] != nil
    }

    func put(_ url: String, _ entry: MapLibreTileEntry) {
        lock.lock()
        defer { lock.unlock() }
        if let existing = slots[url] {
            bytes -= existing.entry.bodyCost
        }
        clock += 1
        slots[url] = Slot(entry: entry, usedAt: clock)
        bytes += entry.bodyCost
        trimLocked()
    }

    /// Drops every entry whose URL contains one of [needles]. Empty = drop all.
    func evict(matching needles: [String]) {
        lock.lock()
        defer { lock.unlock() }
        if needles.isEmpty {
            slots.removeAll()
            bytes = 0
            return
        }
        for (url, slot) in slots where needles.contains(where: { url.contains($0) }) {
            slots.removeValue(forKey: url)
            bytes -= slot.entry.bodyCost
        }
    }

    /// Current memory usage and the cap, for the Dart side to decide how much
    /// more it may inject before the mirror trims.
    func usage() -> (used: Int, limit: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (bytes, limit)
    }

    private func trimLocked() {
        guard bytes > limit else { return }
        let target = Int(Double(limit) * Self.trimTarget)
        for url in slots.sorted(by: { $0.value.usedAt < $1.value.usedAt }).map(\.key) {
            guard bytes > target else { return }
            if let slot = slots.removeValue(forKey: url) {
                bytes -= slot.entry.bodyCost
            }
        }
    }
}

/// Dart owns the tile cache (SQLite + traffic metering); native owns only a
/// bounded in-process mirror and **never parks a loader thread on Flutter**.
///
/// Three rules are what let a radar timeline scrub at finger speed:
///
/// 1. **Memory first.** A repeat request for a tile already seen this session is
///    answered synchronously with zero IPC.
/// 2. **Batched, asynchronous lookups.** Misses are coalesced by URL into one
///    `getBatch` per ~6 ms window and answered through a callback, so a viewport
///    of tiles costs one message and one SQLite query instead of dozens of
///    blocking round-trips. Nothing waits on a semaphore — the previous design
///    blocked each URL-loading thread for up to 100 ms, and a timeout there read
///    as a cache *miss*, re-downloading bytes that were already on disk.
/// 3. **Batched writes.** Network bodies buffer and flush as one `putBatch`, so
///    a tile storm is a handful of messages rather than one per tile.
///
/// Dart may also push bytes ahead of demand (`injectTiles`) — the warm path a
/// timeline uses so the next frame is a memory hit *before* it is revealed.
enum MapLibreDartTileBridge {
    // MARK: - Channel

    private static let channelLock = NSLock()
    private static var channel: FlutterMethodChannel?

    // MARK: - Tuning

    /// Coalescing window for `getBatch` — long enough to gather the tiles
    /// MapLibre asks for in one frame, far below a network round-trip.
    private static let getWindowMs = 6
    private static let maxGetBatch = 64

    /// Writes are pure bookkeeping for Dart, so they can wait longer and batch
    /// harder than reads.
    private static let putWindowMs = 250
    private static let maxPutBatch = 32

    /// Fail a batch open rather than stranding tiles if the engine goes away.
    private static let batchTimeout: TimeInterval = 4

    private static let defaultMemoryLimit = 2 * 1024 * 1024

    /// `data: 1` on the MethodChannel expands to this process-wide body.
    /// Empty radar tiles occur under thousands of distinct URLs, so repeating
    /// the same 68 bytes only creates codec copies and GC pressure.
    private static let transparentRgbaPngWireToken = 1
    private static let malformedEmptyGif = Data(
        base64Encoded: "R0lGODlhAQABAAAAACH5BAEAAAAALAAAAAABAAEAAAIBADs="
    )!
    private static let opaqueBlackPlaceholderPng = Data(
        base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )!
    private static let transparentRgbaPng = Data(
        base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="
    )!
    private static var dartSupportsTransparentRgbaPngToken = false

    /// Canonicalises every known empty-raster representation before it reaches
    /// MapLibre, L1, or Dart. Exact-body matching cannot rewrite a valid tile.
    static func normalizeRasterPayload(
        data: Data,
        contentType: String?
    ) -> (data: Data, contentType: String?, repaired: Bool) {
        let knownEmpty = data == malformedEmptyGif
            || data == opaqueBlackPlaceholderPng
            || data == transparentRgbaPng
        guard knownEmpty else {
            return (data, contentType, false)
        }
        let alreadyCanonical = data == transparentRgbaPng
            && contentType?.lowercased().hasPrefix("image/png") == true
        return (transparentRgbaPng, "image/png", !alreadyCanonical)
    }

    private static func entryFromWire(
        wireData: Any?,
        contentType: String?,
        etag: String?,
        sourceSize: Int? = nil
    ) -> MapLibreTileEntry? {
        let data: Data
        if let token = wireData as? Int,
           token == transparentRgbaPngWireToken
        {
            data = transparentRgbaPng
        } else if let typed = wireData as? FlutterStandardTypedData {
            data = Data(typed.data)
        } else {
            return nil
        }
        let payload = normalizeRasterPayload(data: data, contentType: contentType)
        return MapLibreTileEntry(
            data: payload.data,
            contentType: payload.contentType,
            etag: etag,
            sourceSize: sourceSize,
            bodyCost: payload.data == transparentRgbaPng ? 0 : payload.data.count
        )
    }

    private static func wireData(_ data: Data) -> Any {
        dartSupportsTransparentRgbaPngToken && data == transparentRgbaPng
            ? transparentRgbaPngWireToken
            : FlutterStandardTypedData(bytes: data)
    }

    // MARK: - State

    private static let queue = DispatchQueue(label: "org.maplibre.dart-tile-bridge")
    private static let mem = MapLibreTileMemStore(limit: defaultMemoryLimit)

    /// In-flight/pending lookups, deduped by URL — concurrent requests for the
    /// same tile share one round-trip.
    private static var pendingGets: [String: [(MapLibreTileEntry?) -> Void]] = [:]
    private static var getFlushScheduled = false

    private static var pendingPuts: [String: MapLibreTileEntry] = [:]
    private static var putFlushScheduled = false

    // MARK: - Attach

    static func attach(messenger: FlutterBinaryMessenger) {
        dartSupportsTransparentRgbaPngToken = false
        let ch = FlutterMethodChannel(
            name: "plugins.flutter.io/maplibre_gl/tile_cache",
            binaryMessenger: messenger
        )
        ch.setMethodCallHandler { call, result in
            switch call.method {
            case "injectTiles":
                let args = call.arguments as? [String: Any]
                let entries = args?["entries"] as? [[String: Any]] ?? []
                for row in entries {
                    guard let url = row["url"] as? String,
                          let entry = entryFromWire(
                              wireData: row["data"],
                              contentType: row["contentType"] as? String,
                              etag: row["etag"] as? String
                          )
                    else { continue }
                    mem.put(url, entry)
                }
                // Echo the post-injection usage back so Dart can stop injecting
                // when the mirror is near full, without a second round-trip.
                let usage = mem.usage()
                result(["used": usage.used, "limit": usage.limit])

            case "filterMissing":
                let args = call.arguments as? [String: Any]
                let urls = args?["urls"] as? [String] ?? []
                result(urls.filter { !mem.contains($0) })

            case "evictTiles":
                let args = call.arguments as? [String: Any]
                mem.evict(matching: args?["contains"] as? [String] ?? [])
                result(nil)

            case "setCacheablePatterns":
                let args = call.arguments as? [String: Any]
                let patterns = args?["patterns"] as? [String] ?? []
                patternLock.lock()
                cacheablePatterns = patterns
                patternLock.unlock()
                result(nil)

            case "setMemoryLimit":
                let args = call.arguments as? [String: Any]
                mem.setLimit(args?["bytes"] as? Int ?? defaultMemoryLimit)
                let wireTokens = args?["wireTokens"] as? [Int] ?? []
                dartSupportsTransparentRgbaPngToken = wireTokens.contains(
                    transparentRgbaPngWireToken
                )
                result(["wireTokens": [transparentRgbaPngWireToken]])

            case "cancelPendingFetches":
                let args = call.arguments as? [String: Any]
                let needles = args?["contains"] as? [String] ?? []
                // Completes once the cancels are applied, without ever blocking
                // the platform thread to get that guarantee.
                MapLibreHeadersProtocol.cancelForwardTasks(matching: needles) {
                    result(nil)
                }

            default:
                result(FlutterMethodNotImplemented)
            }
        }
        channelLock.lock()
        channel = ch
        channelLock.unlock()

        // Pull what Dart wants cached, rather than waiting to be told.
        //
        // Dart binds the cache during app bootstrap — long before the first map
        // widget, and therefore before this plugin is registered — so its push
        // lands on a channel that does not exist yet and is silently lost. The
        // symptom is not a slow cache but a **dead** one: no URL matches, so
        // nothing is served from or written to Dart at all.
        DispatchQueue.main.async {
            ch.invokeMethod("cacheablePatterns", arguments: nil) { result in
                guard let patterns = result as? [String] else { return }
                patternLock.lock()
                cacheablePatterns = patterns
                patternLock.unlock()
            }
        }
    }

    private static var activeChannel: FlutterMethodChannel? {
        channelLock.lock()
        defer { channelLock.unlock() }
        return channel
    }

    // MARK: - URL classification

    /// URL fragments Dart wants to own the caching of.
    ///
    /// Empty until Dart configures it, which is deliberate: an unconfigured
    /// bridge caches nothing rather than guessing at URL shapes the Dart side
    /// may not actually store. Guessing is how the two ends drift into a URL
    /// native keeps asking about and Dart never has.
    private static let patternLock = NSLock()
    private static var cacheablePatterns: [String] = []

    /// Whether [url] is one Dart caches — see [cacheablePatterns].
    static func isCacheableUrl(_ url: String) -> Bool {
        patternLock.lock()
        let patterns = cacheablePatterns
        patternLock.unlock()
        for pattern in patterns where url.contains(pattern) {
            return true
        }
        return false
    }

    // MARK: - Read

    /// Synchronous memory probe — safe from any thread, never touches Flutter.
    static func cached(url: String) -> MapLibreTileEntry? {
        guard isCacheableUrl(url) else { return nil }
        return mem.get(url)
    }

    /// Asks Dart for [url], batching with any other lookup in the same window.
    ///
    /// [completion] runs on the bridge queue and always runs exactly once —
    /// `nil` means "not cached, go to the network".
    static func lookup(url: String, completion: @escaping (MapLibreTileEntry?) -> Void) {
        guard isCacheableUrl(url) else {
            completion(nil)
            return
        }
        if let hit = mem.get(url) {
            completion(hit)
            return
        }
        queue.async {
            if var waiters = pendingGets[url] {
                waiters.append(completion)
                pendingGets[url] = waiters
                return
            }
            pendingGets[url] = [completion]
            if pendingGets.count >= maxGetBatch {
                flushGets()
                return
            }
            guard !getFlushScheduled else { return }
            getFlushScheduled = true
            queue.asyncAfter(deadline: .now() + .milliseconds(getWindowMs)) {
                getFlushScheduled = false
                flushGets()
            }
        }
    }

    /// Must run on [queue].
    private static func flushGets() {
        guard !pendingGets.isEmpty else { return }
        let batch = pendingGets
        pendingGets.removeAll(keepingCapacity: true)

        guard let ch = activeChannel else {
            for waiters in batch.values {
                for waiter in waiters { waiter(nil) }
            }
            return
        }

        var settled = false
        let settle: ([String: Any]) -> Void = { payload in
            // Always on [queue].
            guard !settled else { return }
            settled = true
            for (url, waiters) in batch {
                var entry: MapLibreTileEntry?
                if let row = payload[url] as? [String: Any],
                   let hit = entryFromWire(
                       wireData: row["data"],
                       contentType: row["contentType"] as? String,
                       etag: row["etag"] as? String
                   )
                {
                    mem.put(url, hit)
                    entry = hit
                }
                for waiter in waiters { waiter(entry) }
            }
        }

        let urls = Array(batch.keys)
        DispatchQueue.main.async {
            ch.invokeMethod("getBatch", arguments: ["urls": urls]) { result in
                let payload = result as? [String: Any] ?? [:]
                queue.async { settle(payload) }
            }
        }
        // Watchdog: a detached engine must degrade to "miss", not strand tiles.
        queue.asyncAfter(deadline: .now() + batchTimeout) { settle([:]) }
    }

    // MARK: - Write

    /// Records a network-fetched tile: memory immediately, Dart on the next
    /// batch flush.
    static func put(
        url: String,
        data: Data,
        contentType: String?,
        etag: String?,
        sourceSize: Int? = nil
    ) {
        guard isCacheableUrl(url) else { return }
        let payload = normalizeRasterPayload(data: data, contentType: contentType)
        let entry = MapLibreTileEntry(
            data: payload.data,
            contentType: payload.contentType,
            etag: etag,
            sourceSize: sourceSize,
            bodyCost: payload.data == transparentRgbaPng ? 0 : payload.data.count
        )
        mem.put(url, entry)
        queue.async {
            pendingPuts[url] = entry
            if pendingPuts.count >= maxPutBatch {
                flushPuts()
                return
            }
            guard !putFlushScheduled else { return }
            putFlushScheduled = true
            queue.asyncAfter(deadline: .now() + .milliseconds(putWindowMs)) {
                putFlushScheduled = false
                flushPuts()
            }
        }
    }

    /// Must run on [queue].
    private static func flushPuts() {
        guard !pendingPuts.isEmpty, let ch = activeChannel else {
            pendingPuts.removeAll(keepingCapacity: true)
            return
        }
        let batch = pendingPuts
        pendingPuts.removeAll(keepingCapacity: true)

        let entries: [[String: Any]] = batch.map { url, entry in
            var row: [String: Any] = [
                "url": url,
                "data": wireData(entry.data),
            ]
            if let contentType = entry.contentType { row["contentType"] = contentType }
            if let etag = entry.etag { row["etag"] = etag }
            if let sourceSize = entry.sourceSize { row["sourceSize"] = sourceSize }
            return row
        }
        DispatchQueue.main.async {
            ch.invokeMethod("putBatch", arguments: ["entries": entries])
        }
    }
}
