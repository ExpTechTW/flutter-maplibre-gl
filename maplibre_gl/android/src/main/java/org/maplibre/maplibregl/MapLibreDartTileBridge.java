package org.maplibre.maplibregl;

import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import androidx.annotation.Nullable;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Dart owns the tile cache (SQLite + traffic metering); native owns only a bounded in-process
 * mirror.
 *
 * <p>Three rules are what let a radar timeline scrub at finger speed:
 *
 * <ol>
 *   <li><b>Memory first.</b> A repeat request for a tile already seen this session is answered with
 *       zero IPC.
 *   <li><b>Batched lookups.</b> Misses are coalesced by URL into one {@code getBatch} per ~6 ms
 *       window, so a viewport of tiles costs one message and one SQLite query instead of dozens of
 *       independent round-trips. Only OkHttp's own dispatcher threads ever wait, never MapLibre's
 *       map thread.
 *   <li><b>Batched writes.</b> Network bodies buffer and flush as one {@code putBatch}.
 * </ol>
 *
 * <p>Dart may also push bytes ahead of demand ({@code injectTiles}) — the warm path a timeline uses
 * so the next frame is a memory hit <i>before</i> it is revealed.
 */
final class MapLibreDartTileBridge {

  /** One cached tile body plus the response metadata needed to replay it. */
  static final class Entry {
    final byte[] data;
    @Nullable final String contentType;
    @Nullable final String etag;
    @Nullable final Integer sourceSize;
    final int bodyCost;

    Entry(
        byte[] data,
        @Nullable String contentType,
        @Nullable String etag,
        @Nullable Integer sourceSize) {
      this.data = data;
      this.contentType = contentType;
      this.etag = etag;
      this.sourceSize = sourceSize;
      this.bodyCost = data == TRANSPARENT_RGBA_PNG ? 0 : data.length;
    }
  }

  /** Coalescing window for {@code getBatch} — well under a network round-trip. */
  private static final long GET_WINDOW_MS = 6;

  private static final int MAX_GET_BATCH = 64;
  private static final long GET_TIMEOUT_MS = 2500;
  private static final long PUT_WINDOW_MS = 250;
  private static final int MAX_PUT_BATCH = 32;
  private static final int DEFAULT_MEMORY_LIMIT = 2 * 1024 * 1024;

  /** {@code data: 1} on the MethodChannel expands to this shared body. */
  private static final int TRANSPARENT_RGBA_PNG_WIRE_TOKEN = 1;
  private static final byte[] MALFORMED_EMPTY_GIF =
      Base64.decode("R0lGODlhAQABAAAAACH5BAEAAAAALAAAAAABAAEAAAIBADs=", Base64.DEFAULT);
  private static final byte[] OPAQUE_BLACK_PLACEHOLDER_PNG =
      Base64.decode(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
          Base64.DEFAULT);
  private static final byte[] TRANSPARENT_RGBA_PNG =
      Base64.decode(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII=",
          Base64.DEFAULT);
  private static volatile boolean dartSupportsTransparentRgbaPngToken;

  /** A body/content-type pair after exact empty-raster canonicalisation. */
  static final class NormalizedPayload {
    final byte[] data;
    @Nullable final String contentType;
    final boolean repaired;

    NormalizedPayload(byte[] data, @Nullable String contentType, boolean repaired) {
      this.data = data;
      this.contentType = contentType;
      this.repaired = repaired;
    }
  }

  /** Canonicalises known empty raster bodies; valid tile bytes are untouched. */
  static NormalizedPayload normalizeRasterPayload(
      byte[] data, @Nullable String contentType) {
    boolean knownEmpty =
        Arrays.equals(data, MALFORMED_EMPTY_GIF)
            || Arrays.equals(data, OPAQUE_BLACK_PLACEHOLDER_PNG)
            || Arrays.equals(data, TRANSPARENT_RGBA_PNG);
    if (!knownEmpty) return new NormalizedPayload(data, contentType, false);
    boolean alreadyCanonical =
        Arrays.equals(data, TRANSPARENT_RGBA_PNG)
            && contentType != null
            && contentType.toLowerCase(java.util.Locale.US).startsWith("image/png");
    return new NormalizedPayload(TRANSPARENT_RGBA_PNG, "image/png", !alreadyCanonical);
  }

  @Nullable
  private static Entry entryFromWire(
      Object rawData,
      @Nullable String contentType,
      @Nullable String etag,
      @Nullable Integer sourceSize) {
    final byte[] data;
    if (rawData instanceof Number
        && ((Number) rawData).intValue() == TRANSPARENT_RGBA_PNG_WIRE_TOKEN) {
      data = TRANSPARENT_RGBA_PNG;
    } else if (rawData instanceof byte[]) {
      data = (byte[]) rawData;
    } else {
      return null;
    }
    NormalizedPayload payload = normalizeRasterPayload(data, contentType);
    return new Entry(payload.data, payload.contentType, etag, sourceSize);
  }

  private static Object wireData(byte[] data) {
    return dartSupportsTransparentRgbaPngToken && data == TRANSPARENT_RGBA_PNG
        ? TRANSPARENT_RGBA_PNG_WIRE_TOKEN
        : data;
  }

  private static final Object channelLock = new Object();
  private static final Object patternLock = new Object();
  private static List<String> cacheablePatterns = Collections.emptyList();
  private static MethodChannel channel;
  private static BinaryMessenger activeMessenger;
  private static final Map<BinaryMessenger, MethodChannel> attachedChannels =
      new IdentityHashMap<>();
  private static final Handler main = new Handler(Looper.getMainLooper());

  private static final MemStore mem = new MemStore(DEFAULT_MEMORY_LIMIT);

  /** Pending lookups, deduped by URL — concurrent requests share one round-trip. */
  private static final Object batchLock = new Object();

  private static final LinkedHashMap<String, Waiter> pendingGets = new LinkedHashMap<>();
  private static boolean getFlushScheduled = false;
  private static final LinkedHashMap<String, Entry> pendingPuts = new LinkedHashMap<>();
  private static boolean putFlushScheduled = false;

  private MapLibreDartTileBridge() {}

  /** A shared lookup: every caller for the same URL blocks on one latch. */
  private static final class Waiter {
    final CountDownLatch latch = new CountDownLatch(1);
    final AtomicReference<Entry> value = new AtomicReference<>(null);

    void settle(@Nullable Entry entry) {
      value.set(entry);
      latch.countDown();
    }
  }

  /** Bounded LRU by byte cost. Enumerable so Dart can evict a frame's tiles by URL substring. */
  private static final class MemStore {
    private final LinkedHashMap<String, Entry> map = new LinkedHashMap<>(64, 0.75f, true);
    private int bytes = 0;
    private int limit;

    MemStore(int limit) {
      this.limit = limit;
    }

    synchronized void setLimit(int newLimit) {
      limit = Math.max(0, newLimit);
      trim();
    }

    @Nullable
    synchronized Entry get(String url) {
      return map.get(url);
    }

    synchronized boolean contains(String url) {
      return map.containsKey(url);
    }

    synchronized void put(String url, Entry entry) {
      Entry previous = map.put(url, entry);
      if (previous != null) bytes -= previous.bodyCost;
      bytes += entry.bodyCost;
      trim();
    }

    /** Drops every entry whose URL contains one of {@code needles}; empty drops all. */
    synchronized void evict(List<String> needles) {
      if (needles == null || needles.isEmpty()) {
        map.clear();
        bytes = 0;
        return;
      }
      Iterator<Map.Entry<String, Entry>> it = map.entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry<String, Entry> row = it.next();
        for (String needle : needles) {
          if (row.getKey().contains(needle)) {
            bytes -= row.getValue().bodyCost;
            it.remove();
            break;
          }
        }
      }
    }

    /** Current memory usage and the cap, for the Dart side to decide how much
     * more it may inject before the mirror trims. */
    synchronized int[] usage() {
      return new int[] {bytes, limit};
    }

    private void trim() {
      Iterator<Map.Entry<String, Entry>> it = map.entrySet().iterator();
      while (bytes > limit && it.hasNext()) {
        bytes -= it.next().getValue().bodyCost;
        it.remove();
      }
    }
  }

  static void attach(BinaryMessenger messenger) {
    MethodChannel ch =
        new MethodChannel(messenger, "plugins.flutter.io/maplibre_gl/tile_cache");
    synchronized (channelLock) {
      attachedChannels.put(messenger, ch);
    }
    ch.setMethodCallHandler(
        (call, result) -> {
          // Receiving a Dart -> native cache command proves this engine owns
          // the Dart cache handler. Firebase Messaging starts a background
          // FlutterEngine on Android and registers every plugin there too, but
          // that isolate never binds the tile cache. Eagerly replacing the
          // process-wide channel from attach() routed all lookups to that
          // handler-less engine, making every tile wait for GET_TIMEOUT_MS.
          activate(messenger, ch);
          switch (call.method) {
            case "injectTiles":
              {
                List<Map<String, Object>> entries = call.argument("entries");
                if (entries != null) {
                  for (Map<String, Object> row : entries) {
                    Object url = row.get("url");
                    Entry entry =
                        entryFromWire(
                            row.get("data"),
                            (String) row.get("contentType"),
                            (String) row.get("etag"),
                            null);
                    if (url instanceof String && entry != null) {
                      mem.put((String) url, entry);
                    }
                  }
                }
                // Echo the post-injection usage back so Dart can stop injecting
                // when the mirror is near full, without a second round-trip.
                int[] usage = mem.usage();
                Map<String, Object> out = new HashMap<>();
                out.put("used", usage[0]);
                out.put("limit", usage[1]);
                result.success(out);
                return;
              }
            case "filterMissing":
              {
                List<String> urls = call.argument("urls");
                List<String> missing = new ArrayList<>();
                if (urls != null) {
                  for (String url : urls) {
                    if (!mem.contains(url)) missing.add(url);
                  }
                }
                result.success(missing);
                return;
              }
            case "setCacheablePatterns":
              {
                final List<String> patterns = call.argument("patterns");
                synchronized (patternLock) {
                  cacheablePatterns =
                      patterns != null ? new ArrayList<>(patterns) : Collections.emptyList();
                }
                result.success(null);
                return;
              }
            case "evictTiles":
              mem.evict(call.argument("contains"));
              result.success(null);
              return;
            case "setMemoryLimit":
              {
                Integer bytes = call.argument("bytes");
                mem.setLimit(bytes != null ? bytes : DEFAULT_MEMORY_LIMIT);
                List<Number> wireTokens = call.argument("wireTokens");
                dartSupportsTransparentRgbaPngToken =
                    wireTokens != null
                        && wireTokens.stream()
                            .anyMatch(
                                token ->
                                    token != null
                                        && token.intValue()
                                            == TRANSPARENT_RGBA_PNG_WIRE_TOKEN);
                Map<String, Object> capabilities = new HashMap<>();
                capabilities.put(
                    "wireTokens", Collections.singletonList(TRANSPARENT_RGBA_PNG_WIRE_TOKEN));
                result.success(capabilities);
                return;
              }
            case "cancelPendingFetches":
              MapLibreHttpRequestUtil.cancelPendingFetches(call.argument("contains"));
              result.success(null);
              return;
            default:
              result.notImplemented();
          }
        });

    // Pull what Dart wants cached, rather than waiting to be told.
    //
    // Dart binds the cache during app bootstrap — long before the first map
    // widget, and therefore before this plugin is registered — so its push lands
    // on a channel that does not exist yet and is silently lost. The symptom is
    // not a slow cache but a dead one: no URL matches, so nothing is served from
    // or written to Dart at all.
    main.post(
        () ->
            ch.invokeMethod(
                "cacheablePatterns",
                null,
                new MethodChannel.Result() {
                  @Override
                  public void success(@Nullable Object result) {
                    if (!(result instanceof List)) return;
                    activate(messenger, ch);
                    final List<String> patterns = new ArrayList<>();
                    for (Object item : (List<?>) result) {
                      if (item instanceof String) patterns.add((String) item);
                    }
                    synchronized (patternLock) {
                      cacheablePatterns = patterns;
                    }
                  }

                  @Override
                  public void error(String code, String msg, Object details) {}

                  @Override
                  public void notImplemented() {}
                }));
  }

  /** Stops a detached/background engine from retaining the process-wide bridge. */
  static void detach(BinaryMessenger messenger) {
    final MethodChannel detached;
    synchronized (channelLock) {
      detached = attachedChannels.remove(messenger);
      if (activeMessenger == messenger) {
        activeMessenger = null;
        channel = null;
      }
    }
    if (detached != null) detached.setMethodCallHandler(null);
  }

  /** Selects only an engine that has proved its Dart tile handler is alive. */
  private static void activate(BinaryMessenger messenger, MethodChannel candidate) {
    synchronized (channelLock) {
      if (attachedChannels.get(messenger) != candidate) return;
      activeMessenger = messenger;
      channel = candidate;
    }
  }

  @Nullable
  private static MethodChannel activeChannel() {
    synchronized (channelLock) {
      return channel;
    }
  }

  /**
   * Whether {@code url} is one Dart wants to own the caching of.
   *
   * <p>Nothing matches until Dart configures the patterns, which is deliberate: an unconfigured
   * bridge caches nothing rather than guessing at URL shapes the Dart side may not actually store.
   * Guessing is how the two ends drift into a URL native keeps asking about and Dart never has.
   */
  static boolean isCacheableUrl(String url) {
    if (url == null) return false;
    final List<String> patterns;
    synchronized (patternLock) {
      patterns = cacheablePatterns;
    }
    for (String pattern : patterns) {
      if (pattern != null && url.contains(pattern)) return true;
    }
    return false;
  }

  /**
   * Resolves {@code url} from memory, else from Dart via a batched lookup.
   *
   * <p>Returns {@code null} for "not cached, go to the network". Called from OkHttp dispatcher
   * threads only — MapLibre's map thread never reaches here.
   */
  @Nullable
  static Entry get(String url) {
    if (!isCacheableUrl(url)) return null;
    Entry hit = mem.get(url);
    if (hit != null) return hit;
    if (activeChannel() == null) return null;

    Waiter waiter;
    boolean flushNow = false;
    synchronized (batchLock) {
      Waiter existing = pendingGets.get(url);
      if (existing != null) {
        waiter = existing;
      } else {
        waiter = new Waiter();
        pendingGets.put(url, waiter);
        if (pendingGets.size() >= MAX_GET_BATCH) {
          flushNow = true;
        } else if (!getFlushScheduled) {
          getFlushScheduled = true;
          main.postDelayed(MapLibreDartTileBridge::flushGets, GET_WINDOW_MS);
        }
      }
    }
    if (flushNow) flushGets();

    try {
      if (!waiter.latch.await(GET_TIMEOUT_MS, TimeUnit.MILLISECONDS)) return null;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return null;
    }
    return waiter.value.get();
  }

  private static void flushGets() {
    final LinkedHashMap<String, Waiter> batch;
    synchronized (batchLock) {
      getFlushScheduled = false;
      if (pendingGets.isEmpty()) return;
      batch = new LinkedHashMap<>(pendingGets);
      pendingGets.clear();
    }
    MethodChannel ch = activeChannel();
    if (ch == null) {
      for (Waiter waiter : batch.values()) waiter.settle(null);
      return;
    }

    Map<String, Object> args = new HashMap<>();
    args.put("urls", new ArrayList<>(batch.keySet()));
    main.post(
        () ->
            ch.invokeMethod(
                "getBatch",
                args,
                new MethodChannel.Result() {
                  @Override
                  public void success(@Nullable Object result) {
                    Map<?, ?> payload = result instanceof Map ? (Map<?, ?>) result : null;
                    for (Map.Entry<String, Waiter> row : batch.entrySet()) {
                      Entry entry = null;
                      Object raw = payload == null ? null : payload.get(row.getKey());
                      if (raw instanceof Map) {
                        Map<?, ?> hit = (Map<?, ?>) raw;
                        entry =
                            entryFromWire(
                                hit.get("data"),
                                (String) hit.get("contentType"),
                                (String) hit.get("etag"),
                                null);
                        if (entry != null) {
                          mem.put(row.getKey(), entry);
                        }
                      }
                      row.getValue().settle(entry);
                    }
                  }

                  @Override
                  public void error(String code, String msg, Object details) {
                    for (Waiter waiter : batch.values()) waiter.settle(null);
                  }

                  @Override
                  public void notImplemented() {
                    for (Waiter waiter : batch.values()) waiter.settle(null);
                  }
                }));
  }

  /** Records a network-fetched tile: memory now, Dart on the next batch flush. */
  static void put(
      String url,
      byte[] data,
      @Nullable String contentType,
      @Nullable String etag,
      int sourceSize) {
    if (!isCacheableUrl(url) || data == null) return;
    NormalizedPayload payload = normalizeRasterPayload(data, contentType);
    Entry entry = new Entry(payload.data, payload.contentType, etag, sourceSize);
    mem.put(url, entry);
    if (activeChannel() == null) return;

    boolean flushNow = false;
    synchronized (batchLock) {
      pendingPuts.put(url, entry);
      if (pendingPuts.size() >= MAX_PUT_BATCH) {
        flushNow = true;
      } else if (!putFlushScheduled) {
        putFlushScheduled = true;
        main.postDelayed(MapLibreDartTileBridge::flushPuts, PUT_WINDOW_MS);
      }
    }
    if (flushNow) flushPuts();
  }

  private static void flushPuts() {
    final LinkedHashMap<String, Entry> batch;
    synchronized (batchLock) {
      putFlushScheduled = false;
      if (pendingPuts.isEmpty()) return;
      batch = new LinkedHashMap<>(pendingPuts);
      pendingPuts.clear();
    }
    MethodChannel ch = activeChannel();
    if (ch == null) return;

    List<Map<String, Object>> entries = new ArrayList<>(batch.size());
    for (Map.Entry<String, Entry> row : batch.entrySet()) {
      Map<String, Object> item = new HashMap<>();
      item.put("url", row.getKey());
      item.put("data", wireData(row.getValue().data));
      if (row.getValue().contentType != null) item.put("contentType", row.getValue().contentType);
      if (row.getValue().etag != null) item.put("etag", row.getValue().etag);
      if (row.getValue().sourceSize != null) item.put("sourceSize", row.getValue().sourceSize);
      entries.add(item);
    }
    Map<String, Object> args = new HashMap<>();
    args.put("entries", entries);
    main.post(() -> ch.invokeMethod("putBatch", args));
  }
}
