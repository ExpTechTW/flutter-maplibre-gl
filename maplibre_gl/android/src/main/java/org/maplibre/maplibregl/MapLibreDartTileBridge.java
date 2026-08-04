package org.maplibre.maplibregl;

import android.os.Handler;
import android.os.Looper;
import androidx.annotation.Nullable;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
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

    Entry(byte[] data, @Nullable String contentType, @Nullable String etag) {
      this.data = data;
      this.contentType = contentType;
      this.etag = etag;
    }
  }

  /** Coalescing window for {@code getBatch} — well under a network round-trip. */
  private static final long GET_WINDOW_MS = 6;

  private static final int MAX_GET_BATCH = 64;
  private static final long GET_TIMEOUT_MS = 2500;
  private static final long PUT_WINDOW_MS = 250;
  private static final int MAX_PUT_BATCH = 32;
  private static final int DEFAULT_MEMORY_LIMIT = 2 * 1024 * 1024;

  private static final Object channelLock = new Object();
  private static final Object patternLock = new Object();
  private static List<String> cacheablePatterns = Collections.emptyList();
  private static MethodChannel channel;
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
      if (previous != null) bytes -= previous.data.length;
      bytes += entry.data.length;
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
            bytes -= row.getValue().data.length;
            it.remove();
            break;
          }
        }
      }
    }

    private void trim() {
      Iterator<Map.Entry<String, Entry>> it = map.entrySet().iterator();
      while (bytes > limit && it.hasNext()) {
        bytes -= it.next().getValue().data.length;
        it.remove();
      }
    }
  }

  static void attach(BinaryMessenger messenger) {
    MethodChannel ch =
        new MethodChannel(messenger, "plugins.flutter.io/maplibre_gl/tile_cache");
    ch.setMethodCallHandler(
        (call, result) -> {
          switch (call.method) {
            case "injectTiles":
              {
                List<Map<String, Object>> entries = call.argument("entries");
                if (entries != null) {
                  for (Map<String, Object> row : entries) {
                    Object url = row.get("url");
                    Object data = row.get("data");
                    if (url instanceof String && data instanceof byte[]) {
                      mem.put(
                          (String) url,
                          new Entry(
                              (byte[]) data,
                              (String) row.get("contentType"),
                              (String) row.get("etag")));
                    }
                  }
                }
                result.success(null);
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
                result.success(null);
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
    synchronized (channelLock) {
      channel = ch;
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
                        Object data = hit.get("data");
                        if (data instanceof byte[]) {
                          entry =
                              new Entry(
                                  (byte[]) data,
                                  (String) hit.get("contentType"),
                                  (String) hit.get("etag"));
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
      String url, byte[] data, @Nullable String contentType, @Nullable String etag) {
    if (!isCacheableUrl(url) || data == null) return;
    mem.put(url, new Entry(data, contentType, etag));
    if (activeChannel() == null) return;

    boolean flushNow = false;
    synchronized (batchLock) {
      pendingPuts.put(url, new Entry(data, contentType, etag));
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
      item.put("data", row.getValue().data);
      if (row.getValue().contentType != null) item.put("contentType", row.getValue().contentType);
      if (row.getValue().etag != null) item.put("etag", row.getValue().etag);
      entries.add(item);
    }
    Map<String, Object> args = new HashMap<>();
    args.put("entries", entries);
    main.post(() -> ch.invokeMethod("putBatch", args));
  }
}
