package org.maplibre.maplibregl;

import android.os.Handler;
import android.os.Looper;
import androidx.annotation.Nullable;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Thin bridge: Dart owns SQLite + metering. Native asks {@code get} before
 * fetch and {@code put} after a tile miss download.
 */
final class MapLibreDartTileBridge {
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

  private static final Object lock = new Object();
  private static MethodChannel channel;
  private static final Handler main = new Handler(Looper.getMainLooper());
  private static final long GET_TIMEOUT_MS = 100;

  private MapLibreDartTileBridge() {}

  static void attach(BinaryMessenger messenger) {
    MethodChannel ch =
        new MethodChannel(messenger, "plugins.flutter.io/maplibre_gl/tile_cache");
    ch.setMethodCallHandler(
        (call, result) -> {
          if ("cancelPendingFetches".equals(call.method)) {
            MapLibreHttpRequestUtil.cancelPendingFetches();
            result.success(null);
          } else {
            result.notImplemented();
          }
        });
    synchronized (lock) {
      channel = ch;
    }
  }

  /** ExpTech immutable tiles only — not glyphs / other {@code *.pbf}. */
  static boolean isTileUrl(String url) {
    if (url == null) return false;
    if (url.contains("/api/v1/map/tiles/")) return true;
    return url.contains("/api/v2/tiles/");
  }

  @Nullable
  static Entry get(String url) {
    if (!isTileUrl(url)) return null;
    final MethodChannel ch;
    synchronized (lock) {
      ch = channel;
    }
    if (ch == null) return null;

    CountDownLatch latch = new CountDownLatch(1);
    AtomicReference<Entry> out = new AtomicReference<>(null);
    main.post(
        () ->
            ch.invokeMethod(
                "get",
                mapOf("url", url),
                new MethodChannel.Result() {
                  @Override
                  public void success(@Nullable Object result) {
                    if (result instanceof Map) {
                      @SuppressWarnings("unchecked")
                      Map<String, Object> m = (Map<String, Object>) result;
                      Object data = m.get("data");
                      if (data instanceof byte[]) {
                        out.set(
                            new Entry(
                                (byte[]) data,
                                (String) m.get("contentType"),
                                (String) m.get("etag")));
                      }
                    }
                    latch.countDown();
                  }

                  @Override
                  public void error(String code, String msg, Object details) {
                    latch.countDown();
                  }

                  @Override
                  public void notImplemented() {
                    latch.countDown();
                  }
                }));
    try {
      latch.await(GET_TIMEOUT_MS, TimeUnit.MILLISECONDS);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    return out.get();
  }

  static void putAsync(
      String url, byte[] data, @Nullable String contentType, @Nullable String etag) {
    if (!isTileUrl(url) || data == null) return;
    final MethodChannel ch;
    synchronized (lock) {
      ch = channel;
    }
    if (ch == null) return;
    Map<String, Object> args = new HashMap<>();
    args.put("url", url);
    args.put("data", data);
    if (contentType != null) args.put("contentType", contentType);
    if (etag != null) args.put("etag", etag);
    main.post(() -> ch.invokeMethod("put", args));
  }

  private static Map<String, Object> mapOf(String k, Object v) {
    Map<String, Object> m = new HashMap<>();
    m.put(k, v);
    return m;
  }
}
