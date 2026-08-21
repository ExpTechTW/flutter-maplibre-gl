package org.maplibre.maplibregl;

import androidx.annotation.Nullable;
import org.maplibre.android.module.http.HttpRequestUtil;
import io.flutter.plugin.common.MethodChannel;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import okhttp3.Call;
import okhttp3.Dispatcher;
import okhttp3.Interceptor;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Protocol;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;

abstract class MapLibreHttpRequestUtil {

  private static Map<String, String> currentHeaders;
  private static List<String> currentFilter;
  private static Integer currentMaxRequests;
  private static Integer currentMaxRequestsPerHost;
  private static final int MAX_PUT_BYTES = 2 * 1024 * 1024;

  /**
   * Freshness advertised for ExpTech tiles.
   *
   * <p>Their URLs are content-addressed — a new radar frame is a new path — so a body can never go
   * stale under its own URL. Saying so lets MapLibre's own ambient database answer repeat requests
   * without issuing a call at all. Without this header MapLibre treats every tile as already
   * expired and re-requests it on every reveal, which is what made a timeline scrub generate
   * traffic for tiles it had just drawn.
   */
  private static final String TILE_CACHE_CONTROL = "public, max-age=604800, immutable";

  private static volatile OkHttpClient client;

  public static void install() {
    try {
      rebuildClient();
    } catch (RuntimeException ignored) {
      // Best-effort at plugin attach.
    }
  }

  /**
   * Drops in-flight / queued OkHttp calls whose URL contains one of {@code needles}; a null or
   * empty list drops everything.
   *
   * <p>Scoping matters: a timeline scrub abandons the frames it swept past, but the basemap and the
   * frame the finger landed on must keep loading — cancelling those too is what made a settle look
   * like a stall.
   */
  public static void cancelPendingFetches(@Nullable List<String> needles) {
    OkHttpClient c = client;
    if (c == null) return;
    if (needles == null || needles.isEmpty()) {
      c.dispatcher().cancelAll();
      return;
    }
    cancelMatching(c.dispatcher().queuedCalls(), needles);
    cancelMatching(c.dispatcher().runningCalls(), needles);
  }

  private static void cancelMatching(List<Call> calls, List<String> needles) {
    for (Call call : calls) {
      String url = call.request().url().toString();
      for (String needle : needles) {
        if (needle != null && url.contains(needle)) {
          call.cancel();
          break;
        }
      }
    }
  }

  public static void setHttpHeaders(Map<String, String> headers, MethodChannel.Result result) {
    currentHeaders = headers;
    try {
      rebuildClient();
      result.success(null);
    } catch (RuntimeException e) {
      result.error("SetHttpHeadersError", e.getMessage(), null);
    }
  }

  public static void setCustomHeaders(
      Map<String, String> headers, List<String> filter, MethodChannel.Result result) {
    currentHeaders = headers;
    currentFilter = filter;
    try {
      rebuildClient();
      result.success(null);
    } catch (RuntimeException e) {
      result.error("SetCustomHeadersError", e.getMessage(), null);
    }
  }

  /** Headers last passed to [setCustomHeaders] / [setHttpHeaders] (empty if none). */
  public static Map<String, String> getCustomHeaders() {
    return currentHeaders != null ? currentHeaders : java.util.Collections.emptyMap();
  }

  public static void setMaxConcurrentRequests(
      Integer maxRequests, Integer maxRequestsPerHost, MethodChannel.Result result) {
    if (maxRequests != null && maxRequests < 1) {
      result.error(
          "InvalidMaxRequests",
          "maxRequests must be >= 1 (got " + maxRequests + ")",
          null);
      return;
    }
    if (maxRequestsPerHost != null && maxRequestsPerHost < 1) {
      result.error(
          "InvalidMaxRequestsPerHost",
          "maxRequestsPerHost must be >= 1 (got " + maxRequestsPerHost + ")",
          null);
      return;
    }
    currentMaxRequests = maxRequests;
    currentMaxRequestsPerHost = maxRequestsPerHost;
    try {
      rebuildClient();
      result.success(null);
    } catch (RuntimeException e) {
      result.error("SetMaxConcurrentRequestsError", e.getMessage(), null);
    }
  }

  private static void rebuildClient() {
    OkHttpClient.Builder builder =
        new OkHttpClient.Builder()
            // Scrub storms: fail fast instead of holding 30s slots for abandoned frames.
            .connectTimeout(8, TimeUnit.SECONDS)
            .readTimeout(8, TimeUnit.SECONDS)
            .writeTimeout(8, TimeUnit.SECONDS)
            .callTimeout(12, TimeUnit.SECONDS);

    Dispatcher dispatcher = new Dispatcher();
    dispatcher.setMaxRequests(currentMaxRequests != null ? currentMaxRequests : 64);
    dispatcher.setMaxRequestsPerHost(
        currentMaxRequestsPerHost != null ? currentMaxRequestsPerHost : 16);
    builder.dispatcher(dispatcher);

    // Both are application interceptors. OkHttp transparently gunzips gzip
    // responses in its BridgeInterceptor, which sits *outside* network
    // interceptors — a network interceptor would read raw gzip bytes, repack
    // them as the body, and (after stripping Content-Encoding) hand MapLibre
    // compressed data it then fails to decode as PBF/bitmap (blank basemap +
    // radar on Android only; iOS's URLSession inflates automatically).
    builder.addInterceptor(MapLibreHttpRequestUtil::serveFromDart);
    builder.addInterceptor(MapLibreHttpRequestUtil::applyHeadersAndPut);

    OkHttpClient built = builder.build();
    client = built;
    HttpRequestUtil.setOkHttpClient(built);
  }

  /** ExpTech tile hit from Dart; miss falls through to a single native fetch. */
  private static Response serveFromDart(Interceptor.Chain chain) throws IOException {
    Request request = chain.request();
    String url = request.url().toString();
    if (!MapLibreDartTileBridge.isCacheableUrl(url)) {
      return chain.proceed(request);
    }

    MapLibreDartTileBridge.Entry hit = MapLibreDartTileBridge.get(url);
    if (hit != null) {
      MediaType mediaType =
          MediaType.parse(
              hit.contentType != null ? hit.contentType : "application/octet-stream");
      Response.Builder rb =
          new Response.Builder()
              .request(request)
              .protocol(Protocol.HTTP_1_1)
              .code(200)
              .message("OK")
              .body(ResponseBody.create(hit.data, mediaType))
              .header(
                  "Content-Type",
                  hit.contentType != null ? hit.contentType : "application/octet-stream")
              .header("Content-Length", String.valueOf(hit.data.length))
              .header("Cache-Control", TILE_CACHE_CONTROL);
      if (hit.etag != null) {
        rb.header("ETag", hit.etag);
      }
      return rb.build();
    }
    return chain.proceed(request);
  }

  private static Response applyHeadersAndPut(Interceptor.Chain chain) throws IOException {
    Request.Builder reqBuilder = chain.request().newBuilder();
    String url = chain.request().url().toString();

    if (currentHeaders != null) {
      boolean shouldApply = currentFilter == null || currentFilter.isEmpty();
      if (!shouldApply && currentFilter != null) {
        for (String pattern : currentFilter) {
          if (Pattern.matches(pattern, url)) {
            shouldApply = true;
            break;
          }
        }
      }
      if (shouldApply) {
        for (Map.Entry<String, String> header : currentHeaders.entrySet()) {
          if (header.getKey() == null || header.getKey().trim().isEmpty()) {
            continue;
          }
          if (header.getValue() == null || header.getValue().trim().isEmpty()) {
            reqBuilder.removeHeader(header.getKey());
          } else {
            reqBuilder.header(header.getKey(), header.getValue());
          }
        }
      }
    }

    Response response = chain.proceed(reqBuilder.build());

    // Always strip Content-Encoding. OkHttp usually decompresses already; when
    // the header survives, MapLibre double-gunzips (same iOS footgun). Do not
    // gate on isCacheableUrl — that is false until Dart registers patterns, and
    // bootstrap often calls setCacheablePatterns before the plugin is attached.
    Response.Builder rebuilt = response.newBuilder().removeHeader("Content-Encoding");
    String cacheControl = response.header("Cache-Control");
    if (cacheControl == null
        || cacheControl.toLowerCase(Locale.US).contains("no-store")) {
      if (MapLibreDartTileBridge.isCacheableUrl(url)) {
        rebuilt.header("Cache-Control", TILE_CACHE_CONTROL);
      }
    }

    if (!MapLibreDartTileBridge.isCacheableUrl(url)) {
      return rebuilt.build();
    }

    // 200 only: a 404 body is the origin's error text, not the asset, and these
    // URLs are served `immutable` so a cached error is never revalidated away.
    if (response.code() == 200 && response.body() != null) {
      ResponseBody body = response.body();
      // Bound by the bytes that are actually here, never by what the server
      // declared. `contentLength()` is -1 whenever the length is unknown: a
      // chunked response, and every response OkHttp's own bridge interceptor
      // transparently gunzipped — it drops Content-Length along with the
      // encoding. The gate used to be `contentLength >= 0 && <= MAX`, so every
      // one of those tiles was skipped here. MapLibre still drew them, because
      // the body is handed back either way, but they never entered `mem` and
      // never reached Dart, and `filterMissing` therefore reported them absent
      // for the rest of the session. A settled timeline frame gates its reveal
      // on exactly that probe, so the frame never became display-ready: on
      // Android the map kept drawing the previous timestamp after every scrub,
      // and the store never learned the tile, so returning re-downloaded it.
      //
      // iOS gates on its buffered body's own length instead
      // (MapLibreHeadersProtocol.swift), which is why only Android stalled.
      //
      // Buffering an unknown-length body is bounded in practice: only
      // `isCacheableUrl` URLs reach this line — tiles, glyph ranges, small
      // images — and the real length is still checked before anything is kept.
      long declared = body.contentLength();
      if (declared <= MAX_PUT_BYTES) {
        byte[] bytes = body.bytes();
        // Double insurance: as an application interceptor the body should
        // already be inflated (OkHttp's bridge gunzips before it reaches us),
        // but if a request ever carries its own Accept-Encoding OkHttp skips
        // that — inflate here so MapLibre never receives gzip framing.
        if ("gzip".equalsIgnoreCase(response.header("Content-Encoding"))) {
          bytes = gunzip(bytes);
        }
        MediaType mediaType = body.contentType();
        // An unknown length can still turn out to be too big to keep. Measure
        // it now that it is buffered, and store only what fits.
        if (bytes.length <= MAX_PUT_BYTES) {
          int sourceSize = bytes.length;
          MapLibreDartTileBridge.NormalizedPayload payload =
              MapLibreDartTileBridge.normalizeRasterPayload(
                  bytes, response.header("Content-Type"));
          bytes = payload.data;
          MapLibreDartTileBridge.put(
              url, bytes, payload.contentType, response.header("ETag"), sourceSize);
          if (payload.repaired) {
            mediaType = MediaType.parse("image/png");
            if (payload.contentType != null) {
              rebuilt.header("Content-Type", payload.contentType);
            }
          }
        }
        // The body was consumed to measure it, so it has to be handed back
        // whether or not it was stored.
        rebuilt
            .body(ResponseBody.create(bytes, mediaType))
            .header("Content-Length", String.valueOf(bytes.length));
      }
    }
    return rebuilt.build();
  }

  private static byte[] gunzip(byte[] compressed) {
    try (GZIPInputStream in = new GZIPInputStream(new ByteArrayInputStream(compressed));
        ByteArrayOutputStream out = new ByteArrayOutputStream()) {
      byte[] buffer = new byte[8192];
      int n;
      while ((n = in.read(buffer)) != -1) {
        out.write(buffer, 0, n);
      }
      return out.toByteArray();
    } catch (IOException e) {
      // A bad gzip stream is worse than none — return the original bytes so
      // the caller still hands MapLibre *something* (it may be plain anyway).
      return compressed;
    }
  }
}
