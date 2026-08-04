package org.maplibre.maplibregl;

import androidx.annotation.Nullable;
import org.maplibre.android.module.http.HttpRequestUtil;
import io.flutter.plugin.common.MethodChannel;
import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.concurrent.TimeUnit;
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

    // App interceptor: Dart hit short-circuits. Network interceptor: headers + put.
    builder.addInterceptor(MapLibreHttpRequestUtil::serveFromDart);
    builder.addNetworkInterceptor(MapLibreHttpRequestUtil::applyHeadersAndPut);

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
    if (!MapLibreDartTileBridge.isCacheableUrl(url)) return response;

    // Transparent gzip leaves a plain body; strip Content-Encoding so MapLibre
    // does not try to inflate again (same footgun as the iOS URLProtocol path).
    Response.Builder rebuilt = response.newBuilder().removeHeader("Content-Encoding");
    String cacheControl = response.header("Cache-Control");
    if (cacheControl == null
        || cacheControl.toLowerCase(Locale.US).contains("no-store")) {
      rebuilt.header("Cache-Control", TILE_CACHE_CONTROL);
    }
    if ((response.code() == 200 || response.code() == 404) && response.body() != null) {
      ResponseBody body = response.body();
      long contentLength = body.contentLength();
      if (contentLength >= 0 && contentLength <= MAX_PUT_BYTES) {
        byte[] bytes = body.bytes();
        MapLibreDartTileBridge.put(
            url, bytes, response.header("Content-Type"), response.header("ETag"));
        rebuilt
            .body(ResponseBody.create(bytes, body.contentType()))
            .header("Content-Length", String.valueOf(bytes.length));
      }
    }
    return rebuilt.build();
  }
}
