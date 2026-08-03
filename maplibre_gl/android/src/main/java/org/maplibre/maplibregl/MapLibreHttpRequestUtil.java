package org.maplibre.maplibregl;

import org.maplibre.android.module.http.HttpRequestUtil;
import io.flutter.plugin.common.MethodChannel;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
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

  public static void install() {
    try {
      rebuildClient();
    } catch (RuntimeException ignored) {
      // Best-effort at plugin attach.
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
    OkHttpClient.Builder builder = new OkHttpClient.Builder();

    if (currentMaxRequests != null || currentMaxRequestsPerHost != null) {
      Dispatcher dispatcher = new Dispatcher();
      if (currentMaxRequests != null) {
        dispatcher.setMaxRequests(currentMaxRequests);
      }
      if (currentMaxRequestsPerHost != null) {
        dispatcher.setMaxRequestsPerHost(currentMaxRequestsPerHost);
      }
      builder.dispatcher(dispatcher);
    }

    // App interceptor: Dart hit short-circuits. Network interceptor: headers + put.
    builder.addInterceptor(MapLibreHttpRequestUtil::serveFromDart);
    builder.addNetworkInterceptor(MapLibreHttpRequestUtil::applyHeadersAndPut);

    HttpRequestUtil.setOkHttpClient(builder.build());
  }

  /** ExpTech tile hit from Dart; miss falls through to a single native fetch. */
  private static Response serveFromDart(Interceptor.Chain chain) throws IOException {
    Request request = chain.request();
    String url = request.url().toString();
    if (!MapLibreDartTileBridge.isTileUrl(url)) {
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
              .header("Content-Length", String.valueOf(hit.data.length));
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
    if (MapLibreDartTileBridge.isTileUrl(url)
        && (response.code() == 200 || response.code() == 404)
        && response.body() != null) {
      ResponseBody body = response.body();
      long contentLength = body.contentLength();
      if (contentLength >= 0 && contentLength <= MAX_PUT_BYTES) {
        byte[] bytes = body.bytes();
        MapLibreDartTileBridge.putAsync(
            url, bytes, response.header("Content-Type"), response.header("ETag"));
        return response.newBuilder().body(ResponseBody.create(bytes, body.contentType())).build();
      }
    }
    return response;
  }
}
