package org.maplibre.maplibregl;

import io.flutter.plugin.common.MethodChannel;
import java.util.List;
import java.util.Map;

/**
 * Back-compat façade — custom headers now rebuild the shared OkHttp client in
 * {@link MapLibreHttpRequestUtil} (which also mounts the tile memory cache).
 */
public class MapLibreCustomHttpInterceptor {
  public static void setCustomHeaders(
      Map<String, String> headers, List<String> filter, MethodChannel.Result result) {
    MapLibreHttpRequestUtil.setCustomHeaders(headers, filter, result);
  }
}
