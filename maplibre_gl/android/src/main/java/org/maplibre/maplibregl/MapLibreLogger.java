package org.maplibre.maplibregl;

import android.util.Log;
import androidx.annotation.Nullable;
import org.maplibre.android.log.LoggerDefinition;

/** Keeps actionable MapLibre logs while dropping expected OkHttp cancellation noise. */
final class MapLibreLogger implements LoggerDefinition {
  private static final String HTTP_REQUEST_TAG = "Mbgl-HttpRequest";
  private static final String CANCELLED_REQUEST_PREFIX =
      "Request failed due to a permanent error: Canceled";

  private static boolean isExpectedCancellation(String tag, String message) {
    return HTTP_REQUEST_TAG.equals(tag)
        && message != null
        && message.startsWith(CANCELLED_REQUEST_PREFIX);
  }

  private static void print(
      int priority, String tag, String message, @Nullable Throwable throwable) {
    if (priority == Log.WARN && isExpectedCancellation(tag, message)) return;
    if (throwable == null) {
      Log.println(priority, tag, message);
    } else {
      Log.println(priority, tag, message + '\n' + Log.getStackTraceString(throwable));
    }
  }

  @Override
  public void v(String tag, String message) {
    print(Log.VERBOSE, tag, message, null);
  }

  @Override
  public void v(String tag, String message, Throwable throwable) {
    print(Log.VERBOSE, tag, message, throwable);
  }

  @Override
  public void d(String tag, String message) {
    print(Log.DEBUG, tag, message, null);
  }

  @Override
  public void d(String tag, String message, Throwable throwable) {
    print(Log.DEBUG, tag, message, throwable);
  }

  @Override
  public void i(String tag, String message) {
    print(Log.INFO, tag, message, null);
  }

  @Override
  public void i(String tag, String message, Throwable throwable) {
    print(Log.INFO, tag, message, throwable);
  }

  @Override
  public void w(String tag, String message) {
    print(Log.WARN, tag, message, null);
  }

  @Override
  public void w(String tag, String message, Throwable throwable) {
    print(Log.WARN, tag, message, throwable);
  }

  @Override
  public void e(String tag, String message) {
    print(Log.ERROR, tag, message, null);
  }

  @Override
  public void e(String tag, String message, Throwable throwable) {
    print(Log.ERROR, tag, message, throwable);
  }
}
