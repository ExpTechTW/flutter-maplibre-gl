package org.maplibre.maplibregl;

import java.util.ArrayList;
import java.util.List;
import org.maplibre.android.location.CompassEngine;
import org.maplibre.android.location.CompassListener;

/**
 * Passes a heading on only once it has moved far enough to see.
 *
 * <p>Every heading the location component receives redraws the map, and a phone lying on a table
 * still reports a heading that wanders by a fraction of a degree, so the unfiltered sensor keeps
 * the map rendering at full rate with nothing visibly changing.
 */
final class SteadyCompassEngine implements CompassEngine, CompassListener {
  private static final float MIN_CHANGE_DEGREES = 2f;

  private final CompassEngine source;
  private final List<CompassListener> listeners = new ArrayList<>();
  private boolean hasHeading = false;
  private float lastHeading = 0f;

  SteadyCompassEngine(CompassEngine source) {
    this.source = source;
  }

  @Override
  public void addCompassListener(CompassListener listener) {
    if (listeners.isEmpty()) {
      source.addCompassListener(this);
    }
    listeners.add(listener);
  }

  @Override
  public void removeCompassListener(CompassListener listener) {
    listeners.remove(listener);
    if (listeners.isEmpty()) {
      source.removeCompassListener(this);
    }
  }

  @Override
  public float getLastHeading() {
    return hasHeading ? lastHeading : source.getLastHeading();
  }

  @Override
  public int getLastAccuracySensorStatus() {
    return source.getLastAccuracySensorStatus();
  }

  @Override
  public void onCompassChanged(float heading) {
    if (hasHeading) {
      final float difference = Math.abs(((heading - lastHeading) % 360f + 540f) % 360f - 180f);
      if (difference < MIN_CHANGE_DEGREES) {
        return;
      }
    }
    hasHeading = true;
    lastHeading = heading;
    for (int i = 0; i < listeners.size(); i++) {
      listeners.get(i).onCompassChanged(heading);
    }
  }

  @Override
  public void onCompassAccuracyChange(int compassStatus) {
    for (int i = 0; i < listeners.size(); i++) {
      listeners.get(i).onCompassAccuracyChange(compassStatus);
    }
  }
}
