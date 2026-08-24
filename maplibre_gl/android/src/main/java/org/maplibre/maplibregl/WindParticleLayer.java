package org.maplibre.maplibregl;

import android.opengl.EGL14;
import android.opengl.EGLContext;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.util.Log;
import androidx.annotation.Nullable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import org.maplibre.android.maps.MapView;
import org.maplibre.android.maps.renderer.MapRenderer;
import org.maplibre.android.maps.renderer.surfaceview.MapLibreSurfaceView;

/**
 * GPU wind-particle field drawn inside the map's own GL surface.
 *
 * <h3>Why this is not a Flutter widget</h3>
 *
 * It used to be one: a {@code Ticker} stepping a CPU simulation into a
 * {@code CustomPaint} laid over the platform view. On Android that is a memory
 * leak with a stopwatch on it. Flutter 3.47's HCPP platform-view mode allocates
 * a full-screen {@code AHardwareBuffer} for every Flutter frame presented above
 * a platform view and — because {@code AHBTexturePoolVK}'s cap and GC were
 * deleted upstream and its only recycle path is a SurfaceControl transaction
 * callback the engine deliberately does not fire — never gives one back.
 * Measured on a Pixel 9: 394 MB to 8042 MB of GPU memory in sixteen seconds,
 * then the process was killed. An animated overlay presents a frame every
 * vsync, so it hit that ceiling faster than anything else in the app.
 *
 * <p>Drawing here instead means the particles cost the map one more pass and
 * cost Flutter nothing at all: no Flutter frame is produced, so no buffer is
 * allocated, so nothing leaks. It is also simply the right place — the field is
 * map content, and it now pans, zooms and rotates in the same pass as the map
 * it describes rather than chasing it a frame behind.
 *
 * <h3>Where the draw happens</h3>
 *
 * {@link MapLibreSurfaceView}'s render thread keeps a one-shot
 * {@code finishDrawingRunnable} that {@code GLThread.guardedRun} invokes
 * between the map's {@code onDrawFrame} and {@code eglSwapBuffers} — verified
 * in the bytecode of the linked AAR (android-sdk-opengl 13.3.0), offsets 850,
 * 860 and 872. That is the one place reachable through public API where the GL
 * context is current, the map is fully drawn, and the frame has not yet been
 * presented. {@link #arm()} re-registers it every frame; the map is switched to
 * {@link MapRenderer.RenderingRefreshMode#CONTINUOUS} while running so those
 * frames keep coming.
 *
 * <p>The alternative, {@code CustomLayer}, takes a raw {@code CustomLayerHost*}
 * and so requires an NDK component and an ABI contract with a header that has
 * already drifted upstream ({@code mbgl::} to {@code mln::}, plus a new virtual
 * inserted between {@code initialize} and {@code render}). This path needs
 * neither.
 *
 * <h3>What it does not do</h3>
 *
 * Nothing here parses WND1. The Dart side already has a tested parser, so it
 * hands over the raw bytes plus the header it already read; this class only
 * uploads. Likewise the tuning curves arrive as their two endpoints and are
 * interpolated here with the same rules Dart uses, so the numbers have exactly
 * one home.
 */
final class WindParticleLayer {

  private static final String TAG = "WindParticleLayer";

  /**
   * Particle-state texture edge, fixed for the life of the layer.
   *
   * <p>6400 is the largest count the zoom curve asks for, so the texture is
   * allocated once at that size and zoom only changes how many of its texels
   * are drawn. Resizing instead would reseed the whole field on every zoom
   * threshold — a visible flash — and would throw away particles that are
   * already correctly advected.
   */
  private static final int STATE_EDGE = 80;

  /** The frame rate the advection curves were tuned at. See {@link #setPlaying}. */
  private static final int TARGET_FPS = 60;

  // ---------------------------------------------------------------- shaders

  /**
   * Full-screen triangle strip in the same 0..1 space the reference uses.
   *
   * <p>The double flip (here and in {@link #FRAG_SCREEN}) is inherited from the
   * reference implementation rather than simplified away: it is what makes the
   * fade pass sample the texel it is about to overwrite.
   */
  private static final String VERT_QUAD =
      "precision mediump float;\n"
          + "attribute vec2 a_pos;\n"
          + "varying vec2 v_tex;\n"
          + "void main() {\n"
          + "  v_tex = a_pos;\n"
          + "  gl_Position = vec4(1.0 - 2.0 * a_pos, 0.0, 1.0);\n"
          + "}";

  private static final String FRAG_SCREEN =
      "precision mediump float;\n"
          + "uniform sampler2D u_screen;\n"
          + "uniform float u_opacity;\n"
          + "varying vec2 v_tex;\n"
          + "void main() {\n"
          // Multiply in premultiplied space so the fade cannot drift to grey.
          + "  vec4 c = texture2D(u_screen, 1.0 - v_tex);\n"
          + "  gl_FragColor = vec4(floor(255.0 * c * u_opacity) / 255.0);\n"
          + "}";

  /**
   * Shared projection: field space to screen pixels.
   *
   * <p>A literal port of {@code projectLatLng} in {@code wind_particle_sim.dart},
   * including the bearing sign and the world-x fold. The reference
   * implementation projects from an axis-aligned lon/lat box instead, which is
   * only correct while the map points north — this map rotates.
   *
   * <p>The fold matters for more than the antimeridian: a particle's longitude
   * is {@code lon0 + x * 360}, and ECMWF's grid starts at 180, so every one of
   * its particles arrives in [180, 540).
   */
  private static final String PROJECT_GLSL =
      "const float PI = 3.141592653589793;\n"
          + "uniform float u_world;\n" // 512 * 2^zoom
          + "uniform vec2 u_center;\n" // camera centre, world px
          + "uniform vec2 u_half;\n" // viewport half size, px
          + "uniform vec2 u_rot;\n" // cos(-bearing), sin(-bearing)
          + "uniform float u_lon0;\n"
          + "uniform vec2 u_lat;\n" // lat0, latitude span
          + "float mercY(float lat) {\n"
          + "  float phi = clamp(lat, -85.051129, 85.051129) * PI / 180.0;\n"
          + "  return (1.0 - log(tan(PI / 4.0 + phi / 2.0)) / PI) / 2.0;\n"
          + "}\n"
          + "float mercLat(float y) {\n"
          + "  return degrees(2.0 * atan(exp(PI * (1.0 - 2.0 * y))) - PI / 2.0);\n"
          + "}\n"
          + "vec2 projectField(vec2 p) {\n"
          + "  float lon = u_lon0 + p.x * 360.0;\n"
          + "  float lat = u_lat.x + p.y * u_lat.y;\n"
          + "  float wx = (lon + 180.0) / 360.0 * u_world;\n"
          + "  float wy = mercY(lat) * u_world;\n"
          + "  float dx = wx - u_center.x;\n"
          + "  dx = dx - u_world * floor(dx / u_world + 0.5);\n"
          + "  float dy = wy - u_center.y;\n"
          + "  return vec2(u_half.x + dx * u_rot.x - dy * u_rot.y,\n"
          + "              u_half.y + dx * u_rot.y + dy * u_rot.x);\n"
          + "}\n";

  private static final String VERT_PARTICLE =
      "precision highp float;\n"
          + "attribute float a_index;\n"
          + "uniform sampler2D u_particles;\n"
          + "uniform float u_particles_res;\n"
          + "uniform float u_point_size;\n"
          + "varying vec2 v_pos;\n"
          + PROJECT_GLSL
          + "void main() {\n"
          + "  vec4 c = texture2D(u_particles, vec2(\n"
          + "    fract(a_index / u_particles_res),\n"
          + "    floor(a_index / u_particles_res) / u_particles_res));\n"
          // Two bytes per axis, for sub-pixel precision.
          + "  v_pos = vec2(c.r / 255.0 + c.b, c.g / 255.0 + c.a);\n"
          + "  vec2 px = projectField(v_pos);\n"
          + "  gl_PointSize = u_point_size;\n"
          + "  gl_Position = vec4(px.x / u_half.x - 1.0, 1.0 - px.y / u_half.y, 0.0, 1.0);\n"
          + "}";

  /**
   * Point sprite: white, brighter where the air is faster, and round.
   *
   * <p>The roundness is not decoration. {@code GL_POINTS} are square, and the
   * Flutter painter this replaces drew with {@code StrokeCap.round}; leaving
   * them square changes the texture of the whole field from a weave of strokes
   * into a grid of tiles.
   */
  private static final String FRAG_PARTICLE =
      "precision mediump float;\n"
          + "uniform sampler2D u_wind;\n"
          + "uniform vec2 u_wind_min;\n"
          + "uniform vec2 u_wind_max;\n"
          + "uniform float u_speed_scale;\n"
          + "varying vec2 v_pos;\n"
          + "void main() {\n"
          + "  vec2 d = gl_PointCoord - vec2(0.5);\n"
          + "  float r = 1.0 - smoothstep(0.35, 0.5, length(d));\n"
          + "  if (r <= 0.0) discard;\n"
          + "  vec2 v = mix(u_wind_min, u_wind_max, texture2D(u_wind, v_pos).ra);\n"
          + "  float t = clamp(length(v) / u_speed_scale, 0.0, 1.0);\n"
          + "  gl_FragColor = vec4(vec3(1.0), (0.35 + 0.55 * t) * r);\n"
          + "}";

  private static final String FRAG_UPDATE =
      "precision highp float;\n"
          + "uniform sampler2D u_particles;\n"
          + "uniform sampler2D u_wind;\n"
          + "uniform vec2 u_wind_res;\n"
          + "uniform vec2 u_wind_min;\n"
          + "uniform vec2 u_wind_max;\n"
          + "uniform float u_rand_seed;\n"
          + "uniform float u_speed_factor;\n"
          + "uniform float u_drop_rate;\n"
          + "uniform float u_speed_scale;\n"
          + "uniform vec2 u_density;\n"
          + "uniform vec4 u_view_world;\n" // x0, y0, x1, y1 in world px
          + "varying vec2 v_tex;\n"
          + PROJECT_GLSL
          // Bilinear by hand: the wind texture is NEAREST so paths stay smooth
          // without relying on driver filtering of a packed texture.
          + "vec2 lookup_wind(const vec2 uv) {\n"
          + "  vec2 px = 1.0 / u_wind_res;\n"
          + "  vec2 vc = (floor(uv * u_wind_res)) * px;\n"
          + "  vec2 f = fract(uv * u_wind_res);\n"
          + "  vec2 tl = texture2D(u_wind, vc).ra;\n"
          + "  vec2 tr = texture2D(u_wind, vc + vec2(px.x, 0)).ra;\n"
          + "  vec2 bl = texture2D(u_wind, vc + vec2(0, px.y)).ra;\n"
          + "  vec2 br = texture2D(u_wind, vc + px).ra;\n"
          + "  return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);\n"
          + "}\n"
          + "const vec3 rand_constants = vec3(12.9898, 78.233, 4375.85453);\n"
          + "float rand(const vec2 co) {\n"
          + "  float t = dot(rand_constants.xy, co);\n"
          + "  return fract(sin(t) * (rand_constants.z + t));\n"
          + "}\n"
          // Uniform draw from the world rectangle the viewport covers. Seeding
          // globally leaves the flow empty: at zoom 5 the view is a hundredth
          // of the world, so a hundredth of the particles are on screen and the
          // rest are animated for nobody.
          + "vec2 respawnAt(const vec2 seed) {\n"
          + "  float wx = mix(u_view_world.x, u_view_world.z, rand(seed + 1.3));\n"
          + "  float wy = mix(u_view_world.y, u_view_world.w, rand(seed + 2.1));\n"
          + "  float lon = wx / u_world * 360.0 - 180.0;\n"
          + "  float lat = mercLat(wy / u_world);\n"
          + "  return vec2(fract((lon - u_lon0) / 360.0), (lat - u_lat.x) / u_lat.y);\n"
          + "}\n"
          + "float speedAt(const vec2 pos) {\n"
          + "  return length(mix(u_wind_min, u_wind_max, texture2D(u_wind, pos).ra));\n"
          + "}\n"
          // Fewer particles where the wind is strong, more where it is weak.
          // A fast particle covers ground before its trail fades, so a handful
          // already draw a dense weave; keeping the even count there smears the
          // region white and hides both the colour beneath and the direction.
          + "float densityWeight(const float speed) {\n"
          + "  return mix(u_density.x, u_density.y,\n"
          + "             smoothstep(0.0, 0.6 * u_speed_scale, speed));\n"
          + "}\n"
          // Rejection, not best-of-N. Choosing the best of three cannot reach
          // the asked-for contrast: when all three land in calm air it returns
          // calm whatever the weights say. Rejection's equilibrium density is
          // exactly the acceptance probability. A rejected particle stays put
          // and tries again next frame.
          + "vec2 respawn(const vec2 seed, const vec2 current) {\n"
          + "  vec2 cand = respawnAt(seed);\n"
          + "  float accept = densityWeight(speedAt(cand)) / max(u_density.x, u_density.y);\n"
          + "  return rand(seed + 11.3) < accept ? cand : current;\n"
          + "}\n"
          + "bool inView(const vec2 pos) {\n"
          + "  vec2 px = projectField(pos);\n"
          + "  vec2 t = px / (2.0 * u_half);\n"
          + "  return t.x > -0.1 && t.x < 1.1 && t.y > -0.1 && t.y < 1.1;\n"
          + "}\n"
          + "void main() {\n"
          + "  vec4 c = texture2D(u_particles, v_tex);\n"
          + "  vec2 pos = vec2(c.r / 255.0 + c.b, c.g / 255.0 + c.a);\n"
          + "  vec2 v = mix(u_wind_min, u_wind_max, lookup_wind(pos));\n"
          // Degrees of longitude per metre grow as the cosine of latitude
          // shrinks, so a particle near the pole must move further in field
          // space for the same ground speed. Without this the flow visibly
          // stalls at high latitude, which is a property of the grid rather
          // than of the weather.
          + "  float distortion = cos(radians(u_lat.x + pos.y * u_lat.y));\n"
          + "  vec2 offset = vec2(v.x / distortion, -v.y) * u_speed_factor;\n"
          // Longitude is cyclic; latitude is not. Wrapping both — which a plain
          // fract() on the pair does — teleports a particle drifting off the
          // north pole to the south one, against the flow it was following.
          + "  pos = vec2(fract(1.0 + pos.x + offset.x), pos.y + offset.y);\n"
          + "  vec2 seed = (pos + v_tex) * u_rand_seed;\n"
          + "  bool onGrid = pos.y >= 0.0 && pos.y <= 1.0;\n"
          + "  float leave = (onGrid && inView(pos)) ? 0.0 : 1.0;\n"
          + "  pos = mix(pos, respawn(seed, pos),\n"
          + "            max(step(1.0 - u_drop_rate, rand(seed)), leave));\n"
          + "  gl_FragColor = vec4(fract(pos * 255.0), floor(pos * 255.0) / 255.0);\n"
          + "}";

  // ------------------------------------------------------------------ state

  private final MapView mapView;

  /** Set from the platform thread, read on the GL thread. */
  private volatile CameraSnapshot camera;

  private volatile FieldUpload pendingField;
  private volatile Tuning tuning = Tuning.defaults();
  private volatile boolean playing;
  private volatile boolean released;

  @Nullable private MapLibreSurfaceView renderView;
  @Nullable private EGLContext glContext;

  private final Runnable drawRunnable = this::onFinishDrawing;
  private final Runnable armRunnable = this::arm;

  /**
   * Re-arms the draw callback from off the render thread.
   *
   * <p>Not an optimisation — a requirement. {@code requestRenderAndNotify}
   * opens with {@code if (Thread.currentThread() == this) return;}, so arming
   * from inside the draw callback (which runs on the render thread) is silently
   * dropped. The first frame drew and no other ever did, which looks exactly
   * like a shader that failed to compile.
   *
   * <p>Its own thread rather than the main looper: on Android the platform and
   * UI threads are one, and posting there every vsync adds traffic to the
   * thread this whole design exists to keep out of the loop.
   */
  @Nullable private HandlerThread pumpThread;

  @Nullable private Handler pumpHandler;

  // GL objects, only ever touched on the GL thread.
  private int quadProgram;
  private int drawProgram;
  private int updateProgram;
  private int quadBuffer;
  private int indexBuffer;
  private int framebuffer;
  private final int[] stateTex = new int[2];
  private final int[] screenTex = new int[2];
  private int windTex;
  private int stateFront;
  private int screenFront;
  private int indexCount;
  private int screenWidth;
  private int screenHeight;
  private boolean fieldReady;
  private int windWidth;
  private int windHeight;
  private float[] windRange = new float[4];
  private float fieldLat0;
  private float fieldLatSpan;
  private float fieldLon0;
  private long frame;

  /** The camera the surviving trail pixels were drawn for.
   *
   * The trail buffers live in SCREEN space, so the moment the camera moves
   * every pixel in them is anchored to the wrong place on the map. The Flutter
   * overlay this replaces dropped its buffer on any camera change; losing that
   * in the port is what made pans and pinches smear the field behind the map
   * until the fade caught up — and made the whole layer look displaced by
   * exactly the distance travelled since the streaks were laid down.
   */
  private boolean haveTrailCam;
  private double trailZoom;
  private double trailLat;
  private double trailLng;
  private double trailBearing;

  /** Set when the trails must be discarded: camera moved, or a new field landed. */
  private boolean trailsDirty = true;

  /** Camera + viewport for one frame, snapshotted so the GL thread never blocks. */
  static final class CameraSnapshot {
    final double centerLat;
    final double centerLng;
    final double zoom;
    final double bearing;

    /** Viewport in **physical** pixels — what the GL viewport and the trail
     * textures are sized in. */
    final int width;

    final int height;

    /**
     * Display density, the bridge between the two pixel spaces in play.
     *
     * <p>MapLibre's zoom is defined against logical pixels: at zoom z the world
     * is {@code 512 * 2^z} of them. An Android {@code View}'s width is physical.
     * Projecting one against the other scales the whole field by the density —
     * 2.625 on a Pixel 9 — which looks like the wind blowing over the wrong
     * part of the map rather than like a unit bug.
     */
    final double density;

    CameraSnapshot(
        double centerLat,
        double centerLng,
        double zoom,
        double bearing,
        int width,
        int height,
        double density) {
      this.centerLat = centerLat;
      this.centerLng = centerLng;
      this.zoom = zoom;
      this.bearing = bearing;
      this.width = width;
      this.height = height;
      this.density = density <= 0 ? 1 : density;
    }

    /** Viewport half-size in logical pixels — the space {@code u_world} is in. */
    double halfWidth() {
      return width / density / 2;
    }

    double halfHeight() {
      return height / density / 2;
    }
  }

  /** One WND1 payload waiting to be uploaded on the GL thread. */
  static final class FieldUpload {
    final byte[] bytes;
    final int planeOffset;
    final int width;
    final int height;
    final float lat0;
    final float lon0;
    final float dLat;
    final float dLon;
    final float uMin;
    final float uMax;
    final float vMin;
    final float vMax;

    FieldUpload(
        byte[] bytes,
        int planeOffset,
        int width,
        int height,
        float lat0,
        float lon0,
        float dLat,
        float dLon,
        float uMin,
        float uMax,
        float vMin,
        float vMax) {
      this.bytes = bytes;
      this.planeOffset = planeOffset;
      this.width = width;
      this.height = height;
      this.lat0 = lat0;
      this.lon0 = lon0;
      this.dLat = dLat;
      this.dLon = dLon;
      this.uMin = uMin;
      this.uMax = uMax;
      this.vMin = vMin;
      this.vMax = vMax;
    }
  }

  /**
   * The zoom-dependent curves, as their endpoints.
   *
   * <p>Sent whole rather than evaluated per frame in Dart: the numbers keep one
   * home (the Dart simulation, which stays as the numeric oracle) and nothing
   * has to cross the channel while a finger is on the map.
   */
  static final class Tuning {
    double zoomLo = 3;
    double zoomHi = 7;
    double particlesLo = 6400;
    double particlesHi = 1024;
    double pointSizeLo = 1.5;
    double pointSizeHi = 1.8;
    double speedFactorLo = 0.2;
    double speedFactorHi = 0.0151;
    double fadeOpacityLo = 0.95;
    double fadeOpacityHi = 0.945;
    double dropRate = 0.011;
    double densityCalm = 0.5;
    double densityStrong = 5.5;
    double speedScale = 32;
    double pixelRatio = 1;

    static Tuning defaults() {
      return new Tuning();
    }

    private double fraction(double zoom) {
      double f = (zoom - zoomLo) / (zoomHi - zoomLo);
      return f < 0 ? 0 : (f > 1 ? 1 : f);
    }

    private double lin(double lo, double hi, double zoom) {
      double f = fraction(zoom);
      return lo * (1 - f) + hi * f;
    }

    private double log(double lo, double hi, double zoom) {
      double f = fraction(zoom);
      return Math.exp(Math.log(lo) * (1 - f) + Math.log(hi) * f);
    }

    /** Matches {@code particleCountFor}: rounded to a square. */
    int particleCount(double zoom) {
      int edge = (int) Math.max(1, Math.round(Math.sqrt(log(particlesLo, particlesHi, zoom))));
      return edge * edge;
    }

    float pointSize(double zoom) {
      return (float) (lin(pointSizeLo, pointSizeHi, zoom) * pixelRatio);
    }

    /** Matches {@code fieldStepFor}: the 1e-4 scale is part of the curve. */
    float speedFactor(double zoom) {
      return (float) (0.0001 * log(speedFactorLo, speedFactorHi, zoom));
    }

    float fadeOpacity(double zoom) {
      return (float) lin(fadeOpacityLo, fadeOpacityHi, zoom);
    }
  }

  WindParticleLayer(MapView mapView) {
    this.mapView = mapView;
  }

  // --------------------------------------------------------------- lifecycle

  /**
   * Attaches to the map's render thread. Returns false when the map is not on
   * the SurfaceView renderer, which is the only one that exposes this hook.
   */
  boolean attach() {
    android.view.View view = mapView.getRenderView();
    if (!(view instanceof MapLibreSurfaceView)) {
      Log.w(
          TAG,
          "render view is "
              + (view == null ? "null" : view.getClass().getName())
              + ", not MapLibreSurfaceView — wind particles need the SurfaceView renderer "
              + "(textureMode must stay off)");
      return false;
    }
    renderView = (MapLibreSurfaceView) view;
    HandlerThread thread = new HandlerThread("wind-particle-pump", Process.THREAD_PRIORITY_DISPLAY);
    thread.start();
    pumpThread = thread;
    pumpHandler = new Handler(thread.getLooper());
    return true;
  }

  void setPlaying(boolean value) {
    if (playing == value || released) {
      return;
    }
    playing = value;
    if (value) {
      // CONTINUOUS is what keeps frames coming; without it the map only redraws
      // when something changed and the animation stops between gestures.
      mapView.setRenderingRefreshMode(MapRenderer.RenderingRefreshMode.CONTINUOUS);
      // The integrator advances once per FRAME, not per second, so the frame
      // rate is part of the tuning. Left uncapped the render thread free-runs —
      // measured at 237 fps on a 120 Hz panel — and the wind blows at four
      // times the speed it was tuned for. 60 is the rate the curves in
      // wind_particle_sim.dart were fitted at.
      mapView.setMaximumFps(TARGET_FPS);
      Log.i(TAG, "playing: continuous rendering on, first arm from the platform thread");
      arm();
    } else {
      mapView.setRenderingRefreshMode(MapRenderer.RenderingRefreshMode.WHEN_DIRTY);
    }
  }

  void setCamera(CameraSnapshot snapshot) {
    camera = snapshot;
  }

  void setTuning(Tuning value) {
    tuning = value;
  }

  void setField(FieldUpload upload) {
    pendingField = upload;
    // A new forecast is new weather: streaks laid down by the previous one do
    // not describe what is about to animate.
    trailsDirty = true;
    arm();
  }

  /**
   * Releases GL objects on the render thread, then stops.
   *
   * <p>Queued rather than run here: {@code release} arrives on the platform
   * thread, and deleting a texture from it either does nothing (no current
   * context) or corrupts another context's state.
   */
  void release() {
    if (released) {
      return;
    }
    released = true;
    playing = false;
    Handler handler = pumpHandler;
    if (handler != null) {
      handler.removeCallbacks(armRunnable);
    }
    pumpHandler = null;
    HandlerThread thread = pumpThread;
    if (thread != null) {
      thread.quitSafely();
      pumpThread = null;
    }
    MapLibreSurfaceView view = renderView;
    if (view != null) {
      view.queueEvent(this::deleteGlObjects);
      mapView.setRenderingRefreshMode(MapRenderer.RenderingRefreshMode.WHEN_DIRTY);
    }
    renderView = null;
  }

  /**
   * Re-registers the one-shot draw callback. **Must not be called from the
   * render thread** — see [pumpHandler].
   *
   * <p>{@code finishDrawingRunnable} is consumed and cleared by the frame that
   * runs it, so every frame has to re-arm; missing one costs that frame's
   * particles, not correctness. Arming while one is still pending *chains*
   * rather than replaces (MapLibre wraps the old and the new in a lambda that
   * runs both), so the pump posts one at a time and the chain cannot grow.
   */
  private void arm() {
    MapLibreSurfaceView view = renderView;
    if (view == null || released) {
      return;
    }
    view.surfaceRedrawNeededAsync(null, drawRunnable);
  }

  // ------------------------------------------------------------------ render

  private void onFinishDrawing() {
    if (released) {
      return;
    }
    try {
      drawFrame();
    } catch (RuntimeException e) {
      // Never let this escape: it runs inside the map's render loop, and an
      // exception here takes the map down with it.
      Log.e(TAG, "wind particle frame failed", e);
    } finally {
      // Through the pump, never directly: see [pumpHandler]. This runs on the
      // render thread, where arm() is a no-op.
      Handler handler = pumpHandler;
      if (playing && handler != null) {
        handler.post(armRunnable);
      }
    }
  }

  private void drawFrame() {
    CameraSnapshot cam = camera;
    if (cam == null || cam.width <= 0 || cam.height <= 0) {
      return;
    }
    if (contextLost()) {
      deleteGlObjects();
      trailsDirty = true;
    }
    if (quadProgram == 0 && !createGlObjects()) {
      return;
    }
    FieldUpload upload = pendingField;
    if (upload != null) {
      pendingField = null;
      uploadField(upload);
    }
    if (!fieldReady) {
      return;
    }
    // A paused layer draws nothing at all: compositing the frozen trails on
    // every interaction redraw painted ghosts anchored to a camera that has
    // since moved. Consuming the pending field above keeps it warm for the
    // next play; resuming clears the trails because the camera is compared
    // per drawn frame, and any movement during the pause marks them dirty.
    if (!playing) {
      return;
    }
    ensureScreenTextures(cam.width, cam.height);

    // Screen-space trails are only valid for the camera they were laid down
    // with. Sub-pixel thresholds: below them the offset is invisible, above
    // them every streak in the buffer points somewhere nobody sees.
    boolean cameraMoved = haveTrailCam
        && (Math.abs(cam.zoom - trailZoom) > 1e-4
            || Math.abs(cam.centerLat - trailLat) > 1e-6
            || Math.abs(cam.centerLng - trailLng) > 1e-6
            || Math.abs(cam.bearing - trailBearing) > 0.05);
    if (cameraMoved) {
      trailsDirty = true;
    }
    haveTrailCam = true;
    trailZoom = cam.zoom;
    trailLat = cam.centerLat;
    trailLng = cam.centerLng;
    trailBearing = cam.bearing;

    GlState saved = GlState.capture();
    try {
      renderPasses(cam, saved);
    } finally {
      saved.restore();
    }
  }

  /**
   * True when the EGL context has been replaced since the objects were made.
   *
   * <p>{@code preserveEGLContextOnPause} is not set on this render thread, so a
   * background/foreground round trip can destroy the context and every name we
   * hold with it. There is no callback for that on this path, so the check is
   * done by hand each frame — it is one {@code eglGetCurrentContext} call.
   */
  private boolean contextLost() {
    EGLContext current = EGL14.eglGetCurrentContext();
    if (glContext == null || quadProgram == 0) {
      glContext = current;
      return false;
    }
    return !glContext.equals(current);
  }

  private void renderPasses(CameraSnapshot cam, GlState target) {
    Tuning t = tuning;
    int count = t.particleCount(cam.zoom);
    if (count > indexCount) {
      count = indexCount;
    }

    // --- update pass: advect every particle into the back state texture
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
    GLES20.glFramebufferTexture2D(
        GLES20.GL_FRAMEBUFFER,
        GLES20.GL_COLOR_ATTACHMENT0,
        GLES20.GL_TEXTURE_2D,
        stateTex[1 - stateFront],
        0);
    GLES20.glViewport(0, 0, STATE_EDGE, STATE_EDGE);
    GLES20.glDisable(GLES20.GL_BLEND);
    GLES20.glDisable(GLES20.GL_DEPTH_TEST);
    GLES20.glDisable(GLES20.GL_STENCIL_TEST);
    GLES20.glUseProgram(updateProgram);
    bindQuad(updateProgram);
    bindTexture(GLES20.GL_TEXTURE0, windTex, updateProgram, "u_wind", 0);
    bindTexture(GLES20.GL_TEXTURE1, stateTex[stateFront], updateProgram, "u_particles", 1);
    uniform2f(updateProgram, "u_wind_res", windWidth, windHeight);
    uniform2f(updateProgram, "u_wind_min", windRange[0], windRange[2]);
    uniform2f(updateProgram, "u_wind_max", windRange[1], windRange[3]);
    uniform1f(updateProgram, "u_rand_seed", (float) Math.random());
    uniform1f(updateProgram, "u_speed_factor", t.speedFactor(cam.zoom));
    uniform1f(updateProgram, "u_drop_rate", (float) t.dropRate);
    uniform1f(updateProgram, "u_speed_scale", (float) t.speedScale);
    uniform2f(updateProgram, "u_density", (float) t.densityCalm, (float) t.densityStrong);
    setProjection(updateProgram, cam);
    setViewWorld(updateProgram, cam);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    stateFront = 1 - stateFront;

    // --- fade pass: last frame's streaks, dimmed, into the back screen texture
    GLES20.glFramebufferTexture2D(
        GLES20.GL_FRAMEBUFFER,
        GLES20.GL_COLOR_ATTACHMENT0,
        GLES20.GL_TEXTURE_2D,
        screenTex[1 - screenFront],
        0);
    GLES20.glViewport(0, 0, screenWidth, screenHeight);
    if (trailsDirty) {
      // Both textures, not just the one about to be drawn into — otherwise the
      // swap hands the ghosts to the other buffer and they come right back.
      for (int i = 0; i < 2; i++) {
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER,
            GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D,
            screenTex[i],
            0);
        GLES20.glClearColor(0f, 0f, 0f, 0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
      }
      trailsDirty = false;
    } else {
      GLES20.glUseProgram(quadProgram);
      bindQuad(quadProgram);
      bindTexture(GLES20.GL_TEXTURE2, screenTex[screenFront], quadProgram, "u_screen", 2);
      uniform1f(quadProgram, "u_opacity", t.fadeOpacity(cam.zoom));
      GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    }

    // --- draw pass: this frame's particles on top of the faded streaks
    GLES20.glEnable(GLES20.GL_BLEND);
    GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
    GLES20.glUseProgram(drawProgram);
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, indexBuffer);
    int index = GLES20.glGetAttribLocation(drawProgram, "a_index");
    GLES20.glEnableVertexAttribArray(index);
    GLES20.glVertexAttribPointer(index, 1, GLES20.GL_FLOAT, false, 0, 0);
    bindTexture(GLES20.GL_TEXTURE0, windTex, drawProgram, "u_wind", 0);
    bindTexture(GLES20.GL_TEXTURE1, stateTex[stateFront], drawProgram, "u_particles", 1);
    uniform1f(drawProgram, "u_particles_res", STATE_EDGE);
    uniform1f(drawProgram, "u_point_size", t.pointSize(cam.zoom));
    uniform2f(drawProgram, "u_wind_min", windRange[0], windRange[2]);
    uniform2f(drawProgram, "u_wind_max", windRange[1], windRange[3]);
    uniform1f(drawProgram, "u_speed_scale", (float) t.speedScale);
    setProjection(drawProgram, cam);
    GLES20.glDrawArrays(GLES20.GL_POINTS, 0, count);
    screenFront = 1 - screenFront;

    // --- composite: the accumulated streaks onto the map
    //
    // Back to whatever the map was drawing into, not a hardcoded 0. It is 0 on
    // this renderer today, but a renderer that draws through its own FBO would
    // make a hardcoded zero paint somewhere nobody can see — a bug that looks
    // identical to a shader that never ran.
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, target.framebuffer[0]);
    GLES20.glViewport(
        target.viewport[0], target.viewport[1], target.viewport[2], target.viewport[3]);
    GLES20.glUseProgram(quadProgram);
    bindQuad(quadProgram);
    bindTexture(GLES20.GL_TEXTURE2, screenTex[screenFront], quadProgram, "u_screen", 2);
    uniform1f(quadProgram, "u_opacity", 1.0f);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

    frame++;
    // Once, on the first frame that actually reaches the screen. The previous
    // failure mode was total silence — every call succeeded and nothing drew —
    // so the one thing worth a log line is proof that a frame completed.
    if (frame == 1) {
      Log.i(
          TAG,
          "first frame: points="
              + count
              + " field="
              + windWidth
              + "x"
              + windHeight
              + " screen="
              + screenWidth
              + "x"
              + screenHeight
              + " zoom="
              + cam.zoom
              + " density="
              + cam.density
              + " halfLogical="
              + (float) cam.halfWidth()
              + "x"
              + (float) cam.halfHeight()
              + " fpsCap="
              + TARGET_FPS
              + " glError="
              + GLES20.glGetError());
    } else if (frame % 600 == 0) {
      // Roughly every ten seconds at 60 Hz: proves the re-arm is still landing.
      // If this stops while the layer is playing, the pump is the suspect.
      Log.i(TAG, "frames=" + frame + " points=" + count);
    }
  }

  private void setProjection(int program, CameraSnapshot cam) {
    double world = 512 * Math.pow(2.0, cam.zoom);
    double cx = (cam.centerLng + 180) / 360 * world;
    double cy = mercatorY(cam.centerLat) * world;
    double r = -cam.bearing * Math.PI / 180;
    uniform1f(program, "u_world", (float) world);
    uniform2f(program, "u_center", (float) cx, (float) cy);
    uniform2f(program, "u_half", (float) cam.halfWidth(), (float) cam.halfHeight());
    uniform2f(program, "u_rot", (float) Math.cos(r), (float) Math.sin(r));
    uniform1f(program, "u_lon0", fieldLon0);
    uniform2f(program, "u_lat", fieldLat0, fieldLatSpan);
  }

  /**
   * The axis-aligned world rectangle the viewport covers, the box respawns draw
   * from. A rotated view makes this a super-set of what is visible, which only
   * wastes a few candidates — and {@code inView} rejects those anyway.
   */
  private void setViewWorld(int program, CameraSnapshot cam) {
    double world = 512 * Math.pow(2.0, cam.zoom);
    double cx = (cam.centerLng + 180) / 360 * world;
    double cy = mercatorY(cam.centerLat) * world;
    double r = -cam.bearing * Math.PI / 180;
    double cosR = Math.cos(r);
    double sinR = Math.sin(r);
    double hw = cam.halfWidth();
    double hh = cam.halfHeight();
    double minX = Double.MAX_VALUE;
    double minY = Double.MAX_VALUE;
    double maxX = -Double.MAX_VALUE;
    double maxY = -Double.MAX_VALUE;
    for (int i = 0; i < 4; i++) {
      double dx = (i == 0 || i == 3) ? -hw : hw;
      double dy = (i < 2) ? -hh : hh;
      // The exact inverse of the projection's rotation.
      double wx = cx + dx * cosR + dy * sinR;
      double wy = cy - dx * sinR + dy * cosR;
      minX = Math.min(minX, wx);
      maxX = Math.max(maxX, wx);
      minY = Math.min(minY, wy);
      maxY = Math.max(maxY, wy);
    }
    int loc = GLES20.glGetUniformLocation(program, "u_view_world");
    GLES20.glUniform4f(loc, (float) minX, (float) minY, (float) maxX, (float) maxY);
  }

  private static double mercatorY(double lat) {
    double clamped = Math.max(-85.051129, Math.min(85.051129, lat));
    double phi = clamped * Math.PI / 180;
    return (1 - Math.log(Math.tan(Math.PI / 4 + phi / 2)) / Math.PI) / 2;
  }

  // ------------------------------------------------------------- GL plumbing

  private boolean createGlObjects() {
    quadProgram = link(VERT_QUAD, FRAG_SCREEN);
    drawProgram = link(VERT_PARTICLE, FRAG_PARTICLE);
    updateProgram = link(FRAG_UPDATE_VERT, FRAG_UPDATE);
    if (quadProgram == 0 || drawProgram == 0 || updateProgram == 0) {
      Log.e(
          TAG,
          "shader programs failed to build (quad="
              + quadProgram
              + " draw="
              + drawProgram
              + " update="
              + updateProgram
              + ") — the compile log is above");
      deleteGlObjects();
      return false;
    }
    int[] buffers = new int[2];
    GLES20.glGenBuffers(2, buffers, 0);
    quadBuffer = buffers[0];
    indexBuffer = buffers[1];

    FloatBuffer quad = floats(new float[] {0, 0, 1, 0, 0, 1, 1, 1});
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadBuffer);
    GLES20.glBufferData(
        GLES20.GL_ARRAY_BUFFER, quad.capacity() * 4, quad, GLES20.GL_STATIC_DRAW);

    indexCount = STATE_EDGE * STATE_EDGE;
    float[] indices = new float[indexCount];
    for (int i = 0; i < indexCount; i++) {
      indices[i] = i;
    }
    FloatBuffer indexData = floats(indices);
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, indexBuffer);
    GLES20.glBufferData(
        GLES20.GL_ARRAY_BUFFER, indexData.capacity() * 4, indexData, GLES20.GL_STATIC_DRAW);

    int[] fb = new int[1];
    GLES20.glGenFramebuffers(1, fb, 0);
    framebuffer = fb[0];

    // Seed the state textures with noise: an unseeded field starts as a grid.
    byte[] seed = new byte[STATE_EDGE * STATE_EDGE * 4];
    for (int i = 0; i < seed.length; i++) {
      seed[i] = (byte) (Math.random() * 256);
    }
    for (int i = 0; i < 2; i++) {
      stateTex[i] = createTexture(GLES20.GL_NEAREST);
      GLES20.glTexImage2D(
          GLES20.GL_TEXTURE_2D,
          0,
          GLES20.GL_RGBA,
          STATE_EDGE,
          STATE_EDGE,
          0,
          GLES20.GL_RGBA,
          GLES20.GL_UNSIGNED_BYTE,
          ByteBuffer.wrap(seed));
    }
    glContext = EGL14.eglGetCurrentContext();
    return true;
  }

  /** The update pass is a full-screen quad, so it reuses the quad vertex shader. */
  private static final String FRAG_UPDATE_VERT = VERT_QUAD;

  private void ensureScreenTextures(int width, int height) {
    if (screenWidth == width && screenHeight == height && screenTex[0] != 0) {
      return;
    }
    for (int i = 0; i < 2; i++) {
      if (screenTex[i] != 0) {
        GLES20.glDeleteTextures(1, screenTex, i);
      }
      screenTex[i] = createTexture(GLES20.GL_NEAREST);
      // Zeroed, not left undefined: the first fade pass samples this texture
      // and composites the result over the map, so uninitialised contents are
      // driver-dependent garbage painted across the whole viewport.
      GLES20.glTexImage2D(
          GLES20.GL_TEXTURE_2D,
          0,
          GLES20.GL_RGBA,
          width,
          height,
          0,
          GLES20.GL_RGBA,
          GLES20.GL_UNSIGNED_BYTE,
          ByteBuffer.wrap(new byte[width * height * 4]));
    }
    screenWidth = width;
    screenHeight = height;
    trailsDirty = true;
  }

  /**
   * Interleaves the two quantised planes into the two channels a
   * LUMINANCE_ALPHA texture carries, which is what {@code .ra} in the shaders
   * reads. Done here, once per forecast frame, rather than in Dart: the wire
   * then carries the untouched WND1 body and the parser stays in one place.
   */
  private void uploadField(FieldUpload upload) {
    int n = upload.width * upload.height;
    if (upload.bytes.length < upload.planeOffset + n * 2) {
      Log.e(TAG, "wind field truncated: need " + (upload.planeOffset + n * 2));
      return;
    }
    byte[] texels = new byte[n * 2];
    for (int i = 0; i < n; i++) {
      texels[i * 2] = upload.bytes[upload.planeOffset + i];
      texels[i * 2 + 1] = upload.bytes[upload.planeOffset + n + i];
    }
    if (windTex == 0) {
      windTex = createTexture(GLES20.GL_NEAREST);
    } else {
      GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, windTex);
    }
    // REPEAT in S because longitude is cyclic; CLAMP in T because latitude is
    // not — wrapping it walks a particle off the pole onto the other pole.
    GLES20.glTexParameteri(
        GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_REPEAT);
    GLES20.glTexParameteri(
        GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    GLES20.glTexImage2D(
        GLES20.GL_TEXTURE_2D,
        0,
        GLES20.GL_LUMINANCE_ALPHA,
        upload.width,
        upload.height,
        0,
        GLES20.GL_LUMINANCE_ALPHA,
        GLES20.GL_UNSIGNED_BYTE,
        ByteBuffer.wrap(texels));
    windWidth = upload.width;
    windHeight = upload.height;
    windRange = new float[] {upload.uMin, upload.uMax, upload.vMin, upload.vMax};
    fieldLat0 = upload.lat0;
    fieldLatSpan = upload.dLat * upload.height;
    fieldLon0 = upload.lon0;
    fieldReady = true;
  }

  private void deleteGlObjects() {
    if (quadProgram != 0) {
      GLES20.glDeleteProgram(quadProgram);
      quadProgram = 0;
    }
    if (drawProgram != 0) {
      GLES20.glDeleteProgram(drawProgram);
      drawProgram = 0;
    }
    if (updateProgram != 0) {
      GLES20.glDeleteProgram(updateProgram);
      updateProgram = 0;
    }
    if (quadBuffer != 0) {
      GLES20.glDeleteBuffers(2, new int[] {quadBuffer, indexBuffer}, 0);
      quadBuffer = 0;
      indexBuffer = 0;
    }
    if (framebuffer != 0) {
      GLES20.glDeleteFramebuffers(1, new int[] {framebuffer}, 0);
      framebuffer = 0;
    }
    for (int i = 0; i < 2; i++) {
      if (stateTex[i] != 0) {
        GLES20.glDeleteTextures(1, stateTex, i);
        stateTex[i] = 0;
      }
      if (screenTex[i] != 0) {
        GLES20.glDeleteTextures(1, screenTex, i);
        screenTex[i] = 0;
      }
    }
    if (windTex != 0) {
      GLES20.glDeleteTextures(1, new int[] {windTex}, 0);
      windTex = 0;
    }
    screenWidth = 0;
    screenHeight = 0;
    fieldReady = false;
    glContext = null;
    haveTrailCam = false;
    trailsDirty = true;
  }

  private void bindQuad(int program) {
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadBuffer);
    int pos = GLES20.glGetAttribLocation(program, "a_pos");
    GLES20.glEnableVertexAttribArray(pos);
    GLES20.glVertexAttribPointer(pos, 2, GLES20.GL_FLOAT, false, 0, 0);
  }

  private static void bindTexture(int unit, int texture, int program, String name, int index) {
    GLES20.glActiveTexture(unit);
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture);
    GLES20.glUniform1i(GLES20.glGetUniformLocation(program, name), index);
  }

  private static void uniform1f(int program, String name, float value) {
    GLES20.glUniform1f(GLES20.glGetUniformLocation(program, name), value);
  }

  private static void uniform2f(int program, String name, float a, float b) {
    GLES20.glUniform2f(GLES20.glGetUniformLocation(program, name), a, b);
  }

  private static int createTexture(int filter) {
    int[] id = new int[1];
    GLES20.glGenTextures(1, id, 0);
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id[0]);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, filter);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, filter);
    GLES20.glTexParameteri(
        GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
    GLES20.glTexParameteri(
        GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    return id[0];
  }

  private static FloatBuffer floats(float[] values) {
    ByteBuffer bb = ByteBuffer.allocateDirect(values.length * 4);
    bb.order(ByteOrder.nativeOrder());
    FloatBuffer fb = bb.asFloatBuffer();
    fb.put(values);
    fb.position(0);
    return fb;
  }

  private static int link(String vertexSource, String fragmentSource) {
    int vertex = compile(GLES20.GL_VERTEX_SHADER, vertexSource);
    int fragment = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
    if (vertex == 0 || fragment == 0) {
      return 0;
    }
    int program = GLES20.glCreateProgram();
    GLES20.glAttachShader(program, vertex);
    GLES20.glAttachShader(program, fragment);
    GLES20.glLinkProgram(program);
    // The shaders are owned by the program once attached and linked.
    GLES20.glDeleteShader(vertex);
    GLES20.glDeleteShader(fragment);
    int[] status = new int[1];
    GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0);
    if (status[0] == 0) {
      Log.e(TAG, "link failed: " + GLES20.glGetProgramInfoLog(program));
      GLES20.glDeleteProgram(program);
      return 0;
    }
    return program;
  }

  private static int compile(int type, String source) {
    int shader = GLES20.glCreateShader(type);
    GLES20.glShaderSource(shader, source);
    GLES20.glCompileShader(shader);
    int[] status = new int[1];
    GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0);
    if (status[0] == 0) {
      Log.e(TAG, "compile failed: " + GLES20.glGetShaderInfoLog(shader));
      GLES20.glDeleteShader(shader);
      return 0;
    }
    return shader;
  }

  /**
   * The GL state this pass disturbs, captured and put back.
   *
   * <p>MapLibre's own {@code gl::Context} caches state across frames and has no
   * way to be told it went stale on this path, so anything left changed here is
   * a map rendering bug one frame later — usually a missing layer rather than
   * anything that looks like a GL problem. The list is the one the reference
   * implementation restores, which was arrived at the same way.
   */
  private static final class GlState {
    /** Read back by the composite pass, which must land where the map was drawing. */
    final int[] framebuffer = new int[1];

    final int[] viewport = new int[4];
    private final int[] arrayBuffer = new int[1];
    private final int[] program = new int[1];
    private final int[] activeTexture = new int[1];
    private final int[] blendSrcRgb = new int[1];
    private final int[] blendDstRgb = new int[1];
    private final int[] blendSrcAlpha = new int[1];
    private final int[] blendDstAlpha = new int[1];
    private boolean blend;
    private boolean depth;
    private boolean stencil;

    static GlState capture() {
      GlState s = new GlState();
      GLES20.glGetIntegerv(GLES20.GL_FRAMEBUFFER_BINDING, s.framebuffer, 0);
      GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, s.viewport, 0);
      GLES20.glGetIntegerv(GLES20.GL_ARRAY_BUFFER_BINDING, s.arrayBuffer, 0);
      GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, s.program, 0);
      GLES20.glGetIntegerv(GLES20.GL_ACTIVE_TEXTURE, s.activeTexture, 0);
      GLES20.glGetIntegerv(GLES20.GL_BLEND_SRC_RGB, s.blendSrcRgb, 0);
      GLES20.glGetIntegerv(GLES20.GL_BLEND_DST_RGB, s.blendDstRgb, 0);
      GLES20.glGetIntegerv(GLES20.GL_BLEND_SRC_ALPHA, s.blendSrcAlpha, 0);
      GLES20.glGetIntegerv(GLES20.GL_BLEND_DST_ALPHA, s.blendDstAlpha, 0);
      s.blend = GLES20.glIsEnabled(GLES20.GL_BLEND);
      s.depth = GLES20.glIsEnabled(GLES20.GL_DEPTH_TEST);
      s.stencil = GLES20.glIsEnabled(GLES20.GL_STENCIL_TEST);
      return s;
    }

    void restore() {
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer[0]);
      GLES20.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
      GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, arrayBuffer[0]);
      GLES20.glUseProgram(program[0]);
      GLES20.glActiveTexture(activeTexture[0]);
      GLES20.glBlendFuncSeparate(
          blendSrcRgb[0], blendDstRgb[0], blendSrcAlpha[0], blendDstAlpha[0]);
      setEnabled(GLES20.GL_BLEND, blend);
      setEnabled(GLES20.GL_DEPTH_TEST, depth);
      setEnabled(GLES20.GL_STENCIL_TEST, stencil);
    }

    private static void setEnabled(int cap, boolean enabled) {
      if (enabled) {
        GLES20.glEnable(cap);
      } else {
        GLES20.glDisable(cap);
      }
    }
  }
}
