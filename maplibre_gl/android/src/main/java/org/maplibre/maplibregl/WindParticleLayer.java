package org.maplibre.maplibregl;

import android.opengl.EGL14;
import android.opengl.EGLContext;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Choreographer;
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
  private static final int STATE_EDGE = 256;

  /**
   * Time blocks the pool is cut into, and how long each one owns the newest
   * generation before the next is reborn.
   *
   * <p>This is the whole lifetime model, and it replaces the per-particle
   * random drop rate the first version used. A random rebirth cannot be drawn
   * with a fade: a particle that may die on any frame has no age, so there is
   * nothing to fade *along*. Cutting the pool into 16 blocks and reseeding one
   * of them every {@link #BLOCK_FRAMES} frames gives every particle a known
   * position in a 128-frame cycle, which is what {@link #alphaLut} is indexed
   * by — and what turns a field of blinking dots into streaks that grow, run
   * and dim.
   *
   * <p>16 × 8 = 128 frames ≈ 2.13 s at {@link #TARGET_FPS}. One sixteenth of
   * the field is replaced every 0.13 s, so the count on screen never changes
   * even though every particle is short-lived.
   */
  private static final int BLOCKS = 16;

  private static final int BLOCK_FRAMES = 8;
  private static final int BLOCK_ROWS = STATE_EDGE / BLOCKS;
  private static final int LIFETIME = BLOCKS * BLOCK_FRAMES;

  /** Particles per block — the pool the draw count is taken out of. */
  private static final int PARTICLES_PER_BLOCK = STATE_EDGE * BLOCK_ROWS;

  /**
   * Fade-in and fade-out shape, from the reference: in over the first 20 % of
   * a life with a 0.9 power, out over the last 30 % with a 0.8 power.
   *
   * <p>The two windows overlap the rebirth boundary, which is why the cycle is
   * seamless — the block being reseeded is already invisible when it is.
   */
  private static final float[] alphaLut = buildAlphaLut(0.2f, 0.9f, 0.3f, 0.8f);

  private static float[] buildAlphaLut(float inFrac, float inPow, float outFrac, float outPow) {
    float[] lut = new float[LIFETIME];
    int fadeInEnd = Math.round(inFrac * LIFETIME);
    int fadeOutFrom = LIFETIME - Math.round(outFrac * LIFETIME);
    for (int i = 0; i < LIFETIME; i++) {
      float a = 1;
      if (i < fadeInEnd) {
        a = (float) Math.pow((double) i / fadeInEnd, inPow);
      } else if (i >= fadeOutFrom) {
        a = (float) Math.pow((double) (LIFETIME - i) / (LIFETIME - fadeOutFrom), outPow);
      }
      lut[i] = a;
    }
    return lut;
  }

  /** Low-zoom grey multiplier from the reference's copy-to-canvas pass. */
  /**
   * How much of the accumulated trail reaches the map.
   *
   * <p>The reference's own number is 0.44 — {@code 0.4 * mulRGB} and
   * {@code 0.4 * mulA} at its default opacity — and paired with a lift of
   * -0.1 it caps the stroke at 0.34 alpha. That is calibrated for Windy's dark
   * basemap. DPIP composites over a saturated ECMWF ramp, greens and yellows
   * and oranges, and 34% grey on top of that is barely there: measured at 1406
   * particle pixels a frame averaging 222 luminance against a background in
   * the same range.
   *
   * <p>Raised, and the lift eased with it. The lift exists to stop the haze
   * between strokes reading as milk, which a dark background needs and a
   * bright one does not — on this map it was mostly deleting the tail.
   *
   * <p>These two are the brightness knob. Nothing else in the pipeline scales
   * the visible result.
   */
  private static final float COMPOSITE_GAIN = 0.62f;

  /**
   * Multiplier on the reference's particle count.
   *
   * <p>The document's figure is calibrated against the reference's own speed
   * model — displacement as a fraction of the *screen*, about four pixels a
   * frame — while this layer advects through the Dart curves, which move a
   * particle roughly 2.4. A shorter path per particle means fewer pixels of
   * streak each, so the same count reads as a much emptier field. Lifting it
   * here rather than editing the formula keeps the document's shape visible
   * and the deviation in one named place.
   */
  /**
   * {@code glCountMul} for the wind particle type, times the halving the
   * reference applies off desktop.
   *
   * <p>The reference's own value is 1, and it then multiplies the count by 0.5
   * on anything that is not a desktop browser. Measured on a Pixel 9 that gives
   * 576 particles over a 411x923 pt viewport at zoom 6 — faithful, and too
   * sparse to read as a flow on a tall phone. The halving dates from browsers
   * on phones; this is a native layer on a GPU that draws the doubled count at
   * 105 fps with the simulation already running full-size.
   *
   * <p>So: 2, cancelling the halving and leaving Windy's desktop density. This
   * is the one number to change if the field should be denser or thinner — the
   * count is linear in it.
   */
  private static final double GL_COUNT_MUL = 2.0;

  /** {@code glSpeedPx}: the reference's speed unit, screen pixels per second. */
  private static final double SPEED_PX = 100.0;

  /**
   * {@code zoom2speed}, transcribed. Wind is slowed at the widest zooms, where
   * a degree is a handful of pixels and true speed reads as a flicker.
   *
   * <p>The Tuning channel's {@code speedScale} and {@code speedFactor} are
   * deliberately not used for this. They belong to the Dart fallback renderer,
   * whose particles move in field units — fractions of the global grid — and
   * 32 there is not 32 pixels per second here. Feeding them in was how the
   * speed came out wrong at every zoom but one.
   */
  private static final double[] ZOOM_TO_SPEED = {
    0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };

  private static double zoomSpeed(double zoom) {
    int i = (int) Math.round(zoom);
    return ZOOM_TO_SPEED[Math.max(0, Math.min(ZOOM_TO_SPEED.length - 1, i))];
  }

  /** Reference black-point subtraction: removes haze without clipping trails. */
  /** Eased from the reference's -0.1; see {@link #COMPOSITE_GAIN}. */
  private static final float COMPOSITE_LIFT = -0.05f;

  /** The frame rate the advection curves were tuned at. See {@link #setPlaying}. */
  private static final int TARGET_FPS = 60;

  // ---------------------------------------------------------------- shaders
  //
  // **Every name that crosses the vertex/fragment boundary carries an explicit
  // precision qualifier.** The two stages default differently — highp in the
  // vertex shader, mediump in the fragment one — and GLSL ES rejects a uniform
  // whose declared precision differs between them. That is a *link* error, so
  // both shaders compile clean and the failure names a variable that looks
  // correct in each file on its own:
  //
  //     L0001 The fragment floating-point variable u_len does not match the
  //           vertex variable u_len. The precision does not match.
  //
  // Nothing catches it before a device runs it. Varyings are the same rule;
  // some drivers tolerate a mismatch there and some do not, which is worse
  // than a consistent failure because it ships.


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
          + "varying mediump vec2 v_tex;\n"
          + "void main() {\n"
          + "  v_tex = a_pos;\n"
          + "  gl_Position = vec4(1.0 - 2.0 * a_pos, 0.0, 1.0);\n"
          + "}";

  /**
   * State updates must preserve the particle-to-texel mapping.
   *
   * <p>{@link #VERT_QUAD} deliberately flips its geometry for the screen/trail
   * passes. Reusing it with {@code v_tex = a_pos} for simulation wrote each
   * updated particle into the opposite texel. The draw pass then joined that
   * particle to an unrelated previous state, producing the regular bundles of
   * random segments visible on device. Matching the varying to the flipped
   * framebuffer coordinate makes the update an identity mapping again.
   */
  private static final String VERT_UPDATE =
      "precision mediump float;\n"
          + "attribute vec2 a_pos;\n"
          + "varying mediump vec2 v_tex;\n"
          + "void main() {\n"
          + "  v_tex = 1.0 - a_pos;\n"
          + "  gl_Position = vec4(1.0 - 2.0 * a_pos, 0.0, 1.0);\n"
          + "}";

  private static final String FRAG_SCREEN =
      "precision mediump float;\n"
          + "uniform sampler2D u_screen;\n"
          + "uniform float u_opacity;\n"
          // The reference composites `tex * uPars0 + uPars1` with uPars1 = -0.1.
          // That subtraction is what stops the accumulated haze between strokes
          // from reading as milk; without it every trail sits on a grey wash.
          + "uniform float u_lift;\n"
          // Re-anchors the accumulated trails when the camera panned since they
          // were laid down. Zero for the composite, which draws them where they
          // already are. See the shift maths in `trailShift`.
          + "uniform vec2 u_shift;\n"
          + "varying mediump vec2 v_tex;\n"
          + "void main() {\n"
          + "  vec2 uv = 1.0 - v_tex + u_shift;\n"
          // Whatever scrolled in from outside the old frame has no history, and
          // sampling it clamped would smear the edge pixel across the strip the
          // pan uncovered.
          + "  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {\n"
          + "    gl_FragColor = vec4(0.0);\n"
          + "    return;\n"
          + "  }\n"
          // Multiply in premultiplied space so the fade cannot drift to grey.
          //
          // No `floor` here, and that is the whole tail. The reference fades
          // with a hardware blend — `(ZERO, CONSTANT_ALPHA)`, a pure multiply —
          // and lets the RGBA8 write round. Truncating instead loses a whole
          // LSB per frame on top of the 3%, which barely touches a bright
          // stroke and destroys a dim one: a trail at 5/255 decays by 0.8 a
          // frame, not 0.97. The tail dies within a few frames and every
          // particle reads as a bright blunt head with nothing behind it.
          + "  vec4 c = texture2D(u_screen, uv);\n"
          + "  gl_FragColor = max(vec4(0.0), c * u_opacity + u_lift);\n"
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

  /**
   * One particle as a short line from where it was to where it is.
   *
   * <p>This is the whole reason the rewrite happened. A {@code GL_POINTS}
   * sprite can only say "a particle is here"; the reference draws a quad whose
   * two ends are pinned to the previous and current positions, so every
   * particle is a stroke *along its own motion* — length proportional to speed,
   * direction exactly the wind's — and sixteen overlapping generations of them
   * read as flowing streamlines rather than as drifting confetti.
   *
   * <p>The quad's four corners are encoded in two bytes of the vertex: {@code z}
   * picks the side (line width, across the direction of travel) and {@code w}
   * picks the end (tail at the old position, head at the new one). One vertex
   * buffer serves all sixteen blocks; {@code u_block.zw} slides the state lookup
   * onto the block being drawn.
   */
  /**
   * The pass DPIP needs and the reference does not: the global grid, redrawn
   * as a viewport-aligned vector field.
   *
   * <p>This is the whole reason the previous attempt could not look like Windy.
   * There, a particle's stored 0..1 position **is** its screen position — the
   * wind texture is stitched from the tiles covering the viewport, so sampling
   * it costs a scale and an offset and the draw shader is literally
   * {@code pos * 2 - 1}. Every constant in that system — count, speed in
   * pixels per second, line width, fade rate — is calibrated in screen space.
   *
   * <p>DPIP's field is one global ECMWF grid, so the previous version kept
   * particles in lat/lon and projected every vertex. Transplanting the
   * reference's constants onto that cannot reproduce it: the field-to-screen
   * mapping changes with zoom, with latitude (Mercator stretch) and with
   * bearing. The magic ×10 on the particle count was that mismatch wearing a
   * constant's clothes.
   *
   * <p>Doing the projection **once per frame into a texture** instead of once
   * per vertex per frame puts the particles back in screen space, and the rest
   * of the port becomes transcription.
   *
   * <p>The encoding is the reference's own: normalise by {@code glMaxSpeedParam},
   * bend by {@code pow(|v|², 0.35)}, floor at {@code glMinSpeedParam}, then
   * store direction × 128 + 128 so that 0.5 is a zero vector. Two special
   * values carry meaning downstream — all-zero means "no data" and freezes a
   * particle where it stands; exactly 0.5 means "no wind" and lets it die.
   */
  private static final String FRAG_WIND_PROJECT =
      "precision highp float;\n"
          + "uniform sampler2D u_field;\n"
          + "uniform vec2 u_wind_min;\n"
          + "uniform vec2 u_wind_max;\n"
          + "uniform vec3 u_curve;\n" // speedMul, speedMin, cPow
          + "varying mediump vec2 v_tex;\n"
          + PROJECT_GLSL
          + "void main() {\n"
          // Particle space has y up (it becomes clip space directly); screen
          // pixels have y down. This pass runs under VERT_UPDATE, so v_tex is
          // already the texel being written.
          + "  vec2 sp = vec2(v_tex.x * 2.0 * u_half.x,\n"
          + "                 (1.0 - v_tex.y) * 2.0 * u_half.y);\n"
          // The exact inverse of projectField: unrotate, then unproject.
          + "  vec2 d = sp - u_half;\n"
          + "  float dx = d.x * u_rot.x + d.y * u_rot.y;\n"
          + "  float dy = -d.x * u_rot.y + d.y * u_rot.x;\n"
          + "  float lon = (u_center.x + dx) / u_world * 360.0 - 180.0;\n"
          + "  float lat = mercLat((u_center.y + dy) / u_world);\n"
          + "  vec2 uv = vec2(fract((lon - u_lon0) / 360.0),\n"
          + "                 (lat - u_lat.x) / u_lat.y);\n"
          // Off the grid's latitude band: no data. All-zero, which the update
          // pass reads as "hold position" rather than "drift to the pole".
          + "  if (uv.y < 0.0 || uv.y > 1.0) { gl_FragColor = vec4(0.0); return; }\n"
          + "  vec2 v = mix(u_wind_min, u_wind_max, texture2D(u_field, uv).ra);\n"
          // Into the viewport's frame. The position above was unrotated to find
          // the grid cell; the vector it holds is still east/north and has to
          // be turned the same way, or a rotated map keeps north-up arrows.
          // The v flip is the y-up/y-down change, folded into the rotation.
          + "  v = vec2(v.x * u_rot.x + v.y * u_rot.y,\n"
          + "           -v.x * u_rot.y + v.y * u_rot.x);\n"
          + "  vec2 orient = v * u_curve.x;\n"
          + "  float d2 = dot(orient, orient);\n"
          + "  if (d2 < 1e-8) { gl_FragColor = vec4(0.5, 0.5, 0.0, 1.0); return; }\n"
          // The speed curve: compresses the middle of the range and lifts the
          // bottom, so a light breeze still moves visibly.
          + "  float scale = (128.0 * max(pow(d2, u_curve.z), u_curve.y)) / sqrt(d2);\n"
          + "  orient = orient * scale + vec2(128.0);\n"
          + "  gl_FragColor = vec4(orient / 255.0, 0.0, 1.0);\n"
          + "}";

  /**
   * One particle as a short line from where it was to where it is — the
   * reference's vertex shader, transcribed.
   *
   * <p>Note what is absent: any projection at all. {@code pos * 2 - 1} is clip
   * space, because the particle's stored position is already its position on
   * screen. Everything the previous version did per vertex — Mercator, the
   * bearing rotation, the world-x fold — now happens once a frame in
   * {@link #FRAG_WIND_PROJECT}.
   */
  private static final String VERT_SEGMENT =
      "precision highp float;\n"
          + "attribute vec4 a_vert;\n" // (texelX, texelY in block, side, end)
          + "uniform sampler2D u_state0;\n"
          + "uniform sampler2D u_state1;\n"
          + "uniform vec4 u_block;\n" // (1/res, 1/res, 0, block/16)
          + "uniform vec4 u_line;\n" // (hw*2/255, hh*2/255, -hw, -hh)
          + "uniform vec4 u_edge;\n" // (0, 0, l*2/255, -l)
          + "varying mediump float v_discard;\n"
          + "varying mediump float v_edge;\n"
          + "void main() {\n"
          // Texel centres. The reference samples the boundary; with nearest
          // filtering that is a rounding coin-flip which can pair a particle's
          // current position with its neighbour's previous one — and the quad
          // drawn between two unrelated particles is a stroke across the map.
          + "  vec2 tc = (a_vert.xy + 0.5) * u_block.xy + u_block.zw;\n"
          + "  vec4 tex0 = texture2D(u_state0, tc);\n"
          + "  v_discard = step(0.025, tex0.r + tex0.g + tex0.b + tex0.a);\n"
          + "  vec4 tex1 = texture2D(u_state1, tc);\n"
          + "  vec2 posA = fract(tex0.ba + tex0.rg / 255.5) * 2.0 - 1.0;\n"
          + "  vec2 posB = fract(tex1.ba + tex1.rg / 255.5) * 2.0 - 1.0;\n"
          + "  vec2 dirF = posA - posB;\n"
          + "  vec2 dirFN = normalize(dirF);\n"
          + "  float d = length(dirF);\n"
          + "  vec2 dirRN = vec2(dirFN.y, -dirFN.x);\n"
          + "  vec2 pos = mix(posB, posA, a_vert.w * 0.003921569);\n"
          + "  pos += dirRN * (a_vert.zz * u_line.xy + u_line.zw);\n"
          // Two ways a segment carries no information, and both must go.
          //
          // `d > 0.5`: wrapped across an edge this frame, so the quad would
          // span the screen.
          //
          // `d < 1e-5`: the two ends are the same point. That happens on the
          // frame after a rebirth, when both state textures hold the identical
          // fresh random position — `normalize(0)` is NaN, it propagates
          // through `dirRN` into `gl_Position`, and a NaN clip coordinate
          // rasterises to whatever the driver feels like. One block every
          // eighth frame is 7.5 flashes a second. The reference carries this
          // same guard, but only under its WAVES define; a zero-length wind
          // segment is just as meaningless.
          + "  if (d > 0.5 || d < 1e-5) { pos.x += 10.0; }\n"
          + "  gl_Position = vec4(pos.xy, 0.0, 1.0);\n"
          + "  v_edge = u_edge.z * a_vert.z + u_edge.w;\n"
          + "}";

  /** Grey, feathered across the width. The reference has no speed term. */
  private static final String FRAG_SEGMENT =
      "precision mediump float;\n"
          + "uniform vec4 u_colour;\n" // alphaLut[f] in all four channels
          + "uniform mediump float u_len;\n"
          + "varying mediump float v_discard;\n"
          + "varying mediump float v_edge;\n"
          + "void main() {\n"
          + "  if (v_discard <= 0.0) { discard; }\n"
          + "  float aa = clamp(u_len - abs(v_edge), 0.0, 1.0);\n"
          + "  gl_FragColor = u_colour * vec4(aa);\n"
          + "}";

  /**
   * The reference's simulation, transcribed.
   *
   * <p>No projection, no respawn, no density weighting, no view test: a
   * particle is a point in screen space, the wind texture is aligned to that
   * space, and the only way to die is to stop moving. Rebirth is the CPU's job
   * (see {@link #reseedBlock}).
   *
   * <p>It runs under {@link #VERT_UPDATE}, not {@link #VERT_QUAD}: the update
   * is the one pass whose output texel must be the input texel, and the quad
   * shader the trail passes share deliberately flips both axes.
   */
  private static final String FRAG_UPDATE =
      "precision highp float;\n"
          + "uniform sampler2D u_particles;\n"
          + "uniform sampler2D u_wind;\n"
          + "uniform vec4 u_disp;\n" // (2o, 2s, -o, -s)
          + "varying mediump vec2 v_tex;\n"
          + "const float treshold = 0.025;\n"
          + "const float prec = 1000.0;\n"
          + "const float e = 0.001;\n"
          + "bool hasMovement(vec2 dp) {\n"
          + "  float r = step(treshold, abs(dp.r * prec));\n"
          + "  float g = step(treshold, abs(dp.g * prec));\n"
          + "  return (r + g) > 0.0;\n"
          + "}\n"
          + "void main() {\n"
          + "  vec4 tex0 = texture2D(u_particles, v_tex);\n"
          + "  vec2 pos = tex0.ba + tex0.rg / 255.5;\n"
          + "  vec2 dpos = texture2D(u_wind, fract(pos)).rg;\n"
          // All zero: no data under this pixel. Hold position rather than
          // drift, so a particle at the edge of the grid stays put.
          + "  if (dpos.x + dpos.y < e) { gl_FragColor = tex0; return; }\n"
          + "  dpos = dpos * u_disp.xy + u_disp.zw;\n"
          // Below the movement threshold is how a particle dies. The draw pass
          // discards an all-zero texel; the block cycle brings it back.
          + "  if (!hasMovement(dpos)) { gl_FragColor = vec4(0.0); return; }\n"
          + "  pos = fract(pos + dpos);\n"
          + "  gl_FragColor.rg = fract(pos * 255.0 + 0.25 / 255.0);\n"
          + "  gl_FragColor.ba = pos - gl_FragColor.rg / 255.0;\n"
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

  /** Display density, needed to turn the renderer's physical size into logical. */
  private double displayDensity = 1;

  /** The style layer MapLibre drives; null until {@link #attach} succeeds. */
  @Nullable private org.maplibre.android.style.layers.CustomLayer styleLayer;
  @Nullable private EGLContext glContext;

  private final Runnable drawRunnable = this::onFinishDrawing;
  static {
    System.loadLibrary("dpip_wind_host");
  }

  /**
   * Creates the C++ {@code mbgl::style::CustomLayerHost} that forwards the
   * map's own render callbacks here, and returns it as the {@code long} that
   * {@code CustomLayer(String, long)} takes. Ownership passes to MapLibre.
   */
  private static native long nativeCreateHost(WindParticleLayer layer);

  // ------------------------------------------- called from the render thread

  /** GL context is current and the map is about to draw. */
  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeInitialize() {
    // Nothing eager: `drawFrame` creates what it needs on first use, and the
    // field may not have arrived yet.
  }

  /**
   * One map frame, inside the map's own render pass.
   *
   * <p>The camera arrives as the renderer has it for *this* frame, which is
   * what the old snapshot pushed from the platform thread could only
   * approximate — it was stale for the whole of every camera animation.
   */
  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeRender(
      double width, double height, double lat, double lng, double zoom, double bearing) {
    if (released || !playing) {
      return;
    }
    camera =
        new CameraSnapshot(
            lat,
            lng,
            zoom,
            bearing,
            (int) Math.round(width),
            (int) Math.round(height),
            displayDensity);
    try {
      drawFrame();
    } catch (RuntimeException e) {
      // Never let this escape: it runs inside the map's render loop, and an
      // exception here takes the map down with it.
      Log.e(TAG, "wind particle frame failed", e);
    }
  }

  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeContextLost() {
    quadProgram = 0;
    glContext = null;
    trailsDirty = true;
  }

  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeDeinitialize() {
    deleteGlObjects();
  }

  private final Runnable armRunnable = this::arm;

  /**
   * The display's own clock, obtained on the pump thread.
   *
   * <p>`surfaceRedrawNeededAsync` does not merely register a callback — it
   * *requests a redraw*. Re-arming it as fast as the pump thread could post
   * therefore made this layer the thing driving the map's frame rate, and it
   * drove it past what the panel can show: measured at 169 fps against a
   * 120 Hz display, so roughly fifty frames a second were rendered and thrown
   * away, at irregular moments. That is the flicker, and it is also why
   * `setMaximumFps(60)` never took — MapLibre's limiter governs its own loop,
   * not redraws somebody else asks for.
   *
   * <p>A Choreographer callback fires once per vsync, so the request rate
   * becomes the display rate exactly, whatever that is: 120 here, 60 on a
   * cheaper panel, and it follows the panel if the system changes it.
   */
  @Nullable private Choreographer pump;

  private final Choreographer.FrameCallback frameCallback =
      new Choreographer.FrameCallback() {
        @Override
        public void doFrame(long frameTimeNanos) {
          arm();
        }
      };

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
  private int projectProgram;
  private int quadBuffer;
  private int vertexBuffer;
  private int indexBuffer;
  private int framebuffer;
  private final int[] stateTex = new int[2];
  private final int[] screenTex = new int[2];
  private int windTex;
  private int projTex;
  private int seedTex;
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

  /** Which block is reseeded next, and how far into its eight frames we are. */
  private int stateBlock;

  /**
   * Time accumulated since the last deposit — see the two-rate note in
   * {@code renderPasses}. Compositing runs every map frame; the simulation and
   * the strokes it lays down run at {@link #TARGET_FPS}.
   */
  private double depositTimer;

  /** Per-interval counters, reported with the periodic frame line. */
  private int deposits;
  private int skips;
  private int fboIncomplete;
  private int glErrors;
  private int lastGlError;
  private double elapsedMin = 1e9;
  private double elapsedMax;
  private double elapsedSum;
  private int elapsedN;

  /** Diagnostics for the trail shift, reported with the periodic frame line. */
  private int shiftFrames;
  private double shiftMaxTexels;
  private final java.util.Random reseedRandom = new java.util.Random();

  /** Position inside the current eight-frame block, in tuned 60 Hz frames. */
  private double blockFrame;

  /** Monotonic clock for frame-rate-independent GPU advection. */
  private long lastFrameNanos;

  /** Time not yet consumed by a tuned 60 Hz simulation step. */

  /** UV offset applied to the surviving trails so a pan does not smear them. */
  private float trailShiftX;

  private float trailShiftY;

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

    /**
     * Viewport in **logical** points, as the renderer reports it.
     *
     * <p>The same units iOS reads off `MLNStyleLayerDrawingContext.size`, and
     * the same three uses: halved for the projection, multiplied by the density
     * for the framebuffer, and multiplied out for the particle count.
     */
    final double width;

    final double height;

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

    /** Framebuffer size in physical pixels — what the offscreen textures are. */
    int physicalWidth() {
      return (int) Math.round(width * density);
    }

    int physicalHeight() {
      return (int) Math.round(height * density);
    }

    CameraSnapshot(
        double centerLat,
        double centerLng,
        double zoom,
        double bearing,
        double width,
        double height,
        double density) {
      this.centerLat = centerLat;
      this.centerLng = centerLng;
      this.zoom = zoom;
      this.bearing = bearing;
      this.width = width;
      this.height = height;
      this.density = density <= 0 ? 1 : density;
    }

    /**
     * Viewport half-size in logical points — the space {@code u_world} is in.
     *
     * <p>{@link #width} and {@link #height} are logical, exactly as they arrive
     * from `CustomLayerRenderParameters` and exactly as iOS reads them off
     * `MLNStyleLayerDrawingContext.size`. They used to be physical, because the
     * old hook read them off an Android `View`, and this divided by the density
     * to compensate. Feeding logical values into that produced a viewport
     * 2.625x too small on this device: the field projected onto the wrong part
     * of the map and every stroke came out the wrong width.
     */
    double halfWidth() {
      return width / 2.0;
    }

    double halfHeight() {
      return height / 2.0;
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
    double lineWidthZ3 = 1;
    double lineWidthZ4 = 1.2;
    double lineWidthZ5 = 1.6;
    double lineWidthZ6 = 1.8;
    double lineWidthZ7 = 2;
    double particleWidth = 1.3;
    double speedFactorLo = 0.2;
    double speedFactorHi = 0.0151;
    double fadeOpacityLo = 0.97;
    double fadeOpacityHi = 0.97;
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

    float lineWidth(double zoom) {
      double z = Math.max(zoomLo, Math.min(zoomHi, zoom));
      int lo = (int) Math.floor(z);
      int hi = (int) Math.ceil(z);
      double a = lineWidthAt(lo);
      if (lo == hi) return (float) a;
      double b = lineWidthAt(hi);
      return (float) (a * (1 - (z - lo)) + b * (z - lo));
    }

    private double lineWidthAt(int zoom) {
      switch (zoom) {
        case 3:
          return lineWidthZ3;
        case 4:
          return lineWidthZ4;
        case 5:
          return lineWidthZ5;
        case 6:
          return lineWidthZ6;
        default:
          return lineWidthZ7;
      }
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
  boolean attach(@Nullable org.maplibre.android.maps.Style style) {
    if (style == null) {
      Log.w(TAG, "no style yet; wind particles need one to hold the layer");
      return false;
    }
    displayDensity = mapView.getResources().getDisplayMetrics().density;
    // A real MapLibre style layer, driven by the renderer itself.
    //
    // The plugin used to hook `MapLibreSurfaceView.surfaceRedrawNeededAsync`
    // instead, because MapLibre Android exposes no Java entry point into the
    // render pass. That hook is not one: it *requests* a redraw as well as
    // registering a one-shot callback, so this layer became the map's frame
    // driver and drove it past the panel — measured at 169 fps against 120 Hz
    // with callback spacing swinging between 0.46 ms and 50 ms. Frames the map
    // started on its own carried no particles, which is what the flicker was.
    //
    // `CustomLayer(String, long)` takes a pointer to a C++
    // `mbgl::style::CustomLayerHost`; see `src/main/cpp/wind_particle_host.cpp`
    // for why that needs no linking against MapLibre. iOS has used the same
    // core mechanism all along, behind `MLNCustomStyleLayer`.
    try {
      styleLayer =
          new org.maplibre.android.style.layers.CustomLayer(
              "dpip-wind-particles", nativeCreateHost(this));
      style.addLayer(styleLayer);
    } catch (RuntimeException | UnsatisfiedLinkError e) {
      Log.e(TAG, "could not add the wind custom layer", e);
      styleLayer = null;
      return false;
    }
    return true;
  }

  void setPlaying(boolean value) {
    if (playing == value || released) {
      return;
    }
    playing = value;
    // Deliberately does NOT clear the trails.
    //
    // It used to, on the reasoning that the camera can move while the layer is
    // paused and the old streaks would no longer share the next frame's
    // projection. True, and already handled: the first frame after a resume
    // compares the live camera against the one saved when the last frame was
    // drawn, then shifts the trail texture to match — or clears it, but only
    // when the move was larger than half the viewport and nothing would have
    // survived anyway.
    //
    // Clearing here on top of that was measured as the flicker. `playing`
    // follows map gestures, so a tap or a short drag paused the layer, wiped a
    // second of accumulated streaks — which is the entire visible field — and
    // let it rebuild. Five of those inside seven seconds is what "偶爾閃爍"
    // looked like.
    lastFrameNanos = 0;
    depositTimer = 0;
    if (value) {
      // WHEN_DIRTY, not CONTINUOUS, and that is the whole point of pacing the
      // re-arm with Choreographer.
      //
      // CONTINUOUS gives the map its own frame driver. This layer has one too:
      // `surfaceRedrawNeededAsync` requests a redraw as well as registering
      // the callback. Two drivers race, and the callback is one-shot — a frame
      // the map starts on its own, after ours was consumed and before the next
      // is registered, renders with no wind layer in it. Alternating present
      // and absent frames is exactly the high-frequency flicker.
      //
      // With WHEN_DIRTY the map draws only when asked, and the only thing
      // asking is the vsync callback: one request, one frame, every frame
      // carrying this layer.
      mapView.setRenderingRefreshMode(MapRenderer.RenderingRefreshMode.WHEN_DIRTY);
      // No fps cap any more, and that is a consequence of the integrator
      // becoming time-based rather than frame-based.
      //
      // The cap existed because every step used to advance by one FRAME:
      // uncapped the render thread free-ran at 237 fps on this 120 Hz panel
      // and the wind blew four times too fast. Displacement, the trail fade
      // and the block clock now all scale by the real elapsed time, so the
      // motion is identical at any rate and the cap buys nothing.
      //
      // What it cost was frames. MapLibre enforces a cap by skipping, and a
      // skipped frame is one the map presents without this layer in it — the
      // field vanishes for a single frame and returns on the next. Removing
      // the cap removes that whole class of gap.
      //
      // TARGET_FPS survives as what it always really was: the reference rate
      // the curves in wind_particle_sim.dart were fitted at, and the unit the
      // time-based maths converts into.
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
    markTrailsDirty("new field");
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
      Choreographer c = pump;
      if (c != null) {
        handler.post(() -> c.removeFrameCallback(frameCallback));
      }
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
    // Re-arm FIRST, then draw.
    //
    // This callback is one-shot: the layer is only in a map frame at all
    // because `surfaceRedrawNeededAsync` was registered for it, and that
    // registration has to go through the pump thread — calling it from here,
    // the render thread, is a no-op (see [pumpHandler]).
    //
    // Re-arming after the draw lost the race. The map runs CONTINUOUS at the
    // panel's rate, so between this method returning and `armRunnable`
    // actually reaching `surfaceRedrawNeededAsync` the map could already be
    // into the next frame — and that frame renders with no wind layer in it.
    // Measured on a Pixel 9: the layer made it into 105 of the panel's 120
    // frames a second, so roughly fifteen times a second the map drew without
    // the particles. In a screen recording that is a single frame where the
    // whole field goes dark and the next frame brings it back, which is
    // exactly what "偶爾閃爍" was.
    //
    // Posting before the draw hands the pump thread the entire draw duration —
    // milliseconds, against a vsync interval of 8.3 ms — to land the
    // registration. Same number of posts, same thread, one frame earlier.
    Handler handler = pumpHandler;
    if (playing && handler != null) {
      // Onto the pump thread, then onto the next vsync. Two hops, and both are
      // needed: `arm()` is a no-op from the render thread, and Choreographer
      // belongs to the thread that asked for it.
      handler.post(
          () -> {
            Choreographer c = pump;
            if (c != null) {
              c.postFrameCallback(frameCallback);
            } else {
              // Before the Choreographer has been fetched, fall back to the
              // old immediate re-arm so the very first frames still come.
              arm();
            }
          });
    }
    try {
      drawFrame();
    } catch (RuntimeException e) {
      // Never let this escape: it runs inside the map's render loop, and an
      // exception here takes the map down with it.
      Log.e(TAG, "wind particle frame failed", e);
    }
  }

  private void drawFrame() {
    CameraSnapshot cam = camera;
    if (cam == null || cam.width <= 0 || cam.height <= 0) {
      return;
    }
    if (contextLost()) {
      deleteGlObjects();
      markTrailsDirty("GL context lost");
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
    ensureScreenTextures(cam.physicalWidth(), cam.physicalHeight());

    // The trails live in screen space, so a camera change invalidates every
    // pixel in them. A pan is recoverable — the whole image simply translated,
    // and re-sampling it at the offset puts it back under the map it describes.
    // A zoom or a rotation is not: they scale and turn the image, and a texture
    // offset cannot express either, so the buffer is dropped instead. Panning
    // is both the common case and the one that looks worst when discarded.
    trailShiftX = 0;
    trailShiftY = 0;
    if (haveTrailCam) {
      boolean reprojected =
          Math.abs(cam.zoom - trailZoom) > 1e-4 || Math.abs(cam.bearing - trailBearing) > 0.05;
      if (reprojected) {
        markTrailsDirty(
            String.format(
                "reprojected dZoom=%.6f dBearing=%.4f",
                cam.zoom - trailZoom, cam.bearing - trailBearing));
      } else if (Math.abs(cam.centerLat - trailLat) > 1e-9
          || Math.abs(cam.centerLng - trailLng) > 1e-9) {
        // `projectField` puts a world point at
        //     half + (w - centre) rotated by u_rot,
        // so moving the centre by d moves every fixed point on screen by -d,
        // rotated the same way.
        double world = 512 * Math.pow(2.0, cam.zoom);
        double dcx = ((cam.centerLng - trailLng) / 360) * world;
        double dcy = (mercatorY(cam.centerLat) - mercatorY(trailLat)) * world;
        double r = -cam.bearing * Math.PI / 180;
        double cosR = Math.cos(r);
        double sinR = Math.sin(r);
        double dx = -dcx * cosR + dcy * sinR;
        double dy = -dcx * sinR - dcy * cosR;
        // The fade pass samples at `1 - v_tex`, where u maps to screen x and v
        // to screen y with y pointing *up*. Content that moved by (dx, dy) is
        // read back from (-dx, +dy).
        double shiftX = -dx / (2 * cam.halfWidth());
        double shiftY = dy / (2 * cam.halfHeight());
        if (Math.abs(shiftX) > 0.5 || Math.abs(shiftY) > 0.5) {
          // More than half the viewport: almost nothing survives the move, and
          // what does is a thin band of stale streaks along one edge.
          markTrailsDirty(
              String.format("panned shift=%.3f,%.3f", shiftX, shiftY));
        } else {
          trailShiftX = (float) shiftX;
          trailShiftY = (float) shiftY;
          // In texels, which is the unit that matters: the trail textures are
          // NEAREST, so a sub-texel shift does not blur, it snaps the whole
          // accumulated image by a whole pixel or not at all. A camera that is
          // nominally still but numerically noisy would make that snap flip
          // back and forth every frame, and the entire field would shimmer.
          double tx = Math.abs(shiftX) * screenWidth;
          double ty = Math.abs(shiftY) * screenHeight;
          shiftFrames++;
          shiftMaxTexels = Math.max(shiftMaxTexels, Math.max(tx, ty));
        }
      }
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

  /**
   * Clears the accumulated streaks, and says why.
   *
   * <p>Every reset is a visible event — the whole field vanishes and rebuilds
   * over the next second — so a reset nobody asked for reads as a flicker.
   * Routing all of them through one logged call is the only way to tell an
   * intended clear (new forecast, resumed layer) from a spurious one, which is
   * a distinction no screenshot can make.
   */
  private void markTrailsDirty(String why) {
    if (!trailsDirty) {
      Log.i(TAG, "trails cleared: " + why);
    }
    trailsDirty = true;
  }

  private static String encodingName(int value) {
    if (value == GLES30.GL_SRGB) return "SRGB";
    if (value == GLES20.GL_LINEAR) return "LINEAR";
    return "0x" + Integer.toHexString(value);
  }

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

    // The offscreen state and trail passes own RGBA textures and need every
    // channel even if the map pass that called us did not. Particle y stores
    // its high byte in A, so inheriting a restricted map mask would corrupt it.
    GLES20.glColorMask(true, true, true, true);

    // Render callbacks may arrive at 107 Hz despite the 60 FPS request. State
    // advances at the tuned 60 Hz cadence; skipped callbacks still composite
    // because MapLibre's framebuffer is new each time.
    // Two clocks, as the reference keeps them, and mixing them up is visible.
    //
    // Particles move by the *real* time since the last drawn frame. Quantising
    // that into whole 60 Hz steps is what put the blunt heads on the strokes:
    // this device draws at 53.8 fps, so `floor(elapsed * 60)` returns 2 on
    // roughly one frame in ten, and those frames lay down a quad twice as long
    // as its neighbours. A double-length quad's flat end is a rectangle nothing
    // else covers.
    //
    // The block clock still advances in whole frames, because the fade curve is
    // a 128-entry table indexed by frame.
    // Two rates, and they are not the same thing.
    //
    // COMPOSITING happens on every map frame — 118 a second on this panel —
    // because a frame the map presents without this layer in it is a frame the
    // field disappears for.
    //
    // DEPOSITING is throttled to TARGET_FPS, and that is not a performance
    // choice. The trail texture is RGBA8. At 118 fps a particle advances about
    // two pixels a frame, so each new quad overlaps the last by more than half
    // and every pixel is re-blended twice as often, in increments that sit near
    // the 8-bit least significant bit. Adjacent frames then round to different
    // levels and the accumulated stroke develops a regular stipple that boils —
    // visible in a frame-by-frame capture as a dot pattern baked into the trail
    // while the bright head moves on. iOS renders at 60 and has none of it,
    // which is the whole of "iOS 的效果就很好".
    //
    // So: hold the physics on real elapsed time (motion stays identical at any
    // rate) but give each deposit a full 1/60 s of travel, exactly as iOS does.
    long nowNanos = System.nanoTime();
    double elapsed = 1.0 / TARGET_FPS;
    if (lastFrameNanos != 0) {
      elapsed =
          Math.max(0, Math.min(0.1, (nowNanos - lastFrameNanos) / 1_000_000_000.0));
    }
    lastFrameNanos = nowNanos;
    // The spacing of our own callbacks, which is what the deposit accumulator
    // actually sees. Even spacing means one driver; a bimodal spread means the
    // map is still being driven by something besides the vsync request.
    double ms = elapsed * 1000;
    elapsedMin = Math.min(elapsedMin, ms);
    elapsedMax = Math.max(elapsedMax, ms);
    elapsedSum += ms;
    elapsedN++;
    depositTimer += elapsed;
    final double step = 1.0 / TARGET_FPS;
    // Three quarters of a step, not a whole one, and the fraction is load
    // bearing. A 60 Hz panel delivers frames 16.6 ms apart against a 16.667 ms
    // step — a hair short — so a strict `>=` deposits on every *other* frame
    // and the simulation silently runs at 30 Hz. The tolerance absorbs that
    // jitter while still halving on a 120 Hz panel (8.3 ms is well under it).
    if (depositTimer < step * 0.75) {
      // Nothing new to lay down, but the map still needs the layer in this
      // frame. Paint the trails as they stand.
      skips++;
      composite(target);
      frame++;
      return;
    }
    deposits++;
    // The whole interval since the last deposit, so a dropped frame is caught
    // up rather than lost. Capped so a stall does not teleport the field.
    double dt = Math.min(depositTimer, 4 * step);
    depositTimer = 0;

    // How many of each block's quads to draw. The pool is fixed at 65,536; zoom
    // only decides how much of it is asked for, exactly as the reference does —
    // resizing the pool instead would reseed the whole field at every zoom
    // threshold, which is a visible flash.
    // getAmount(width, height, zoom), transcribed. It is proportional to the
    // viewport's area in logical pixels — not to a constant — so a tablet gets
    // more particles than a phone at the same zoom, which is the whole reason
    // the density used to look wrong on one device and right on another.
    //
    //     amount(w, h, z) = min(15000, w · h / (50 · 1.6^(z-2)))
    //
    // then halved off desktop, and divided by the 65,536-particle pool.
    double logicalW = cam.width;
    double logicalH = cam.height;
    double amount =
        Math.min(15000, Math.round(logicalW * logicalH / (50.0 * Math.pow(1.6, cam.zoom - 2))));
    double relativeAmount = amount * 0.5 / (STATE_EDGE * (double) STATE_EDGE) * GL_COUNT_MUL;
    int perBlock = Math.max(
        1, Math.min(indexCount, (int) Math.round(relativeAmount * PARTICLES_PER_BLOCK)));

    // The block clock. One block is reseeded every BLOCK_FRAMES frames, so a
    // particle's age is (its block's distance behind the newest) × BLOCK_FRAMES
    // plus however far into the current block we are — which is the index into
    // the fade curve.
    // In frames of real time, not in callbacks.
    //
    // The reference advances this by `max(1, round(elapsed * 60))`, which is
    // correct only while the callback rate is the 60 Hz it assumes. MapLibre
    // hands this layer ~106 callbacks a second on a 120 Hz panel, and the
    // `max(1, ...)` then forces a whole frame per callback: the 128-frame
    // lifetime completes in 1.2 s instead of 2.13 s, every particle is reborn
    // 1.77x too often, and no trail lives long enough to become one. Advancing
    // by the real elapsed time costs nothing — the value is only ever read as
    // an index into the fade table and against the rebirth threshold.
    blockFrame += dt * TARGET_FPS;
    // The block the cursor is leaving is the one reborn, and the cursor then
    // moves past it. Reseeding the block the cursor moves *to* instead puts the
    // newest generation one slot away from where the draw loop looks for it,
    // and the whole fade curve is applied off by one block — the youngest
    // strokes drawn at the dimmest alpha.
    int reseedTarget = -1;
    if (blockFrame >= BLOCK_FRAMES) {
      blockFrame -= BLOCK_FRAMES;
      reseedTarget = stateBlock;
      stateBlock = (stateBlock + 1) % BLOCKS;
    }

    // --- projection pass: the global grid, redrawn viewport-aligned
    //
    // Once a frame, not once per vertex. After this the particles are in screen
    // space and every remaining pass is the reference's, unmodified.
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
    GLES20.glFramebufferTexture2D(
        GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, projTex, 0);
    GLES20.glViewport(0, 0, screenWidth, screenHeight);
    GLES20.glDisable(GLES20.GL_BLEND);
    GLES20.glDisable(GLES20.GL_DEPTH_TEST);
    GLES20.glDisable(GLES20.GL_STENCIL_TEST);
    GLES20.glUseProgram(projectProgram);
    bindQuad(projectProgram);
    bindTexture(GLES20.GL_TEXTURE0, windTex, projectProgram, "u_field", 0);
    uniform2f(projectProgram, "u_wind_min", windRange[0], windRange[2]);
    uniform2f(projectProgram, "u_wind_max", windRange[1], windRange[3]);
    // (speedMul, speedMin, cPow) = (1/glMaxSpeedParam,
    //  glMinSpeedParam/glMaxSpeedParam, glSpeedCurvePowParam/2).
    GLES20.glUniform3f(
        GLES20.glGetUniformLocation(projectProgram, "u_curve"), 1f / 30f, 1.5f / 30f, 0.35f);
    setProjection(projectProgram, cam);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

    // --- update pass: advect every particle into the back state texture
    GLES20.glFramebufferTexture2D(
        GLES20.GL_FRAMEBUFFER,
        GLES20.GL_COLOR_ATTACHMENT0,
        GLES20.GL_TEXTURE_2D,
        stateTex[1 - stateFront],
        0);
    GLES20.glViewport(0, 0, STATE_EDGE, STATE_EDGE);
    GLES20.glUseProgram(updateProgram);
    bindQuad(updateProgram);
    bindTexture(GLES20.GL_TEXTURE0, projTex, updateProgram, "u_wind", 0);
    bindTexture(GLES20.GL_TEXTURE1, stateTex[stateFront], updateProgram, "u_particles", 1);

    // uPars1, transcribed: the stored 0..1 direction becomes a displacement of
    // at most `o` normalised units, where `o` is how far the reference's speed
    // unit carries a particle this frame. glSpeedPx is pixels per second at
    // full strength, so the viewport's own width is the denominator — which is
    // why the speed no longer has to be retuned per zoom level.
    double seconds = dt;
    double timeScale = SPEED_PX * zoomSpeed(cam.zoom) * t.pixelRatio;
    float ox = (float) (seconds * timeScale / screenWidth);
    float oy = (float) (seconds * timeScale / screenHeight);
    GLES20.glUniform4f(
        GLES20.glGetUniformLocation(updateProgram, "u_disp"), 2 * ox, 2 * oy, -ox, -oy);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

    stateFront = 1 - stateFront;
    if (reseedTarget >= 0) {
      reseedBlock(reseedTarget);
    }

    // --- fade pass: last frame's streaks, dimmed and re-anchored, into the
    //     back trail texture
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
      GLES20.glFramebufferTexture2D(
          GLES20.GL_FRAMEBUFFER,
          GLES20.GL_COLOR_ATTACHMENT0,
          GLES20.GL_TEXTURE_2D,
          screenTex[1 - screenFront],
          0);
      trailsDirty = false;
    } else {
      GLES20.glUseProgram(quadProgram);
      bindQuad(quadProgram);
      bindTexture(GLES20.GL_TEXTURE2, screenTex[screenFront], quadProgram, "u_screen", 2);
      // Keyed to real time, not to the callback count. The reference applies
      // its fade once per frame and gets away with it because a browser's rAF
      // is the display refresh; MapLibre hands this layer ~106 callbacks a
      // second on a 120 Hz panel regardless of `setMaximumFps(60)`, and a
      // per-callback 0.97 then halves the trail length against a 60 Hz device.
      // `dt * 60` restores the intended per-second decay on any refresh rate.
      uniform1f(
          quadProgram,
          "u_opacity",
          (float) Math.pow(t.fadeOpacity(cam.zoom), dt * TARGET_FPS));
      uniform1f(quadProgram, "u_lift", 0f);
      uniform2f(quadProgram, "u_shift", trailShiftX, trailShiftY);
      GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    }

    // --- draw pass: sixteen generations of segments over the faded streaks
    //
    // Premultiplied, because the segment shader writes premultiplied alpha: the
    // generations overlap constantly and SRC_ALPHA would let the newest one
    // erase the older ones underneath it instead of adding to them.
    GLES20.glEnable(GLES20.GL_BLEND);
    GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);
    GLES20.glUseProgram(drawProgram);
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer);
    int vertLoc = GLES20.glGetAttribLocation(drawProgram, "a_vert");
    GLES20.glEnableVertexAttribArray(vertLoc);
    // Unnormalised: the shader wants the raw 0..255 so it can pick corners.
    GLES20.glVertexAttribPointer(vertLoc, 4, GLES20.GL_UNSIGNED_BYTE, false, 4, 0);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    bindTexture(GLES20.GL_TEXTURE1, stateTex[stateFront], drawProgram, "u_state0", 1);
    bindTexture(GLES20.GL_TEXTURE3, stateTex[1 - stateFront], drawProgram, "u_state1", 3);


    // Line width in NDC. Wider at high zoom, where there are fewer particles —
    // the reference's trade, and it is what keeps the apparent density even
    // across the range instead of thinning out as you zoom in.
    // Windy's physical-pixel formula: the table width is multiplied by
    // glParticleWidth and DPR, then the antialiased quad gains one physical
    // pixel. Projection below is logical, so only the final stroke converts
    // back; `u_len` deliberately remains physical for the one-pixel feather.
    float widthFactor =
        Math.max(1f, t.lineWidth(cam.zoom) * (float) t.particleWidth * (float) t.pixelRatio);
    float aw = (widthFactor + 1f) / screenWidth;
    float ah = (widthFactor + 1f) / screenHeight;
    float halfLen = Math.max(1f, widthFactor * 0.8f);
    uniform1f(drawProgram, "u_len", halfLen);
    GLES20.glUniform4f(
        GLES20.glGetUniformLocation(drawProgram, "u_line"),
        aw * 2f / 255f, ah * 2f / 255f, -aw, -ah);
    GLES20.glUniform4f(
        GLES20.glGetUniformLocation(drawProgram, "u_edge"), 0f, 0f, halfLen * 2f / 255f, -halfLen);

    // The age of the newest generation, in frames. Every older block is another
    // BLOCK_FRAMES behind it, wrapping through the 128-frame cycle.
    // The reference walks the blocks in storage order and walks the fade curve
    // backwards alongside them, so the block reseeded most recently — the one
    // just below the write cursor — lands at age ~0 and every earlier block is
    // another BLOCK_FRAMES older.
    int age = (int) blockFrame;
    for (int b = 0; b < BLOCKS; b++) {
      int block = (stateBlock - 1 - b + 2 * BLOCKS) % BLOCKS;
      GLES20.glUniform4f(
          GLES20.glGetUniformLocation(drawProgram, "u_block"),
          1f / STATE_EDGE, 1f / STATE_EDGE, 0f, block / (float) BLOCKS);
      float a = alphaLut[age % LIFETIME];
      GLES20.glUniform4f(GLES20.glGetUniformLocation(drawProgram, "u_colour"), a, a, a, a);
      GLES20.glDrawElements(
          GLES20.GL_TRIANGLES, perBlock * 6, GLES20.GL_UNSIGNED_SHORT, 0);
      age += BLOCK_FRAMES;
    }
    GLES20.glDisableVertexAttribArray(vertLoc);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
    // Did the offscreen work actually land? An FBO that is momentarily
    // incomplete, or a driver error mid-pass, produces a frame with no strokes
    // in it — which is indistinguishable from a flicker by eye and obvious in
    // a counter.
    if (GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
        != GLES20.GL_FRAMEBUFFER_COMPLETE) {
      fboIncomplete++;
    }
    int err = GLES20.glGetError();
    if (err != GLES20.GL_NO_ERROR) {
      glErrors++;
      lastGlError = err;
    }
    screenFront = 1 - screenFront;

    composite(target);

    frame++;
    if (frame == 1) {
      // Colour encoding, once. A default framebuffer in sRGB linearises the
      // destination before blending and re-encodes after, so the same
      // premultiplied grey composites to a different brightness than it does
      // against Metal's BGRA8Unorm on iOS — which does no conversion at all.
      // Same shader, same constants, different result, and nothing reports it.
      int[] enc = new int[1];
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, target.framebuffer[0]);
      GLES30.glGetFramebufferAttachmentParameteriv(
          GLES20.GL_FRAMEBUFFER,
          GLES30.GL_BACK,
          GLES30.GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING,
          enc,
          0);
      int mapEncoding = GLES20.glGetError() == GLES20.GL_NO_ERROR ? enc[0] : -1;
      int[] trailEnc = new int[1];
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
      GLES20.glFramebufferTexture2D(
          GLES20.GL_FRAMEBUFFER,
          GLES20.GL_COLOR_ATTACHMENT0,
          GLES20.GL_TEXTURE_2D,
          screenTex[screenFront],
          0);
      GLES30.glGetFramebufferAttachmentParameteriv(
          GLES20.GL_FRAMEBUFFER,
          GLES20.GL_COLOR_ATTACHMENT0,
          GLES30.GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING,
          trailEnc,
          0);
      int trailEncoding = GLES20.glGetError() == GLES20.GL_NO_ERROR ? trailEnc[0] : -1;
      String ext = GLES20.glGetString(GLES20.GL_EXTENSIONS);
      Log.i(
          TAG,
          "gl: version="
              + GLES20.glGetString(GLES20.GL_VERSION)
              + " renderer="
              + GLES20.glGetString(GLES20.GL_RENDERER)
              + " mapEncoding="
              + encodingName(mapEncoding)
              + " trailEncoding="
              + encodingName(trailEncoding)
              + " sRGBWriteControl="
              + (ext != null && ext.contains("EXT_sRGB_write_control")));
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, target.framebuffer[0]);
      Log.i(
          TAG,
          "first frame: perBlock="
              + perBlock
              + " x"
              + BLOCKS
              + " blocks field="
              + windWidth
              + "x"
              + windHeight
              + " screen="
              + screenWidth
              + "x"
              + screenHeight
              + " zoom="
              + cam.zoom
              + " state=packed-rgba8"
              + " shaders="
              + shaderFingerprint()
              + " glError="
              + GLES20.glGetError());
    } else if (frame % 600 == 0) {
      Log.i(
          TAG,
          "frames=" + frame
              + " zoom=" + cam.zoom
              + " phys=" + screenWidth + "x" + screenHeight
              + " density=" + cam.density
              + " logical=" + Math.round(logicalW) + "x" + Math.round(logicalH)
              + " amount=" + amount
              + " perBlock=" + perBlock
              + " total=" + (perBlock * BLOCKS)
              + " dpr=" + t.pixelRatio
              + String.format(
                  " shifted=%d maxTexels=%.3f deposit=%d skip=%d fboBad=%d"
                      + " glErr=%d(0x%x)",
                  shiftFrames, shiftMaxTexels, deposits, skips, fboIncomplete,
                  glErrors, lastGlError));
      shiftFrames = 0;
      shiftMaxTexels = 0;
      Log.i(
          TAG,
          String.format(
              "callback spacing: min=%.2fms max=%.2fms mean=%.2fms n=%d",
              elapsedMin, elapsedMax, elapsedSum / Math.max(1, elapsedN), elapsedN));
      elapsedMin = 1e9;
      elapsedMax = 0;
      elapsedSum = 0;
      elapsedN = 0;
      deposits = 0;
      skips = 0;
      fboIncomplete = 0;
      glErrors = 0;
    }
  }

  /** Paints the latest trail texture into this MapLibre frame. */
  private void composite(GlState target) {
    // Back to whatever the map was drawing into, not a hardcoded 0. It is 0 on
    // this renderer today, but a renderer that draws through its own FBO would
    // make a hardcoded zero paint somewhere nobody can see.
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, target.framebuffer[0]);
    target.applyColorMask();
    GLES20.glViewport(
        target.viewport[0], target.viewport[1], target.viewport[2], target.viewport[3]);
    GLES20.glDisable(GLES20.GL_DEPTH_TEST);
    GLES20.glDisable(GLES20.GL_STENCIL_TEST);
    GLES20.glEnable(GLES20.GL_BLEND);
    GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);
    GLES20.glUseProgram(quadProgram);
    bindQuad(quadProgram);
    bindTexture(GLES20.GL_TEXTURE2, screenTex[screenFront], quadProgram, "u_screen", 2);
    // The reference deposits dim strokes and composites them below full
    // strength so sixteen generations remain a weave rather than a white fill.
    uniform1f(quadProgram, "u_opacity", COMPOSITE_GAIN);
    uniform1f(quadProgram, "u_lift", COMPOSITE_LIFT);
    uniform2f(quadProgram, "u_shift", 0f, 0f);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
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
    drawProgram = link(VERT_SEGMENT, FRAG_SEGMENT);
    updateProgram = link(FRAG_UPDATE_VERT, FRAG_UPDATE);
    projectProgram = link(VERT_UPDATE, FRAG_WIND_PROJECT);
    if (quadProgram == 0 || drawProgram == 0 || updateProgram == 0 || projectProgram == 0) {
      Log.e(
          TAG,
          "shader programs failed to build (quad="
              + quadProgram
              + " draw="
              + drawProgram
              + " update="
              + updateProgram
              + ") shaders="
              + shaderFingerprint()
              + " — the compile log is above");
      deleteGlObjects();
      return false;
    }
    int[] buffers = new int[3];
    GLES20.glGenBuffers(3, buffers, 0);
    quadBuffer = buffers[0];
    vertexBuffer = buffers[1];
    indexBuffer = buffers[2];

    FloatBuffer quad = floats(new float[] {0, 0, 1, 0, 0, 1, 1, 1});
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadBuffer);
    GLES20.glBufferData(
        GLES20.GL_ARRAY_BUFFER, quad.capacity() * 4, quad, GLES20.GL_STATIC_DRAW);

    // One block's worth of quads, reused for all sixteen. Four bytes a vertex:
    // the texel it reads and which corner of the quad it is. The whole buffer
    // is 64 KB and never changes — everything that moves lives in the state
    // texture, which is the point of a GPGPU system.
    byte[] verts = new byte[PARTICLES_PER_BLOCK * 4 * 4];
    short[] inds = new short[PARTICLES_PER_BLOCK * 6];
    for (int i = 0; i < PARTICLES_PER_BLOCK; i++) {
      int x = i % STATE_EDGE;
      int y = i / STATE_EDGE;
      // side (line width) and end (tail → head) as the four corners.
      int[][] corners = {{0, 0}, {255, 0}, {255, 255}, {0, 255}};
      for (int c = 0; c < 4; c++) {
        int o = (i * 4 + c) * 4;
        verts[o] = (byte) x;
        verts[o + 1] = (byte) y;
        verts[o + 2] = (byte) corners[c][0];
        verts[o + 3] = (byte) corners[c][1];
      }
      int base = i * 4;
      int io = i * 6;
      inds[io] = (short) base;
      inds[io + 1] = (short) (base + 1);
      inds[io + 2] = (short) (base + 2);
      inds[io + 3] = (short) base;
      inds[io + 4] = (short) (base + 2);
      inds[io + 5] = (short) (base + 3);
    }
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer);
    ByteBuffer vb = ByteBuffer.allocateDirect(verts.length).order(ByteOrder.nativeOrder());
    vb.put(verts).position(0);
    GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, verts.length, vb, GLES20.GL_STATIC_DRAW);

    ByteBuffer ib = ByteBuffer.allocateDirect(inds.length * 2).order(ByteOrder.nativeOrder());
    ib.asShortBuffer().put(inds);
    ib.position(0);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    GLES20.glBufferData(
        GLES20.GL_ELEMENT_ARRAY_BUFFER, inds.length * 2, ib, GLES20.GL_STATIC_DRAW);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
    indexCount = PARTICLES_PER_BLOCK;

    int[] fb = new int[1];
    GLES20.glGenFramebuffers(1, fb, 0);
    framebuffer = fb[0];

    // Seed the packed state textures with noise: an unseeded field starts as a grid.
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
    // Immutable entropy for GPU rebirth. Unlike a sin hash, texture noise does
    // not collapse into rows when a mobile driver evaluates highp imprecisely.
    // This is uploaded once; no particle state crosses back to the CPU.
    seedTex = createTexture(GLES20.GL_NEAREST);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_REPEAT);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_REPEAT);
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
    glContext = EGL14.eglGetCurrentContext();
    return true;
  }

  /** The update pass has its own texel-preserving full-screen vertex shader. */
  private static final String FRAG_UPDATE_VERT = VERT_UPDATE;

  /**
   * Rebirth, the reference's way: overwrite one block's rows of both state
   * textures with uniform random bytes.
   *
   * <p>Uniform in the packed encoding is uniform on screen, because the packing
   * is linear — {@code pos = ba + rg/255.5} over random bytes is a flat
   * distribution over the unit square, which is now the viewport. No density
   * weighting, no view test, no rejection loop: the previous version needed all
   * three only because a respawn drawn in field space mostly landed off-screen.
   *
   * <p>Both textures, and that is the point of doing it here rather than inside
   * the update shader. A particle whose front state is new and whose back state
   * is its previous life gets a segment drawn between two unrelated points —
   * the streak from one edge of the map to the other. Writing both leaves the
   * segment zero-length for one frame instead.
   */
  private void reseedBlock(int block) {
    int bytes = STATE_EDGE * BLOCK_ROWS * 4;
    byte[] noise = new byte[bytes];
    reseedRandom.nextBytes(noise);
    ByteBuffer buf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
    buf.put(noise).position(0);
    for (int i = 0; i < 2; i++) {
      GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, stateTex[i]);
      GLES20.glTexSubImage2D(
          GLES20.GL_TEXTURE_2D,
          0,
          0,
          block * BLOCK_ROWS,
          STATE_EDGE,
          BLOCK_ROWS,
          GLES20.GL_RGBA,
          GLES20.GL_UNSIGNED_BYTE,
          buf);
    }
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
  }

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
    // The reprojected field shares the trails' resolution. It is resampled by
    // the update pass at arbitrary particle positions, so LINEAR — NEAREST here
    // quantises the flow into visible cells.
    if (projTex != 0) {
      GLES20.glDeleteTextures(1, new int[] {projTex}, 0);
    }
    projTex = createTexture(GLES20.GL_LINEAR);
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
    screenWidth = width;
    screenHeight = height;
    markTrailsDirty("viewport " + width + "x" + height);
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
      // LINEAR, not NEAREST. The reprojection pass resamples this grid at one
      // screen pixel per texel, and ECMWF's cells are degrees wide — nearest
      // would hand the particles a staircase and they would move in squares.
      windTex = createTexture(GLES20.GL_LINEAR);
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
      GLES20.glDeleteProgram(projectProgram);
      updateProgram = 0;
    projectProgram = 0;
    }
    if (quadBuffer != 0) {
      GLES20.glDeleteBuffers(3, new int[] {quadBuffer, vertexBuffer, indexBuffer}, 0);
      quadBuffer = 0;
      vertexBuffer = 0;
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
    if (projTex != 0) {
      GLES20.glDeleteTextures(1, new int[] {projTex}, 0);
      projTex = 0;
    }
    if (seedTex != 0) {
      GLES20.glDeleteTextures(1, new int[] {seedTex}, 0);
      seedTex = 0;
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

  /**
   * A short hash of the shader sources actually compiled into this build.
   *
   * <p>Exists because a shader bug and a stale install look identical from the
   * log: both report the same link failure forever. Twice now the source on
   * disk was demonstrably correct while the device kept printing the old
   * error, and the only way to tell the two apart was to notice the process id
   * had not changed. This turns that into a fact the log states itself — the
   * number changes whenever a shader does, so a report that quotes the old one
   * is a report from the old binary.
   */
  private static String shaderFingerprint() {
    int h = 17;
    for (String src :
        new String[] {
          VERT_QUAD, VERT_UPDATE, FRAG_SCREEN, VERT_SEGMENT, FRAG_SEGMENT, FRAG_UPDATE
        }) {
      h = h * 31 + src.hashCode();
    }
    return Integer.toHexString(h);
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
    private final boolean[] colorMask = new boolean[4];
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
      GLES20.glGetBooleanv(GLES20.GL_COLOR_WRITEMASK, s.colorMask, 0);
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
      applyColorMask();
      setEnabled(GLES20.GL_BLEND, blend);
      setEnabled(GLES20.GL_DEPTH_TEST, depth);
      setEnabled(GLES20.GL_STENCIL_TEST, stencil);
    }

    void applyColorMask() {
      GLES20.glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
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
