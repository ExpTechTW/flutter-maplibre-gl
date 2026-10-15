package org.maplibre.maplibregl;

import android.opengl.EGL14;
import android.opengl.EGLContext;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.util.Log;
import android.view.Choreographer;
import androidx.annotation.Nullable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.maplibre.android.maps.MapLibreMap;
import org.maplibre.android.maps.MapView;

/**
 * GPU wind-particle field drawn inside the map's own GL pass, the way Windy's
 * {@code gl-particles} draws it.
 *
 * <h3>Why it lives in the map</h3>
 *
 * It used to be a Flutter overlay. Under HCPP every Flutter frame presented
 * above a platform view allocates a full-screen {@code AHardwareBuffer} that is
 * never returned — 394 MB to 8 GB of GPU memory in sixteen seconds on a Pixel
 * 9 — and an animated overlay presents one every vsync. Drawn here, the field
 * costs Flutter no frame at all and moves in the same pass as the map.
 *
 * <h3>The pipeline</h3>
 *
 * Everything lives in one <em>reference frame</em>: the viewport as it was the
 * last time the camera came to rest. That is Windy's {@code lastBounds}, and
 * the reason every number below can be a screen-space number.
 *
 * <ol>
 *   <li><b>Velocity.</b> When the camera settles (or a new field lands), the
 *       global grid is resampled once under every pixel of the reference
 *       viewport, rotated into screen axes and bent by Windy's speed curve.
 *       Nothing reprojects per frame.
 *   <li><b>Particles.</b> A 256² RGBA8 texture of 16-bit positions in the
 *       reference viewport, advected on the GPU by sampling (1). Sixteen
 *       blocks, one reborn every eight frames: every particle lives 2.13 s and
 *       is faded along that life by {@link #ALPHA_LUT}.
 *   <li><b>Trails.</b> One RGBA8 texture at 0.8× the screen on dense displays.
 *       Each frame the newest segment of every particle is drawn into it, then
 *       the whole texture is multiplied by 0.97 in place with a constant-alpha
 *       blend.
 *   <li><b>Composite.</b> The trail texture is drawn over the map as a quad
 *       placed where the reference viewport now is. During a pan or a pinch
 *       the field therefore travels with the map; nothing is cleared or paused.
 * </ol>
 *
 * When the camera comes to rest the reference moves to it: after a pan the
 * trails and particles are shifted into the new frame (no visible jump), after
 * a zoom or a rotation the trails are cleared and grow back — Windy's
 * {@code moveend} and {@code zoomend}, derived here from consecutive frames
 * because a custom layer receives no camera events.
 *
 * <h3>Frames</h3>
 *
 * A custom layer does not ask the map to redraw; MapLibre renders it whenever
 * something else does. While playing, a main-thread {@link Choreographer}
 * callback calls {@link MapLibreMap#triggerRepaint()} at most 60 times a second
 * — Windy's {@code requestAnimationFrame} loop — and stops when playback does.
 */
final class WindParticleLayer {

  private static final String TAG = "WindParticleLayer";

  // ------------------------------------------------------------ the reference
  //
  // Every number in this section is Windy's, from the wind entry of its
  // particle configuration and from `updateParamsFromConfig`. They are what
  // makes the field look the way it does; change them together or not at all.

  /** Particle-state texture edge: 65,536 slots, of which zoom draws a part. */
  private static final int STATE_EDGE = 256;

  private static final float INV_STATE_EDGE = 1f / STATE_EDGE;

  /**
   * Sixteen blocks, one reborn every eight frames: a 128-frame (2.13 s) life
   * whose age is always known, which is what {@link #ALPHA_LUT} fades along.
   */
  private static final int BLOCKS = 16;

  private static final int BLOCK_FRAMES = 8;
  private static final int BLOCK_ROWS = STATE_EDGE / BLOCKS;
  private static final int LIFETIME = BLOCKS * BLOCK_FRAMES;
  private static final int PARTICLES_PER_BLOCK = STATE_EDGE * BLOCK_ROWS;

  /** In over the first 20 % of a life (power 0.9), out over the last 30 % (0.8). */
  private static final float[] ALPHA_LUT = buildAlphaLut(0.2f, 0.9f, 0.3f, 0.8f);

  /** {@code getAmount}: {@code min(15000, w·h / (50 · 1.6^(z−2)))} in CSS pixels. */
  private static final double AMOUNT_MAX = 15000;

  private static final double AMOUNT_CONSTANT = 50;
  private static final double AMOUNT_POW = 1.6;
  private static final double AMOUNT_ZOOM = 2;

  /** Windy halves the count off desktop; {@code glCountMul} for wind is 1. */
  private static final double COUNT_MUL = 0.5;

  /** {@code lineWidth}, indexed by Windy zoom. */
  private static final double[] LINE_WIDTH = {
    .6, .6, .6, 1, 1.2, 1.6, 1.8, 2, 2.2, 2.4, 2.4, 2.4, 2.4, 2.6, 2.8, 3, 3, 3, 3, 3, 3, 3, 3, 3
  };

  /** {@code glParticleWidth}. */
  private static final double PARTICLE_WIDTH = 1.3;

  /** {@code zoom2speed}, indexed by Windy zoom: wind is slowed when zoomed out. */
  private static final double[] ZOOM_TO_SPEED = {
    .5, .5, .5, .6, .7, .8, .9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };

  /** {@code glSpeedPx}: physical pixels per second at full curve strength. */
  private static final double SPEED_PX = 100;

  /**
   * The speed curve: {@code 1/glMaxSpeedParam}, {@code glMinSpeedParam /
   * glMaxSpeedParam} and {@code glSpeedCurvePowParam / 2}. Displacement goes
   * as {@code (|v|/30)^0.7}, floored at 0.05 and — unlike the reference, whose
   * byte encoding overflows past 30 m/s and skews the direction — capped at 1.
   */
  private static final float CURVE_MUL = 1f / 30f;

  private static final float CURVE_MIN = 1.5f / 30f;
  private static final float CURVE_POW = 0.35f;

  /**
   * Trail decay per 60 Hz frame. The reference derives it from the blending
   * setting, {@code min(0.9 + 0.5 · (1.06 − 0.92), 0.98)} on mobile.
   */
  private static final double FADE = 0.97;

  /**
   * {@code copyToCanvas}: trails reach the map at 0.44 with 0.1 subtracted.
   * The subtraction is what hides the residue an 8-bit fade cannot remove —
   * {@code round(v · 0.97)} stops decreasing below 17/255 — so it is not a
   * matter of taste.
   */
  private static final float COMPOSITE_GAIN = 0.44f;

  private static final float COMPOSITE_LIFT = -0.1f;

  /** The whole layer fades in over 0.55 s when playback starts. */
  private static final double ALPHA_RATE = 1.8;

  /** Above this density the trails are kept at {@link #TRAIL_SCALE} of the screen. */
  private static final double TRAIL_SCALE_DPR = 1.5;

  private static final double TRAIL_SCALE = 0.8;
  private static final int TRAIL_MAX = 2048;

  /** A stored velocity byte b is (b − 128) / 127 of full speed. */
  private static final float DISP_MUL = 255f / 127f;

  private static final float DISP_ADD = 128f / 127f;

  /**
   * How long the camera must hold still to count as settled. Long enough that
   * a finger resting mid-gesture is not mistaken for the end of one; short
   * enough that a settled view is re-projected before anyone looks for it.
   */
  private static final double SETTLE_SECONDS = 0.1;

  /** The rate the frame budget, the block clock and the fade are counted in. */
  private static final int TARGET_FPS = 60;

  private static final double STEP = 1.0 / TARGET_FPS;
  private static final long STEP_NANOS = 1_000_000_000L / TARGET_FPS;

  /**
   * The reference's zoom: Leaflet's, one above MapLibre's.
   *
   * <p>Kept fractional. The reference rounds it, but its map snaps to whole
   * zooms ({@code zoomSnap: 1}), so it only ever draws at the values where
   * rounding changes nothing. This map rests anywhere, and rounding there makes
   * the field jump by a factor of 1.6 every time the zoom crosses a half — the
   * framed-Taiwan view at 4.67 drew the density meant for a whole zoom closer.
   * Continuous, every table and the count agree with the reference exactly at
   * whole zooms and move smoothly between them.
   */
  private static double windyZoom(double zoom) {
    return zoom + 1;
  }

  /** {@code table} at a fractional index, linear between entries, clamped at the ends. */
  private static double at(double[] table, double index) {
    double i = Math.max(0, Math.min(table.length - 1, index));
    int lo = (int) Math.floor(i);
    int hi = Math.min(table.length - 1, lo + 1);
    return table[lo] + (table[hi] - table[lo]) * (i - lo);
  }

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

  // ---------------------------------------------------------------- shaders
  //
  // GLSL ES 1.00. Every name that crosses the vertex/fragment boundary carries
  // an explicit precision qualifier: the two stages default differently, and a
  // mismatch is a link error that names a variable that looks right in each
  // file on its own.
  //
  // One convention throughout: "particle space" is the reference viewport as
  // 0..1 with y up. It is what a particle stores, what a full-screen pass sees
  // as `v_p`, and — because a GL texture's t = 0 is its first row and a
  // framebuffer's first row is clip y = -1 — also the texture coordinate of
  // every screen-sized texture here. The iOS twin differs exactly there.

  /** Full-screen quad over whatever is bound; {@code v_p} is particle space. */
  private static final String VERT_FULL =
      "precision highp float;\n"
          + "attribute vec2 a_pos;\n"
          + "varying highp vec2 v_p;\n"
          + "void main() {\n"
          + "  v_p = a_pos;\n"
          + "  gl_Position = vec4(a_pos * 2.0 - 1.0, 0.0, 1.0);\n"
          + "}";

  /** The fade: its output is multiplied by ZERO, only the blend constant counts. */
  private static final String FRAG_ZERO =
      "precision mediump float;\n" + "void main() {\n" + "  gl_FragColor = vec4(0.0);\n" + "}";

  /**
   * Copies the trails into a reference frame that moved by {@code u_shift}.
   * Whole texels only, so the copy is exact rather than a blur; what scrolled
   * in from outside has no history.
   */
  private static final String FRAG_SHIFT =
      "precision highp float;\n"
          + "uniform sampler2D u_src;\n"
          + "uniform vec2 u_shift;\n"
          + "varying highp vec2 v_p;\n"
          + "void main() {\n"
          + "  vec2 p = v_p - u_shift;\n"
          + "  if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) {\n"
          + "    gl_FragColor = vec4(0.0);\n"
          + "  } else {\n"
          + "    gl_FragColor = texture2D(u_src, p);\n"
          + "  }\n"
          + "}";

  /**
   * Position encoding shared by the state passes: high byte in {@code ba},
   * low byte in {@code rg}, so a position has 16 bits per axis. An all-zero
   * texel is a dead particle.
   */
  private static final String STATE_GLSL =
      "vec4 readState(sampler2D s) {\n"
          + "  return texture2D(s, gl_FragCoord.xy * (1.0 / 256.0));\n"
          + "}\n"
          + "bool isDead(vec4 t) { return t.r + t.g + t.b + t.a < 0.025; }\n"
          + "vec2 decode(vec4 t) { return t.ba + t.rg / 255.5; }\n"
          + "vec4 encode(vec2 pos) {\n"
          + "  vec2 lo = fract(pos * 255.0 + 0.25 / 255.0);\n"
          + "  return vec4(lo, pos - lo / 255.0);\n"
          + "}\n";

  /**
   * Moves every live particle by {@code u_shift} — the reference frame moved
   * under it. Run twice when the frame moves, into the back texture and back
   * again, so the segment drawn next frame joins two positions in the same
   * frame. With a zero shift it is a copy.
   */
  private static final String FRAG_STATE_SHIFT =
      "precision highp float;\n"
          + "uniform sampler2D u_state;\n"
          + "uniform vec2 u_shift;\n"
          + STATE_GLSL
          + "void main() {\n"
          + "  vec4 t = readState(u_state);\n"
          + "  gl_FragColor = isDead(t) ? vec4(0.0) : encode(fract(decode(t) + u_shift));\n"
          + "}";

  /**
   * The reference's advection. A particle reads the velocity under it and
   * moves by at most {@code o} of the viewport per frame; one with no data
   * under it holds, one with no wind under it dies, and a dead one stays dead
   * until its block is reborn. (The reference lets a dead particle resume
   * from the corner its zeroed position names.)
   */
  private static final String FRAG_UPDATE =
      "precision highp float;\n"
          + "uniform sampler2D u_state;\n"
          + "uniform sampler2D u_vel;\n"
          + "uniform vec4 u_disp;\n" // byte b moves by (b - 128) / 127 · o
          + STATE_GLSL
          + "void main() {\n"
          + "  vec4 t = readState(u_state);\n"
          + "  if (isDead(t)) { gl_FragColor = vec4(0.0); return; }\n"
          + "  vec2 pos = decode(t);\n"
          + "  vec4 w = texture2D(u_vel, fract(pos));\n"
          + "  if (w.a < 0.5) { gl_FragColor = t; return; }\n"
          + "  vec2 dpos = w.rg * u_disp.xy + u_disp.zw;\n"
          + "  if (max(abs(dpos.x), abs(dpos.y)) < 0.000025) {\n"
          + "    gl_FragColor = vec4(0.0);\n"
          + "    return;\n"
          + "  }\n"
          + "  gl_FragColor = encode(fract(pos + dpos));\n"
          + "}";

  /**
   * The global grid, resampled under every pixel of the reference viewport.
   *
   * <p>The reference stitches this from wind tiles; DPIP has one global grid,
   * so the map's projection is inverted here instead — unrotate, then
   * unproject — and the wind is rotated into screen axes, which is what lets
   * the rest of the pipeline ignore the bearing.
   *
   * <p>Encoded as {@code (128 + 127 · direction · magnitude) / 255}, so a calm
   * pixel is exactly byte 128 and decodes to zero. (The reference's
   * {@code 128 ± 128} cannot represent zero in eight bits and drifts every
   * particle toward the top right.) Alpha 0 means no data under this pixel; the
   * neutral RG keeps a linearly filtered edge from pulling neighbouring
   * particles toward one corner.
   *
   * <p>The half-texel term aligns the sample with the grid's nodes: texel
   * {@code i} is centred on {@code (i + 0.5) / n}, and it holds the value at
   * {@code lon0 + i · dLon}. Without it the field sits half a cell (0.125°)
   * west and north of the raster drawn from the same data.
   */
  private static final String FRAG_PROJECT =
      "precision highp float;\n"
          + "const float PI = 3.141592653589793;\n"
          + "uniform sampler2D u_field_u;\n"
          + "uniform sampler2D u_field_v;\n"
          + "uniform vec4 u_range;\n" // (uMin, vMin, uMax, vMax)
          + "uniform vec4 u_grid;\n" // (lon0, lat0, latSpan, -)
          + "uniform vec2 u_half_texel;\n"
          + "uniform vec3 u_curve;\n" // (speedMul, speedMin, cPow)
          + "uniform float u_world;\n" // 512 · 2^zoom
          + "uniform vec2 u_center;\n" // camera centre, world px
          + "uniform vec2 u_half;\n" // viewport half size, logical px
          + "uniform vec2 u_rot;\n" // cos(-bearing), sin(-bearing)
          + "varying highp vec2 v_p;\n"
          + "float mercLat(float y) {\n"
          + "  return degrees(2.0 * atan(exp(PI * (1.0 - 2.0 * y))) - PI / 2.0);\n"
          + "}\n"
          + "void main() {\n"
          + "  vec2 d = vec2(v_p.x * 2.0 - 1.0, 1.0 - v_p.y * 2.0) * u_half;\n"
          + "  float wx = u_center.x + d.x * u_rot.x + d.y * u_rot.y;\n"
          + "  float wy = u_center.y - d.x * u_rot.y + d.y * u_rot.x;\n"
          + "  float lon = wx / u_world * 360.0 - 180.0;\n"
          + "  float lat = mercLat(wy / u_world);\n"
          + "  vec2 uv = vec2(fract((lon - u_grid.x) / 360.0),\n"
          + "                 (lat - u_grid.y) / u_grid.z) + u_half_texel;\n"
          + "  if (uv.y < 0.0 || uv.y > 1.0) {\n"
          + "    gl_FragColor = vec4(128.0 / 255.0, 128.0 / 255.0, 0.0, 0.0);\n"
          + "    return;\n"
          + "  }\n"
          + "  vec2 raw = vec2(texture2D(u_field_u, uv).r, texture2D(u_field_v, uv).r);\n"
          + "  vec2 v = mix(u_range.xy, u_range.zw, raw);\n"
          // Into screen axes, y up: north is +y with the map pointing north.
          + "  v = vec2(v.x * u_rot.x + v.y * u_rot.y, -v.x * u_rot.y + v.y * u_rot.x);\n"
          + "  vec2 o = v * u_curve.x;\n"
          + "  float d2 = dot(o, o);\n"
          + "  if (d2 < 1e-8) {\n"
          + "    gl_FragColor = vec4(128.0 / 255.0, 128.0 / 255.0, 0.0, 1.0);\n"
          + "    return;\n"
          + "  }\n"
          + "  float mag = min(max(pow(d2, u_curve.z), u_curve.y), 1.0);\n"
          + "  vec2 dir = o * (mag * inversesqrt(d2));\n"
          + "  gl_FragColor = vec4((128.0 + 127.0 * dir) / 255.0, 0.0, 1.0);\n"
          + "}";

  /**
   * One particle as a quad from where it was to where it is — so every stroke
   * lies along its own motion, and sixteen overlapping generations of them
   * read as streamlines. {@code a_vert} is (texel x, texel row within the
   * block, side, end); {@code u_block.zw} slides the lookup onto a block.
   */
  private static final String VERT_SEGMENT =
      "precision highp float;\n"
          + "attribute vec4 a_vert;\n"
          + "uniform sampler2D u_state0;\n"
          + "uniform sampler2D u_state1;\n"
          + "uniform vec4 u_block;\n" // (1/256, 1/256, 0, block/16)
          + "uniform vec4 u_line;\n" // (w·2/255, h·2/255, -w, -h)
          + "uniform vec4 u_edge;\n" // (0, 0, l·2/255, -l)
          + "varying mediump float v_discard;\n"
          + "varying mediump float v_edge;\n"
          + "void main() {\n"
          // Texel centres: on a boundary, nearest filtering may pair this
          // particle's position with its neighbour's.
          + "  vec2 tc = (a_vert.xy + 0.5) * u_block.xy + u_block.zw;\n"
          + "  vec4 tex0 = texture2D(u_state0, tc);\n"
          + "  vec4 tex1 = texture2D(u_state1, tc);\n"
          + "  v_discard = step(0.025, tex0.r + tex0.g + tex0.b + tex0.a);\n"
          + "  vec2 posA = fract(tex0.ba + tex0.rg / 255.5) * 2.0 - 1.0;\n"
          + "  vec2 posB = fract(tex1.ba + tex1.rg / 255.5) * 2.0 - 1.0;\n"
          + "  vec2 dirF = posA - posB;\n"
          + "  float d = length(dirF);\n"
          + "  vec2 dirFN = dirF / max(d, 1e-20);\n"
          + "  vec2 pos = mix(posB, posA, a_vert.w * 0.003921569);\n"
          + "  pos += vec2(dirFN.y, -dirFN.x) * (a_vert.zz * u_line.xy + u_line.zw);\n"
          // Wrapped across an edge (the quad would span the screen), or zero
          // length — a rebirth, or a shift — where the normal is NaN.
          + "  if (d > 0.5 || d < 1e-5) { pos.x += 10.0; }\n"
          + "  gl_Position = vec4(pos, 0.0, 1.0);\n"
          + "  v_edge = u_edge.z * a_vert.z + u_edge.w;\n"
          + "}";

  /** White, premultiplied, feathered across the width. */
  private static final String FRAG_SEGMENT =
      "precision mediump float;\n"
          + "uniform vec4 u_colour;\n"
          + "uniform mediump float u_len;\n"
          + "varying mediump float v_discard;\n"
          + "varying mediump float v_edge;\n"
          + "void main() {\n"
          + "  if (v_discard <= 0.0) { discard; }\n"
          + "  gl_FragColor = u_colour * clamp(u_len - abs(v_edge), 0.0, 1.0);\n"
          + "}";

  /** The trail quad, placed where the reference viewport is in this frame. */
  private static final String VERT_COMPOSITE =
      "precision highp float;\n"
          + "attribute vec2 a_pos;\n"
          + "uniform vec4 u_affine;\n" // (origin, x axis) in clip space
          + "uniform vec2 u_axis_y;\n"
          + "varying highp vec2 v_tex;\n"
          + "void main() {\n"
          + "  v_tex = a_pos;\n"
          + "  gl_Position = vec4(u_affine.xy + a_pos.x * u_affine.zw + a_pos.y * u_axis_y,"
          + " 0.0, 1.0);\n"
          + "}";

  private static final String FRAG_COMPOSITE =
      "precision mediump float;\n"
          + "uniform sampler2D u_trail;\n"
          + "uniform float u_gain;\n"
          + "uniform float u_lift;\n"
          + "varying highp vec2 v_tex;\n"
          + "void main() {\n"
          + "  gl_FragColor = max(texture2D(u_trail, v_tex) * u_gain + u_lift, 0.0);\n"
          + "}";

  // ------------------------------------------------------------------ state

  private final MapView mapView;
  private final MapLibreMap map;

  /** Logical-to-physical scale; the renderer reports the viewport in logical points. */
  private final double density;

  private volatile FieldUpload pendingField;
  private volatile boolean playing;
  private volatile boolean restart;
  private volatile boolean released;

  /** The style layer MapLibre drives; null until {@link #attach} succeeds. */
  @Nullable private org.maplibre.android.style.layers.CustomLayer styleLayer;

  /** Camera as the renderer hands it over this frame, at the last deposit, and the reference. */
  private final Camera cam = new Camera();

  private final Camera lastCam = new Camera();
  private final Camera ref = new Camera();
  private boolean haveRef;
  private double stillTime;

  private final GlState savedGl = new GlState();
  @Nullable private EGLContext glContext;

  // GL objects, only ever touched on the GL thread.
  private int fadeProgram;
  private int shiftProgram;
  private int stateProgram;
  private int updateProgram;
  private int projectProgram;
  private int segmentProgram;
  private int compositeProgram;
  private int quadBuffer;
  private int vertexBuffer;
  private int indexBuffer;
  private int framebuffer;
  private final int[] stateTex = new int[2];
  private final int[] trailTex = new int[2];
  private int velTex;
  private int fieldUTex;
  private int fieldVTex;
  private int stateFront;
  private int trailFront;
  private int trailWidth;
  private int trailHeight;
  private int velWidth;
  private int velHeight;
  private int physicalWidth;
  private int physicalHeight;

  // Uniform locations, resolved once per link.
  private int uShiftSrc;
  private int uShiftShift;
  private int uStateState;
  private int uStateShift;
  private int uUpdateState;
  private int uUpdateVel;
  private int uUpdateDisp;
  private int uFieldU;
  private int uFieldV;
  private int uRange;
  private int uGrid;
  private int uHalfTexel;
  private int uCurve;
  private int uWorld;
  private int uCenter;
  private int uHalf;
  private int uRot;
  private int uState0;
  private int uState1;
  private int uBlock;
  private int uLine;
  private int uEdge;
  private int uColour;
  private int uLen;
  private int uAffine;
  private int uAxisY;
  private int uTrail;
  private int uGain;
  private int uLift;

  // The field.
  private boolean fieldReady;
  private int fieldWidth;
  private int fieldHeight;
  private final float[] fieldRange = new float[4];
  private float fieldLon0;
  private float fieldLat0;
  private float fieldLatSpan;

  /** Pending work for the next deposit. */
  private boolean needClear = true;

  private boolean velocityDirty = true;
  private boolean shiftPending;
  private float shiftX;
  private float shiftY;

  /** Which block is reborn next, and how far into its eight frames the clock is. */
  private int stateBlock;

  private double blockFrame;
  private long lastFrameNanos;
  private double depositTimer;

  /** Windy's global alpha: 0 when playback starts, up to 1 at {@link #ALPHA_RATE}. */
  private double alpha;

  private final Random random = new Random();
  private final byte[] reseedBytes = new byte[STATE_EDGE * BLOCK_ROWS * 4];
  private final ByteBuffer reseedBuffer =
      ByteBuffer.allocateDirect(STATE_EDGE * BLOCK_ROWS * 4).order(ByteOrder.nativeOrder());

  /** Composite placement, clip space: origin, x axis, y axis. Reused every frame. */
  private final float[] affine = new float[6];

  private final int[] scratch = new int[2];

  // Diagnostics, reported every 600 frames.
  private long frame;
  private int deposits;
  private int skips;
  private int settles;
  private int glErrors;
  private int lastGlError;

  static {
    System.loadLibrary("dpip_wind_host");
  }

  /**
   * Creates the C++ {@code CustomLayerHost} that forwards the map's render
   * callbacks here, as the {@code long} that {@code CustomLayer(String, long)}
   * takes. Ownership passes to MapLibre.
   */
  private static native long nativeCreateHost(WindParticleLayer layer);

  /** Camera and viewport for one frame, with its projection inputs. */
  static final class Camera {
    double lat;
    double lng;
    double zoom;
    double bearing;

    /** Viewport in logical points — the space MapLibre's zoom is defined in. */
    double width;

    double height;

    /** {@code 512 · 2^zoom}, the centre in world pixels, and cos/sin of −bearing. */
    double world;

    double cx;
    double cy;
    double cos = 1;
    double sin;

    void set(double lat, double lng, double zoom, double bearing, double width, double height) {
      this.lat = lat;
      this.lng = lng;
      this.zoom = zoom;
      this.bearing = bearing;
      this.width = width;
      this.height = height;
      world = 512.0 * Math.pow(2.0, zoom);
      cx = (lng + 180.0) / 360.0 * world;
      cy = mercatorY(lat) * world;
      double r = -bearing * Math.PI / 180.0;
      cos = Math.cos(r);
      sin = Math.sin(r);
    }

    void copyFrom(Camera o) {
      lat = o.lat;
      lng = o.lng;
      zoom = o.zoom;
      bearing = o.bearing;
      width = o.width;
      height = o.height;
      world = o.world;
      cx = o.cx;
      cy = o.cy;
      cos = o.cos;
      sin = o.sin;
    }

    boolean sameAs(Camera o) {
      return lat == o.lat
          && lng == o.lng
          && zoom == o.zoom
          && bearing == o.bearing
          && width == o.width
          && height == o.height;
    }

    /** Same scale, rotation and size: the two differ by a translation only. */
    boolean sameFrame(Camera o) {
      return Math.abs(zoom - o.zoom) < 1e-6
          && Math.abs(bearing - o.bearing) < 1e-6
          && width == o.width
          && height == o.height;
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

  WindParticleLayer(MapView mapView, MapLibreMap map) {
    this.mapView = mapView;
    this.map = map;
    double d = mapView.getResources().getDisplayMetrics().density;
    this.density = d > 0 ? d : 1;
  }

  // --------------------------------------------------------------- lifecycle

  /** Adds the custom layer to {@code style}; false when there is none or it is refused. */
  boolean attach(@Nullable org.maplibre.android.maps.Style style) {
    if (style == null) {
      Log.w(TAG, "no style yet; wind particles need one to hold the layer");
      return false;
    }
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

  /**
   * Platform thread. Starts or stops the frame pump. A new playback starts
   * from a fresh reference and fades in: the camera may have gone anywhere
   * while nothing was drawn.
   */
  void setPlaying(boolean value) {
    if (playing == value || released) {
      return;
    }
    if (value) {
      restart = true;
    }
    playing = value;
    if (value) {
      startPump();
    } else {
      stopPump();
    }
    map.triggerRepaint();
  }

  /** Platform thread. Uploaded on the next GL frame; the trails carry over, as in the reference. */
  void setField(FieldUpload upload) {
    pendingField = upload;
    map.triggerRepaint();
  }

  /**
   * Platform thread. Takes the layer out of the style, which is what makes
   * MapLibre call {@link #onNativeDeinitialize} on the GL thread — and what
   * lets a later add reuse the layer id instead of colliding with this one.
   */
  void release() {
    if (released) {
      return;
    }
    released = true;
    playing = false;
    stopPump();
    org.maplibre.android.maps.Style style = map.getStyle();
    if (style != null && styleLayer != null) {
      try {
        style.removeLayer(styleLayer);
      } catch (RuntimeException e) {
        // The map is being torn down with its style; the layer goes with it.
        Log.w(TAG, "wind layer already gone with its style", e);
      }
    }
    styleLayer = null;
  }

  // ------------------------------------------------------------ frame pump

  private boolean pumping;
  private long nextRepaintNanos;

  /**
   * Asks for one map frame per 60 Hz step on the display's own clock.
   *
   * <p>Scheduled against a running deadline rather than the last request, so a
   * 90 Hz panel alternates one and two vsyncs and still averages 60, and a
   * 120 Hz panel asks on every other vsync instead of rendering the whole map
   * twice as often as the field changes.
   */
  private final Choreographer.FrameCallback pump =
      new Choreographer.FrameCallback() {
        @Override
        public void doFrame(long frameTimeNanos) {
          if (!pumping) {
            return;
          }
          if (frameTimeNanos >= nextRepaintNanos - STEP_NANOS / 4) {
            nextRepaintNanos += STEP_NANOS;
            if (nextRepaintNanos < frameTimeNanos) {
              nextRepaintNanos = frameTimeNanos + STEP_NANOS;
            }
            map.triggerRepaint();
          }
          Choreographer.getInstance().postFrameCallback(this);
        }
      };

  private void startPump() {
    if (pumping) {
      return;
    }
    pumping = true;
    nextRepaintNanos = 0;
    Choreographer.getInstance().postFrameCallback(pump);
  }

  private void stopPump() {
    if (!pumping) {
      return;
    }
    pumping = false;
    Choreographer.getInstance().removeFrameCallback(pump);
  }

  // ------------------------------------------- called from the render thread

  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeInitialize() {
    // Nothing eager: the first frame creates what it needs.
  }

  /** One map frame, inside the map's render pass, with the camera of this frame. */
  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeRender(
      double width, double height, double lat, double lng, double zoom, double bearing) {
    if (released || !playing) {
      return;
    }
    if (restart) {
      restart = false;
      haveRef = false;
      lastFrameNanos = 0;
      depositTimer = 0;
      alpha = 0;
    }
    cam.set(lat, lng, zoom, bearing, width, height);
    try {
      drawFrame();
    } catch (RuntimeException e) {
      // Never let this escape: it runs inside the map's render loop.
      Log.e(TAG, "wind particle frame failed", e);
    }
  }

  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeContextLost() {
    forgetGlObjects();
  }

  @SuppressWarnings("unused") // called from wind_particle_host.cpp
  void onNativeDeinitialize() {
    deleteGlObjects();
  }

  // ------------------------------------------------------------------ render

  private void drawFrame() {
    if (cam.width <= 0 || cam.height <= 0) {
      return;
    }
    if (contextLost()) {
      forgetGlObjects();
    }
    savedGl.capture();
    try {
      // Before anything is created: a buffer bound under the map's VAO would
      // rebind that VAO's index buffer.
      prepareState();
      if (fadeProgram == 0 && !createGlObjects()) {
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
      ensureTargets();

      long now = System.nanoTime();
      double elapsed = STEP;
      if (lastFrameNanos != 0) {
        elapsed = Math.max(0, Math.min(0.1, (now - lastFrameNanos) / 1e9));
      }
      lastFrameNanos = now;
      depositTimer += elapsed;

      // Two rates. The simulation deposits at 60 Hz: on a 120 Hz frame a
      // particle moves about a pixel, and an RGBA8 trail re-blended twice as
      // often in steps that small develops a stipple. The composite runs on
      // every map frame, or the field would vanish from the ones in between.
      // Three quarters of a step, because a 60 Hz panel's frames land a hair
      // short of 1/60 s and a strict test would deposit on every other one.
      if (depositTimer >= STEP * 0.75 || !haveRef) {
        double dt = Math.min(depositTimer, 0.1);
        depositTimer = 0;
        settleCamera(dt);
        simulate(dt);
        deposits++;
      } else {
        skips++;
      }
      composite();
      report();
    } finally {
      savedGl.restore();
    }
  }

  /** The state every pass here assumes, whatever the map's last layer left. */
  private void prepareState() {
    GLES30.glBindVertexArray(0);
    GLES20.glDisable(GLES20.GL_DEPTH_TEST);
    GLES20.glDisable(GLES20.GL_STENCIL_TEST);
    GLES20.glDisable(GLES20.GL_SCISSOR_TEST);
    GLES20.glDisable(GLES20.GL_CULL_FACE);
    GLES20.glColorMask(true, true, true, true);
    GLES20.glBlendEquation(GLES20.GL_FUNC_ADD);
  }

  /**
   * Windy's {@code moveend} and {@code zoomend}, inferred: the camera that has
   * not moved since the last deposit is at rest, and if it is not where the
   * reference is, the reference follows it.
   *
   * <p>Only then. While the camera moves, the particles keep flowing through
   * the reference viewport and the composite carries that viewport with the
   * map, so a gesture costs no projection and loses no trail.
   */
  private void settleCamera(double dt) {
    if (cam.sameAs(lastCam)) {
      stillTime += dt;
    } else {
      lastCam.copyFrom(cam);
      stillTime = 0;
    }
    boolean still = stillTime >= SETTLE_SECONDS;
    if (!haveRef) {
      lastCam.copyFrom(cam);
      ref.copyFrom(cam);
      haveRef = true;
      needClear = true;
      velocityDirty = true;
      shiftPending = false;
    } else if (still && !cam.sameAs(ref)) {
      settles++;
      if (cam.sameFrame(ref) && !needClear) {
        // A pan. The old frame's origin lands somewhere in the new one; that
        // offset, in whole trail texels, is what the trails and the particles
        // move by.
        refToClip(0, 0, cameraWrap(), affine, 0);
        shiftX = Math.round((affine[0] + 1) * 0.5 * trailWidth) / (float) trailWidth;
        shiftY = Math.round((affine[1] + 1) * 0.5 * trailHeight) / (float) trailHeight;
        shiftPending = true;
      } else {
        needClear = true;
        shiftPending = false;
      }
      ref.copyFrom(cam);
      velocityDirty = true;
    }
    if (velocityDirty) {
      projectVelocity();
      velocityDirty = false;
    }
  }

  /** Resamples the field under the reference viewport into {@link #velTex}. */
  private void projectVelocity() {
    bindTarget(velTex, velWidth, velHeight);
    GLES20.glDisable(GLES20.GL_BLEND);
    GLES20.glUseProgram(projectProgram);
    bindTexture(GLES20.GL_TEXTURE0, fieldUTex, uFieldU, 0);
    bindTexture(GLES20.GL_TEXTURE1, fieldVTex, uFieldV, 1);
    GLES20.glUniform4f(uRange, fieldRange[0], fieldRange[2], fieldRange[1], fieldRange[3]);
    GLES20.glUniform4f(uGrid, fieldLon0, fieldLat0, fieldLatSpan, 0f);
    GLES20.glUniform2f(uHalfTexel, 0.5f / fieldWidth, 0.5f / fieldHeight);
    GLES20.glUniform3f(uCurve, CURVE_MUL, CURVE_MIN, CURVE_POW);
    GLES20.glUniform1f(uWorld, (float) ref.world);
    GLES20.glUniform2f(uCenter, (float) ref.cx, (float) ref.cy);
    GLES20.glUniform2f(uHalf, (float) (ref.width / 2), (float) (ref.height / 2));
    GLES20.glUniform2f(uRot, (float) ref.cos, (float) ref.sin);
    drawQuad();
  }

  /** One deposit: draw → fade → advect, the reference's order. */
  private void simulate(double dt) {
    // The reference's parameters, from the reference camera: frozen through a
    // gesture, re-derived when it settles.
    double z = windyZoom(ref.zoom);
    double amount =
        Math.min(
            AMOUNT_MAX,
            Math.round(
                ref.width * ref.height
                    / (AMOUNT_CONSTANT * Math.pow(AMOUNT_POW, z - AMOUNT_ZOOM))));
    double relativeAmount = amount * COUNT_MUL / (STATE_EDGE * STATE_EDGE);
    int perBlock =
        Math.max(
            1, Math.min(PARTICLES_PER_BLOCK, (int) Math.round(relativeAmount * PARTICLES_PER_BLOCK)));
    // Particles are laid out column-major within a block, so drawing the first
    // `perBlock` of each touches only these columns — and only these advect.
    int columns = Math.min(STATE_EDGE, (int) Math.floor(relativeAmount * STATE_EDGE + 1));
    float widthFactor = (float) Math.max(1, at(LINE_WIDTH, z) * PARTICLE_WIDTH * density);
    double timeScale = SPEED_PX * at(ZOOM_TO_SPEED, z) * density;

    // The block clock, in 60 Hz frames of real time. The block the cursor
    // leaves is the one reborn.
    blockFrame += dt * TARGET_FPS;
    int reseed = -1;
    if (blockFrame >= BLOCK_FRAMES) {
      blockFrame -= BLOCK_FRAMES;
      reseed = stateBlock;
      stateBlock = (stateBlock + 1) % BLOCKS;
    }

    // --- trails: clear, or carry into a moved reference frame
    if (needClear) {
      bindTarget(trailTex[trailFront], trailWidth, trailHeight);
      GLES20.glClearColor(0f, 0f, 0f, 0f);
      GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
      needClear = false;
    } else if (shiftPending) {
      bindTarget(trailTex[1 - trailFront], trailWidth, trailHeight);
      GLES20.glDisable(GLES20.GL_BLEND);
      GLES20.glUseProgram(shiftProgram);
      bindTexture(GLES20.GL_TEXTURE0, trailTex[trailFront], uShiftSrc, 0);
      GLES20.glUniform2f(uShiftShift, shiftX, shiftY);
      drawQuad();
      trailFront = 1 - trailFront;
      shiftState(shiftX, shiftY);
    }
    shiftPending = false;

    // --- segments: sixteen generations of last frame's motion
    bindTarget(trailTex[trailFront], trailWidth, trailHeight);
    GLES20.glEnable(GLES20.GL_BLEND);
    GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);
    GLES20.glUseProgram(segmentProgram);
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer);
    GLES20.glEnableVertexAttribArray(0);
    GLES20.glVertexAttribPointer(0, 4, GLES20.GL_UNSIGNED_BYTE, false, 4, 0);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    bindTexture(GLES20.GL_TEXTURE0, stateTex[stateFront], uState0, 0);
    bindTexture(GLES20.GL_TEXTURE1, stateTex[1 - stateFront], uState1, 1);
    // The quad is widthFactor + 1 physical pixels across, against the screen
    // rather than the trail texture, and feathered over its outer pixel.
    float aw = (widthFactor + 1f) / physicalWidth;
    float ah = (widthFactor + 1f) / physicalHeight;
    float len = Math.max(1f, widthFactor * 0.8f);
    GLES20.glUniform4f(uLine, aw * 2f / 255f, ah * 2f / 255f, -aw, -ah);
    GLES20.glUniform4f(uEdge, 0f, 0f, len * 2f / 255f, -len);
    GLES20.glUniform1f(uLen, len);
    // Storage order, with the fade curve walked backwards alongside: the block
    // reborn last lands at age ~0, each earlier one BLOCK_FRAMES older.
    int age = (int) blockFrame;
    for (int b = 0; b < BLOCKS; b++) {
      int block = (stateBlock - 1 - b + 2 * BLOCKS) % BLOCKS;
      GLES20.glUniform4f(uBlock, INV_STATE_EDGE, INV_STATE_EDGE, 0f, block / (float) BLOCKS);
      float a = ALPHA_LUT[age % LIFETIME];
      GLES20.glUniform4f(uColour, a, a, a, a);
      GLES20.glDrawElements(GLES20.GL_TRIANGLES, perBlock * 6, GLES20.GL_UNSIGNED_SHORT, 0);
      age += BLOCK_FRAMES;
    }
    GLES20.glDisableVertexAttribArray(0);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);

    // --- fade, in place: dst · FADE, by the blend unit alone
    GLES20.glBlendFunc(GLES20.GL_ZERO, GLES20.GL_CONSTANT_ALPHA);
    GLES20.glBlendColor(0f, 0f, 0f, (float) Math.pow(FADE, dt * TARGET_FPS));
    GLES20.glUseProgram(fadeProgram);
    drawQuad();
    GLES20.glDisable(GLES20.GL_BLEND);

    // --- advect into the back state texture
    bindTarget(stateTex[1 - stateFront], columns, STATE_EDGE);
    GLES20.glUseProgram(updateProgram);
    bindTexture(GLES20.GL_TEXTURE0, stateTex[stateFront], uUpdateState, 0);
    bindTexture(GLES20.GL_TEXTURE1, velTex, uUpdateVel, 1);
    // At most `o` of the viewport per frame, per axis, at full curve strength:
    // a stored byte b becomes (b − 128) / 127 · o, folded into one multiply-add.
    float ox = (float) (dt * timeScale / physicalWidth);
    float oy = (float) (dt * timeScale / physicalHeight);
    GLES20.glUniform4f(
        uUpdateDisp, ox * DISP_MUL, oy * DISP_MUL, -ox * DISP_ADD, -oy * DISP_ADD);
    drawQuad();
    stateFront = 1 - stateFront;
    if (reseed >= 0) {
      reseedBlock(reseed);
    }

    alpha = Math.min(1, alpha + dt * ALPHA_RATE);
    lastPerBlock = perBlock;
  }

  /** Moves the particles with the reference frame: into the back texture, then back. */
  private void shiftState(float dx, float dy) {
    GLES20.glUseProgram(stateProgram);
    bindTarget(stateTex[1 - stateFront], STATE_EDGE, STATE_EDGE);
    bindTexture(GLES20.GL_TEXTURE0, stateTex[stateFront], uStateState, 0);
    GLES20.glUniform2f(uStateShift, dx, dy);
    drawQuad();
    bindTarget(stateTex[stateFront], STATE_EDGE, STATE_EDGE);
    bindTexture(GLES20.GL_TEXTURE0, stateTex[1 - stateFront], uStateState, 0);
    GLES20.glUniform2f(uStateShift, 0f, 0f);
    drawQuad();
  }

  /** Draws the trails over the map, wherever the reference viewport now is. */
  private void composite() {
    if (!haveRef) {
      return;
    }
    double wrap = cameraWrap();
    refToClip(0, 0, wrap, affine, 0);
    refToClip(1, 0, wrap, affine, 2);
    refToClip(0, 1, wrap, affine, 4);
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, savedGl.framebuffer[0]);
    GLES20.glViewport(
        savedGl.viewport[0], savedGl.viewport[1], savedGl.viewport[2], savedGl.viewport[3]);
    GLES20.glEnable(GLES20.GL_BLEND);
    GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);
    GLES20.glUseProgram(compositeProgram);
    bindTexture(GLES20.GL_TEXTURE0, trailTex[trailFront], uTrail, 0);
    GLES20.glUniform4f(
        uAffine, affine[0], affine[1], affine[2] - affine[0], affine[3] - affine[1]);
    GLES20.glUniform2f(uAxisY, affine[4] - affine[0], affine[5] - affine[1]);
    GLES20.glUniform1f(uGain, (float) (COMPOSITE_GAIN * alpha));
    GLES20.glUniform1f(uLift, COMPOSITE_LIFT);
    drawQuad();
  }

  /**
   * The whole-world shift that puts the reference centre on the copy of the
   * world nearest this frame's centre, so a pan across the antimeridian does
   * not throw the field a world away.
   */
  private double cameraWrap() {
    double dx = ref.cx * (cam.world / ref.world) - cam.cx;
    return cam.world * Math.round(dx / cam.world);
  }

  /**
   * A point of the reference viewport (particle space) in this frame's clip
   * space: unrotate and unscale out of the reference, then into this camera.
   * The map has no pitch, so this is affine and three points place the quad.
   */
  private void refToClip(double px, double py, double wrap, float[] out, int o) {
    double dx = (px * 2 - 1) * ref.width / 2;
    double dy = (1 - py * 2) * ref.height / 2;
    double s = cam.world / ref.world;
    double ex = (ref.cx + dx * ref.cos + dy * ref.sin) * s - cam.cx - wrap;
    double ey = (ref.cy - dx * ref.sin + dy * ref.cos) * s - cam.cy;
    out[o] = (float) ((ex * cam.cos - ey * cam.sin) / (cam.width / 2));
    out[o + 1] = (float) (-(ex * cam.sin + ey * cam.cos) / (cam.height / 2));
  }

  private int lastPerBlock;

  private void report() {
    frame++;
    if ((frame & 63) == 0) {
      int err = GLES20.glGetError();
      if (err != GLES20.GL_NO_ERROR) {
        glErrors++;
        lastGlError = err;
      }
    }
    if (frame == 1 || frame % 600 == 0) {
      Log.i(
          TAG,
          String.format(
              java.util.Locale.ROOT,
              "frames=%d zoom=%.3f logical=%.0fx%.0f density=%.3f trail=%dx%d vel=%dx%d"
                  + " perBlock=%d total=%d deposit=%d skip=%d settle=%d glErr=%d(0x%x)",
              frame,
              cam.zoom,
              cam.width,
              cam.height,
              density,
              trailWidth,
              trailHeight,
              velWidth,
              velHeight,
              lastPerBlock,
              lastPerBlock * BLOCKS,
              deposits,
              skips,
              settles,
              glErrors,
              lastGlError));
      deposits = 0;
      skips = 0;
      settles = 0;
      glErrors = 0;
    }
  }

  // ------------------------------------------------------------- GL plumbing

  /** True when the EGL context was replaced since the objects were made. */
  private boolean contextLost() {
    EGLContext current = EGL14.eglGetCurrentContext();
    if (glContext == null || fadeProgram == 0) {
      glContext = current;
      return false;
    }
    return !glContext.equals(current);
  }

  private boolean createGlObjects() {
    fadeProgram = link(VERT_FULL, FRAG_ZERO, "a_pos");
    shiftProgram = link(VERT_FULL, FRAG_SHIFT, "a_pos");
    stateProgram = link(VERT_FULL, FRAG_STATE_SHIFT, "a_pos");
    updateProgram = link(VERT_FULL, FRAG_UPDATE, "a_pos");
    projectProgram = link(VERT_FULL, FRAG_PROJECT, "a_pos");
    segmentProgram = link(VERT_SEGMENT, FRAG_SEGMENT, "a_vert");
    compositeProgram = link(VERT_COMPOSITE, FRAG_COMPOSITE, "a_pos");
    if (fadeProgram == 0
        || shiftProgram == 0
        || stateProgram == 0
        || updateProgram == 0
        || projectProgram == 0
        || segmentProgram == 0
        || compositeProgram == 0) {
      Log.e(TAG, "shader programs failed to build; the compile log is above");
      deleteGlObjects();
      return false;
    }
    resolveUniforms();

    int[] buffers = new int[3];
    GLES20.glGenBuffers(3, buffers, 0);
    quadBuffer = buffers[0];
    vertexBuffer = buffers[1];
    indexBuffer = buffers[2];

    ByteBuffer quad = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder());
    quad.asFloatBuffer().put(new float[] {0, 0, 1, 0, 0, 1, 1, 1});
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadBuffer);
    GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, 8 * 4, quad, GLES20.GL_STATIC_DRAW);

    // One block's quads, reused for all sixteen: (texel x, row, side, end).
    // Column-major, the reference's order, so a partial draw is whole columns.
    ByteBuffer verts =
        ByteBuffer.allocateDirect(PARTICLES_PER_BLOCK * 16).order(ByteOrder.nativeOrder());
    ByteBuffer inds =
        ByteBuffer.allocateDirect(PARTICLES_PER_BLOCK * 12).order(ByteOrder.nativeOrder());
    final byte[][] corners = {{0, 0}, {(byte) 255, 0}, {(byte) 255, (byte) 255}, {0, (byte) 255}};
    for (int i = 0; i < PARTICLES_PER_BLOCK; i++) {
      byte x = (byte) (i / BLOCK_ROWS);
      byte y = (byte) (i % BLOCK_ROWS);
      for (byte[] c : corners) {
        verts.put(x).put(y).put(c[0]).put(c[1]);
      }
      short base = (short) (i * 4);
      inds.putShort(base)
          .putShort((short) (base + 1))
          .putShort((short) (base + 2))
          .putShort(base)
          .putShort((short) (base + 2))
          .putShort((short) (base + 3));
    }
    verts.position(0);
    inds.position(0);
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer);
    GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, verts.capacity(), verts, GLES20.GL_STATIC_DRAW);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    GLES20.glBufferData(
        GLES20.GL_ELEMENT_ARRAY_BUFFER, inds.capacity(), inds, GLES20.GL_STATIC_DRAW);
    GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);

    int[] fb = new int[1];
    GLES20.glGenFramebuffers(1, fb, 0);
    framebuffer = fb[0];

    // Seeded with noise: an unseeded pool starts as a grid.
    byte[] seed = new byte[STATE_EDGE * STATE_EDGE * 4];
    random.nextBytes(seed);
    ByteBuffer seedBuffer = ByteBuffer.allocateDirect(seed.length).order(ByteOrder.nativeOrder());
    seedBuffer.put(seed).position(0);
    for (int i = 0; i < 2; i++) {
      stateTex[i] = createTexture(GLES20.GL_NEAREST, GLES20.GL_CLAMP_TO_EDGE);
      GLES20.glTexImage2D(
          GLES20.GL_TEXTURE_2D,
          0,
          GLES20.GL_RGBA,
          STATE_EDGE,
          STATE_EDGE,
          0,
          GLES20.GL_RGBA,
          GLES20.GL_UNSIGNED_BYTE,
          seedBuffer);
    }
    glContext = EGL14.eglGetCurrentContext();
    haveRef = false;
    return true;
  }

  private void resolveUniforms() {
    uShiftSrc = GLES20.glGetUniformLocation(shiftProgram, "u_src");
    uShiftShift = GLES20.glGetUniformLocation(shiftProgram, "u_shift");
    uStateState = GLES20.glGetUniformLocation(stateProgram, "u_state");
    uStateShift = GLES20.glGetUniformLocation(stateProgram, "u_shift");
    uUpdateState = GLES20.glGetUniformLocation(updateProgram, "u_state");
    uUpdateVel = GLES20.glGetUniformLocation(updateProgram, "u_vel");
    uUpdateDisp = GLES20.glGetUniformLocation(updateProgram, "u_disp");
    uFieldU = GLES20.glGetUniformLocation(projectProgram, "u_field_u");
    uFieldV = GLES20.glGetUniformLocation(projectProgram, "u_field_v");
    uRange = GLES20.glGetUniformLocation(projectProgram, "u_range");
    uGrid = GLES20.glGetUniformLocation(projectProgram, "u_grid");
    uHalfTexel = GLES20.glGetUniformLocation(projectProgram, "u_half_texel");
    uCurve = GLES20.glGetUniformLocation(projectProgram, "u_curve");
    uWorld = GLES20.glGetUniformLocation(projectProgram, "u_world");
    uCenter = GLES20.glGetUniformLocation(projectProgram, "u_center");
    uHalf = GLES20.glGetUniformLocation(projectProgram, "u_half");
    uRot = GLES20.glGetUniformLocation(projectProgram, "u_rot");
    uState0 = GLES20.glGetUniformLocation(segmentProgram, "u_state0");
    uState1 = GLES20.glGetUniformLocation(segmentProgram, "u_state1");
    uBlock = GLES20.glGetUniformLocation(segmentProgram, "u_block");
    uLine = GLES20.glGetUniformLocation(segmentProgram, "u_line");
    uEdge = GLES20.glGetUniformLocation(segmentProgram, "u_edge");
    uColour = GLES20.glGetUniformLocation(segmentProgram, "u_colour");
    uLen = GLES20.glGetUniformLocation(segmentProgram, "u_len");
    uAffine = GLES20.glGetUniformLocation(compositeProgram, "u_affine");
    uAxisY = GLES20.glGetUniformLocation(compositeProgram, "u_axis_y");
    uTrail = GLES20.glGetUniformLocation(compositeProgram, "u_trail");
    uGain = GLES20.glGetUniformLocation(compositeProgram, "u_gain");
    uLift = GLES20.glGetUniformLocation(compositeProgram, "u_lift");
  }

  /**
   * The trail and velocity textures follow the viewport. A new size is a new
   * reference frame: the old trails no longer map onto anything.
   */
  private void ensureTargets() {
    int pw = (int) Math.round(cam.width * density);
    int ph = (int) Math.round(cam.height * density);
    if (pw == physicalWidth && ph == physicalHeight && trailTex[0] != 0) {
      return;
    }
    physicalWidth = pw;
    physicalHeight = ph;
    double scale = density > TRAIL_SCALE_DPR ? TRAIL_SCALE : 1;
    trailWidth = Math.max(1, Math.min(TRAIL_MAX, (int) Math.round(pw * scale)));
    trailHeight = Math.max(1, Math.min(TRAIL_MAX, (int) Math.round(ph * scale)));
    // The reference's wind texture is its tile mosaic, one texel per CSS pixel.
    velWidth = Math.max(1, Math.min(TRAIL_MAX, (int) Math.round(cam.width)));
    velHeight = Math.max(1, Math.min(TRAIL_MAX, (int) Math.round(cam.height)));
    for (int i = 0; i < 2; i++) {
      trailTex[i] = reallocate(trailTex[i], trailWidth, trailHeight);
    }
    velTex = reallocate(velTex, velWidth, velHeight);
    haveRef = false;
  }

  /** Replaces {@code texture} with a LINEAR, clamped RGBA8 texture of this size. */
  private static int reallocate(int texture, int width, int height) {
    if (texture != 0) {
      GLES20.glDeleteTextures(1, new int[] {texture}, 0);
    }
    int t = createTexture(GLES20.GL_LINEAR, GLES20.GL_CLAMP_TO_EDGE);
    GLES20.glTexImage2D(
        GLES20.GL_TEXTURE_2D,
        0,
        GLES20.GL_RGBA,
        width,
        height,
        0,
        GLES20.GL_RGBA,
        GLES20.GL_UNSIGNED_BYTE,
        null);
    return t;
  }

  /**
   * Rebirth, the reference's way: one block's rows of both state textures
   * overwritten with uniform noise. Both, so the first segment of the new life
   * has zero length and is culled instead of joining it to the last one.
   */
  private void reseedBlock(int block) {
    random.nextBytes(reseedBytes);
    reseedBuffer.position(0);
    reseedBuffer.put(reseedBytes).position(0);
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
          reseedBuffer);
    }
  }

  /**
   * Uploads the two quantised planes as two R8 textures, straight out of the
   * WND1 body: no interleaving copy, half the memory of RGBA. REPEAT in
   * longitude because it is cyclic; CLAMP in latitude because it is not.
   */
  private void uploadField(FieldUpload upload) {
    int n = upload.width * upload.height;
    if (upload.bytes.length < upload.planeOffset + n * 2) {
      Log.e(TAG, "wind field truncated: need " + (upload.planeOffset + n * 2));
      return;
    }
    GLES20.glGetIntegerv(GLES20.GL_UNPACK_ALIGNMENT, scratch, 0);
    GLES20.glGetIntegerv(GLES30.GL_PIXEL_UNPACK_BUFFER_BINDING, scratch, 1);
    GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1);
    GLES20.glBindBuffer(GLES30.GL_PIXEL_UNPACK_BUFFER, 0);
    if (fieldUTex == 0) {
      fieldUTex = createTexture(GLES20.GL_LINEAR, GLES20.GL_REPEAT);
      fieldVTex = createTexture(GLES20.GL_LINEAR, GLES20.GL_REPEAT);
    }
    for (int i = 0; i < 2; i++) {
      GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, i == 0 ? fieldUTex : fieldVTex);
      GLES20.glTexParameteri(
          GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
      GLES20.glTexImage2D(
          GLES20.GL_TEXTURE_2D,
          0,
          GLES30.GL_R8,
          upload.width,
          upload.height,
          0,
          GLES30.GL_RED,
          GLES20.GL_UNSIGNED_BYTE,
          ByteBuffer.wrap(upload.bytes, upload.planeOffset + i * n, n));
    }
    GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, scratch[0]);
    GLES20.glBindBuffer(GLES30.GL_PIXEL_UNPACK_BUFFER, scratch[1]);
    fieldWidth = upload.width;
    fieldHeight = upload.height;
    fieldRange[0] = upload.uMin;
    fieldRange[1] = upload.uMax;
    fieldRange[2] = upload.vMin;
    fieldRange[3] = upload.vMax;
    fieldLon0 = upload.lon0;
    fieldLat0 = upload.lat0;
    fieldLatSpan = upload.dLat * upload.height;
    fieldReady = true;
    velocityDirty = true;
  }

  /** The context is gone with every name in it; start again on the next frame. */
  private void forgetGlObjects() {
    fadeProgram = 0;
    shiftProgram = 0;
    stateProgram = 0;
    updateProgram = 0;
    projectProgram = 0;
    segmentProgram = 0;
    compositeProgram = 0;
    quadBuffer = 0;
    vertexBuffer = 0;
    indexBuffer = 0;
    framebuffer = 0;
    stateTex[0] = stateTex[1] = 0;
    trailTex[0] = trailTex[1] = 0;
    velTex = 0;
    fieldUTex = 0;
    fieldVTex = 0;
    physicalWidth = 0;
    physicalHeight = 0;
    fieldReady = false;
    haveRef = false;
    glContext = null;
  }

  private void deleteGlObjects() {
    for (int p :
        new int[] {
          fadeProgram,
          shiftProgram,
          stateProgram,
          updateProgram,
          projectProgram,
          segmentProgram,
          compositeProgram
        }) {
      if (p != 0) {
        GLES20.glDeleteProgram(p);
      }
    }
    if (quadBuffer != 0) {
      GLES20.glDeleteBuffers(3, new int[] {quadBuffer, vertexBuffer, indexBuffer}, 0);
    }
    if (framebuffer != 0) {
      GLES20.glDeleteFramebuffers(1, new int[] {framebuffer}, 0);
    }
    int[] textures = {
      stateTex[0], stateTex[1], trailTex[0], trailTex[1], velTex, fieldUTex, fieldVTex
    };
    for (int t : textures) {
      if (t != 0) {
        GLES20.glDeleteTextures(1, new int[] {t}, 0);
      }
    }
    forgetGlObjects();
  }

  private void bindTarget(int texture, int width, int height) {
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
    GLES20.glFramebufferTexture2D(
        GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, texture, 0);
    GLES20.glViewport(0, 0, width, height);
  }

  /** The unit quad as a triangle strip on attribute 0, which every quad program binds. */
  private void drawQuad() {
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadBuffer);
    GLES20.glEnableVertexAttribArray(0);
    GLES20.glVertexAttribPointer(0, 2, GLES20.GL_FLOAT, false, 0, 0);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    GLES20.glDisableVertexAttribArray(0);
  }

  private static void bindTexture(int unit, int texture, int location, int index) {
    GLES20.glActiveTexture(unit);
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture);
    GLES20.glUniform1i(location, index);
  }

  private static int createTexture(int filter, int wrapS) {
    int[] id = new int[1];
    GLES20.glGenTextures(1, id, 0);
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id[0]);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, filter);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, filter);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, wrapS);
    GLES20.glTexParameteri(
        GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    return id[0];
  }

  private static double mercatorY(double lat) {
    double clamped = Math.max(-85.051129, Math.min(85.051129, lat));
    double phi = clamped * Math.PI / 180;
    return (1 - Math.log(Math.tan(Math.PI / 4 + phi / 2)) / Math.PI) / 2;
  }

  /** Links a program whose only attribute is bound to location 0. */
  private static int link(String vertexSource, String fragmentSource, String attribute) {
    int vertex = compile(GLES20.GL_VERTEX_SHADER, vertexSource);
    int fragment = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
    if (vertex == 0 || fragment == 0) {
      return 0;
    }
    int program = GLES20.glCreateProgram();
    GLES20.glAttachShader(program, vertex);
    GLES20.glAttachShader(program, fragment);
    GLES20.glBindAttribLocation(program, 0, attribute);
    GLES20.glLinkProgram(program);
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
   * Where the map was drawing when it called in.
   *
   * <p>Only these two: MapLibre resets its own state before calling a custom
   * layer and marks its whole state cache dirty afterwards
   * ({@code DrawableCustomLayerHostTweaker::execute}), so everything else this
   * layer touches is rebound by the map anyway. Reading back the rest — the
   * twenty-odd {@code glGet}s this used to make — was per-frame cost for
   * nothing.
   */
  private static final class GlState {
    final int[] framebuffer = new int[1];
    final int[] viewport = new int[4];

    void capture() {
      GLES20.glGetIntegerv(GLES20.GL_FRAMEBUFFER_BINDING, framebuffer, 0);
      GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, viewport, 0);
    }

    void restore() {
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer[0]);
      GLES20.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
      GLES20.glDisable(GLES20.GL_BLEND);
      GLES30.glBindVertexArray(0);
    }
  }
}
