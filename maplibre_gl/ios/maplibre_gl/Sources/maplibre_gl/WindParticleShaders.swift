import Foundation

/// The Metal half of the wind-particle field, as source compiled at run time.
///
/// Kept as a string and handed to `makeLibrary(source:options:)` rather than
/// added to the SwiftPM target as a `.metal` resource. Two reasons, and the
/// second is the one that decided it:
///
/// * The Android side keeps its GLSL in exactly the same shape — string
///   constants next to the code that binds them. Two platforms drawing the
///   same picture from shaders that must agree line for line are far easier to
///   keep in step when both are read the same way.
/// * A `.metal` file in a SwiftPM target compiles into a `default.metallib`
///   inside the package's resource bundle, and a Flutter plugin's bundle is
///   found through `Bundle.module` — which resolves differently depending on
///   whether the plugin was linked statically or as a framework. A path that
///   works in the example app and returns nil in the host app is the worst
///   kind of difference, because it shows up as a layer that silently never
///   draws.
///
/// The compile costs a few milliseconds, once, on the first frame.
///
/// ## Coordinates
///
/// One convention throughout, the same as the GLSL: *particle space* is the
/// reference viewport as 0..1 with y up — what a particle stores and what
/// `fullVertex` hands a fragment as `p`. Clip space is y up in both APIs, so
/// `p * 2 - 1` is the clip position here as there.
///
/// What differs is textures. GL's t = 0 is the first row, which a
/// framebuffer stores at the bottom, so on Android a screen-sized texture's
/// coordinate *is* particle space. Metal's texture origin is the top-left and
/// its framebuffer's first row is the top, so here the texture coordinate is
/// `(p.x, 1 − p.y)`. Every sample of a screen-sized texture goes through
/// `toTex`; the state texture is addressed by texel, not by position, and does
/// not.
enum WindParticleShaders {
  /// Fifteen `float4`s, matching `WindUniforms` in Swift field for field. The
  /// all-`float4` layout is not tidiness: Metal silently accepts a struct
  /// whose Swift twin has drifted and reads every value from the wrong offset.
  static let source = """
  #include <metal_stdlib>
  using namespace metal;

  constant float PI = 3.141592653589793;

  struct Uniforms {
    float4 disp;
    float4 block;
    float4 line;
    float4 edge;
    float4 colour;
    float4 seg;
    float4 range;
    float4 grid;
    float4 curve;
    float4 texel;
    float4 cam0;
    float4 cam1;
    float4 shift;
    float4 affine0;
    float4 affine1;
  };

  struct FullOut {
    float4 position [[position]];
    float2 p;
  };

  static float2 toTex(float2 p) { return float2(p.x, 1.0 - p.y); }

  vertex FullOut fullVertex(uint vid [[vertex_id]]) {
    float2 corners[4] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1)};
    FullOut out;
    out.p = corners[vid];
    out.position = float4(out.p * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  // ------------------------------------------------------------------ state

  // High byte in `ba`, low byte in `rg`: 16 bits per axis. All-zero is dead.
  static bool isDead(float4 t) { return t.r + t.g + t.b + t.a < 0.025; }
  static float2 decodePos(float4 t) { return t.ba + t.rg / 255.5; }
  static float4 encodePos(float2 pos) {
    float2 lo = fract(pos * 255.0 + 0.25 / 255.0);
    return float4(lo, pos - lo / 255.0);
  }

  // The state is addressed by texel, so the fragment's own pixel is its
  // particle — whatever the viewport covers.
  static float4 readState(texture2d<float> s, float4 frag) {
    return s.read(uint2(frag.xy));
  }

  // The reference's advection; see FRAG_UPDATE in the Android twin.
  fragment float4 updateFragment(FullOut in [[stage_in]],
                                 constant Uniforms &u [[buffer(0)]],
                                 texture2d<float> state [[texture(0)]],
                                 texture2d<float> vel [[texture(1)]],
                                 sampler lin [[sampler(1)]]) {
    float4 t = readState(state, in.position);
    if (isDead(t)) { return float4(0.0); }
    float2 pos = decodePos(t);
    float4 w = vel.sample(lin, toTex(fract(pos)));
    if (w.a < 0.5) { return t; }
    float2 dpos = w.rg * u.disp.xy + u.disp.zw;
    if (max(abs(dpos.x), abs(dpos.y)) < 0.000025) { return float4(0.0); }
    return encodePos(fract(pos + dpos));
  }

  // Moves every live particle by `shift` — the reference frame moved under it.
  fragment float4 stateShiftFragment(FullOut in [[stage_in]],
                                     constant Uniforms &u [[buffer(0)]],
                                     texture2d<float> state [[texture(0)]]) {
    float4 t = readState(state, in.position);
    return isDead(t) ? float4(0.0) : encodePos(fract(decodePos(t) + u.shift.xy));
  }

  // -------------------------------------------------------------- velocity

  static float mercLat(float y) {
    return (2.0 * atan(exp(PI * (1.0 - 2.0 * y))) - PI / 2.0) * 180.0 / PI;
  }

  // The global grid resampled under every pixel of the reference viewport,
  // rotated into screen axes and bent by the speed curve; see FRAG_PROJECT in
  // the Android twin for the encoding and the half-texel term.
  fragment float4 projectFragment(FullOut in [[stage_in]],
                                  constant Uniforms &u [[buffer(0)]],
                                  texture2d<float> fieldU [[texture(0)]],
                                  texture2d<float> fieldV [[texture(1)]],
                                  sampler smp [[sampler(0)]]) {
    float2 d = float2(in.p.x * 2.0 - 1.0, 1.0 - in.p.y * 2.0) * u.cam0.zw;
    float wx = u.cam0.x + d.x * u.cam1.x + d.y * u.cam1.y;
    float wy = u.cam0.y - d.x * u.cam1.y + d.y * u.cam1.x;
    float lon = wx / u.cam1.z * 360.0 - 180.0;
    float lat = mercLat(wy / u.cam1.z);
    float2 uv = float2(fract((lon - u.grid.x) / 360.0), (lat - u.grid.y) / u.grid.z)
        + u.texel.xy;
    if (uv.y < 0.0 || uv.y > 1.0) { return float4(128.0 / 255.0, 128.0 / 255.0, 0.0, 0.0); }
    float2 raw = float2(fieldU.sample(smp, uv).r, fieldV.sample(smp, uv).r);
    float2 v = mix(u.range.xy, u.range.zw, raw);
    v = float2(v.x * u.cam1.x + v.y * u.cam1.y, -v.x * u.cam1.y + v.y * u.cam1.x);
    float2 o = v * u.curve.x;
    float d2 = dot(o, o);
    if (d2 < 1e-8) { return float4(128.0 / 255.0, 128.0 / 255.0, 0.0, 1.0); }
    float mag = min(max(pow(d2, u.curve.z), u.curve.y), 1.0);
    float2 dir = o * (mag * rsqrt(d2));
    return float4((128.0 + 127.0 * dir) / 255.0, 0.0, 1.0);
  }

  // ---------------------------------------------------------------- trails

  // Last deposit's trails, multiplied by the fade — and moved into a panned
  // reference frame by whole texels. What scrolled in has no history.
  fragment float4 fadeFragment(FullOut in [[stage_in]],
                               constant Uniforms &u [[buffer(0)]],
                               texture2d<float> src [[texture(0)]],
                               sampler smp [[sampler(0)]]) {
    float2 p = in.p - u.shift.xy;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) { return float4(0.0); }
    return src.sample(smp, toTex(p)) * u.shift.z;
  }

  struct SegmentOut {
    float4 position [[position]];
    float discardFlag;
    float edge;
  };

  // One particle as a quad from where it was to where it is; see VERT_SEGMENT.
  vertex SegmentOut segmentVertex(uint vid [[vertex_id]],
                                  constant uchar4 *verts [[buffer(1)]],
                                  constant Uniforms &u [[buffer(0)]],
                                  texture2d<float> state0 [[texture(0)]],
                                  texture2d<float> state1 [[texture(1)]],
                                  sampler smp [[sampler(0)]]) {
    uchar4 a = verts[vid];
    float2 tc = (float2(a.xy) + 0.5) * u.block.xy + u.block.zw;
    float4 t0 = state0.sample(smp, tc, level(0));
    float4 t1 = state1.sample(smp, tc, level(0));

    SegmentOut out;
    out.discardFlag = step(0.025, t0.r + t0.g + t0.b + t0.a);
    float2 posA = fract(t0.ba + t0.rg / 255.5) * 2.0 - 1.0;
    float2 posB = fract(t1.ba + t1.rg / 255.5) * 2.0 - 1.0;
    float2 dirF = posA - posB;
    float d = length(dirF);
    float2 dirFN = dirF / max(d, 1e-20);
    float2 pos = mix(posB, posA, float(a.w) * 0.003921569);
    pos += float2(dirFN.y, -dirFN.x) * (float2(float(a.z)) * u.line.xy + u.line.zw);
    if (d > 0.5 || d < 1e-5) { pos.x += 10.0; }
    out.position = float4(pos, 0.0, 1.0);
    out.edge = u.edge.z * float(a.z) + u.edge.w;
    return out;
  }

  // White, premultiplied, feathered across the width. The pipeline blends it
  // as `src · fade + dst · (1 − src.a)` over trails already multiplied by the
  // fade — the reference's draw-then-fade, in one pass.
  fragment float4 segmentFragment(SegmentOut in [[stage_in]],
                                  constant Uniforms &u [[buffer(0)]]) {
    if (in.discardFlag <= 0.0) { discard_fragment(); }
    return u.colour * clamp(u.seg.x - abs(in.edge), 0.0, 1.0);
  }

  // ------------------------------------------------------------- composite

  struct CompositeOut {
    float4 position [[position]];
    float2 p;
  };

  // The trail quad, placed where its reference viewport is in this frame.
  vertex CompositeOut compositeVertex(uint vid [[vertex_id]],
                                      constant Uniforms &u [[buffer(0)]]) {
    float2 corners[4] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1)};
    float2 a = corners[vid];
    CompositeOut out;
    out.p = a;
    out.position = float4(u.affine0.xy + a.x * u.affine0.zw + a.y * u.affine1.xy, 0.0, 1.0);
    return out;
  }

  fragment float4 compositeFragment(CompositeOut in [[stage_in]],
                                    constant Uniforms &u [[buffer(0)]],
                                    texture2d<float> trail [[texture(0)]],
                                    sampler smp [[sampler(0)]]) {
    return max(trail.sample(smp, toTex(in.p)) * u.affine1.z + u.affine1.w, 0.0);
  }
  """
}
