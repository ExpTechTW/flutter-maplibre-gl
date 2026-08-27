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
/// ## What these shaders are
///
/// A transcription of Windy's `gl-particles` renderer, the same one
/// `WindParticleLayer.java` carries. The passes, their names and their
/// arithmetic are deliberately the same on both platforms; where the two
/// differ it is only because Metal's texture and clip conventions differ from
/// GL's, and each such place says so.
///
/// The one pass the reference does not have is `windProjectFragment`, which
/// redraws DPIP's global ECMWF grid as a viewport-aligned field. Windy stitches
/// its wind texture from the map tiles under the viewport, so a particle's
/// stored position already *is* its screen position. Doing the projection once
/// per frame into a texture buys the same property, and every constant the
/// reference calibrated in screen space then transfers unchanged.
enum WindParticleShaders {
  /// Uniforms shared by every pass.
  ///
  /// Eleven `float4`s and nothing else, matching `WindUniforms` in Swift field for
  /// field. The all-`float4` layout is not tidiness: Metal silently accepts a
  /// struct whose Swift twin has drifted and reads every value from the wrong
  /// offset, and mixed `float2`/`float3`/`float` members are where that drift
  /// starts. With one member size there is no padding to disagree about.
  static let source = """
  #include <metal_stdlib>
  using namespace metal;

  constant float PI = 3.141592653589793;

  struct Uniforms {
    float4 disp;    // update: (2ox, 2oy, -ox, -oy)
    float4 block;   // draw: (1/edge, 1/edge, 0, blockIndex/blocks)
    float4 lineW;   // draw: (aw*2/255, ah*2/255, -aw, -ah)
    float4 edge;    // draw: (0, 0, halfLen*2/255, -halfLen)
    float4 colour;  // draw: this generation's alpha, in all four channels
    float4 curve;   // project: (speedMul, speedMin, cPow, -)
    float4 cam0;    // (centreX, centreY, halfWidth, halfHeight)
    float4 cam1;    // (cos(-bearing), sin(-bearing), worldPx, lon0)
    float4 fld;     // (lat0, latSpan, halfLen, fadeOrCompositeGain)
    float4 misc;    // (shiftX, shiftY, lift, -)
    float4 range;   // (minU, minV, maxU, maxV) — the field's quantisation range
  };

  struct QuadOut {
    float4 position [[position]];
    float2 tex;
  };

  static float mercLat(float y) {
    return (2.0 * atan(exp(PI * (1.0 - 2.0 * y))) - PI / 2.0) * 180.0 / PI;
  }

  // ------------------------------------------------------------ full-screen quad

  vertex QuadOut quadVertex(uint vid [[vertex_id]]) {
    // The same 0..1 space and the same double flip the GL side uses, so the
    // fade pass samples the texel it is about to overwrite.
    float2 corners[4] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1)};
    float2 a = corners[vid];
    QuadOut out;
    out.tex = a;
    out.position = float4(1.0 - 2.0 * a, 0.0, 1.0);
    return out;
  }

  // The screen quad intentionally flips x. The update and projection passes
  // cannot inherit that mapping: both must write the texel they read, and a
  // state update that stores a particle in the opposite texel makes the draw
  // pass join it to an unrelated previous position. Metal's framebuffer and
  // texture coordinates both point down in y, so only x needs cancelling —
  // which is the one place this file and the GLSL differ, since GL's point up.
  vertex QuadOut updateVertex(uint vid [[vertex_id]]) {
    float2 corners[4] = {float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1)};
    float2 a = corners[vid];
    QuadOut out;
    out.tex = float2(1.0 - a.x, a.y);
    out.position = float4(1.0 - 2.0 * a, 0.0, 1.0);
    return out;
  }

  // ------------------------------------------------------- projection pass

  // DPIP's addition. Samples the global grid at the lon/lat under each screen
  // pixel and writes the wind there, encoded the reference's way: normalise by
  // glMaxSpeedParam, bend by pow(|v|², cPow), floor at glMinSpeedParam, then
  // store direction * 128 + 128 so 0.5 is a zero vector.
  //
  // Two values carry meaning downstream. All-zero means "no data" and freezes
  // a particle where it stands; exactly 0.5 means "no wind" and lets it die.
  fragment float4 windProjectFragment(QuadOut in [[stage_in]],
                                      constant Uniforms &u [[buffer(0)]],
                                      texture2d<float> field [[texture(0)]],
                                      sampler smp [[sampler(0)]]) {
    float2 vp = u.cam0.zw;
    // Particle space has y up — it becomes clip space directly — and screen
    // pixels have y down. Identical to the GLSL: `in.tex` is the texel being
    // written on both platforms, because both run under `updateVertex`.
    float2 sp = float2(in.tex.x * 2.0 * vp.x, (1.0 - in.tex.y) * 2.0 * vp.y);
    float2 d = sp - vp;
    // Unrotate, then unproject: the exact inverse of the map's projection.
    float dx = d.x * u.cam1.x + d.y * u.cam1.y;
    float dy = -d.x * u.cam1.y + d.y * u.cam1.x;
    float lon = (u.cam0.x + dx) / u.cam1.z * 360.0 - 180.0;
    float lat = mercLat((u.cam0.y + dy) / u.cam1.z);
    float2 uv = float2(fract((lon - u.cam1.w) / 360.0), (lat - u.fld.x) / u.fld.y);
    if (uv.y < 0.0 || uv.y > 1.0) { return float4(0.0); }

    float2 v = mix(u.range.xy, u.range.zw, field.sample(smp, uv).rg);
    // Into the viewport's frame. The position above was unrotated to find the
    // grid cell; the vector it holds is still east/north and has to be turned
    // the same way, or a rotated map keeps north-up arrows. The v flip is the
    // y-up/y-down change, folded into the rotation.
    v = float2(v.x * u.cam1.x + v.y * u.cam1.y,
               -v.x * u.cam1.y + v.y * u.cam1.x);
    float2 orient = v * u.curve.x;
    float d2 = dot(orient, orient);
    // Exactly neutral: no wind here, so let the particle die.
    if (d2 < 1e-8) { return float4(0.5, 0.5, 0.0, 1.0); }
    // The speed curve: compresses the middle of the range and lifts the bottom,
    // so a light breeze still moves visibly.
    float scale = (128.0 * max(pow(d2, u.curve.z), u.curve.y)) / sqrt(d2);
    orient = orient * scale + float2(128.0);
    return float4(orient / 255.0, 0.0, 1.0);
  }

  // ------------------------------------------------------------------ fade pass

  fragment float4 screenFragment(QuadOut in [[stage_in]],
                                 constant Uniforms &u [[buffer(0)]],
                                 texture2d<float> screen [[texture(0)]],
                                 sampler smp [[sampler(0)]]) {
    float2 uv = 1.0 - in.tex + u.misc.xy;
    // Whatever scrolled in from outside the old frame has no history, and
    // sampling it clamped would smear the edge pixel across the strip the pan
    // uncovered.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      return float4(0.0);
    }
    float4 c = screen.sample(smp, uv);
    // No `floor` here, and that is the whole tail. The reference fades with a
    // hardware blend — a pure multiply — and lets the 8-bit write round.
    // Truncating instead loses a whole LSB per frame on top of the fade, which
    // barely touches a bright stroke and destroys a dim one: a trail at 5/255
    // decays by 0.8 a frame, not 0.97. The tail dies within a few frames and
    // every particle reads as a bright blunt head with nothing behind it.
    return max(float4(0.0), c * u.fld.w + u.misc.z);
  }

  // ----------------------------------------------------------------- update pass

  constant float treshold = 0.025;
  constant float prec = 1000.0;
  constant float eps = 0.001;

  static bool hasMovement(float2 dp) {
    float r = step(treshold, abs(dp.r * prec));
    float g = step(treshold, abs(dp.g * prec));
    return (r + g) > 0.0;
  }

  // The reference's simulation, transcribed. No projection, no respawn, no
  // density weighting, no view test: a particle is a point in screen space, the
  // wind texture is aligned to that space, and the only way to die is to stop
  // moving. Rebirth is the CPU's job.
  //
  // Position is split across all four channels — `ba` the high byte, `rg` the
  // low — because one byte per axis is 256 distinct places on the whole screen,
  // and particles snap between them instead of flowing.
  fragment float4 updateFragment(QuadOut in [[stage_in]],
                                 constant Uniforms &u [[buffer(0)]],
                                 texture2d<float> state [[texture(0)]],
                                 texture2d<float> wind [[texture(1)]],
                                 sampler smp [[sampler(0)]],
                                 sampler wsmp [[sampler(1)]]) {
    float4 tex0 = state.sample(smp, in.tex);
    float2 pos = tex0.ba + tex0.rg / 255.5;
    // Nearest for the state — a texel is a particle, never a blend of two —
    // and linear for the field, which the GL side also filters.
    float2 dpos = wind.sample(wsmp, fract(pos)).rg;
    // All zero: no data under this pixel. Hold position rather than drift.
    if (dpos.x + dpos.y < eps) { return tex0; }
    dpos = dpos * u.disp.xy + u.disp.zw;
    // Below the movement threshold is how a particle dies. The draw pass
    // discards an all-zero texel; the block cycle brings it back.
    if (!hasMovement(dpos)) { return float4(0.0); }
    pos = fract(pos + dpos);
    float4 out;
    out.rg = fract(pos * 255.0 + 0.25 / 255.0);
    out.ba = pos - out.rg / 255.0;
    return out;
  }

  // ------------------------------------------------------------------ draw pass

  struct SegmentOut {
    float4 position [[position]];
    float discardFlag;
    float edge;
  };

  // One particle as a short line from where it was to where it is.
  //
  // Note what is absent: any projection at all. `pos * 2 - 1` is clip space,
  // because the stored position is already the position on screen.
  vertex SegmentOut segmentVertex(uint vid [[vertex_id]],
                                  constant uchar4 *verts [[buffer(1)]],
                                  constant Uniforms &u [[buffer(0)]],
                                  texture2d<float> state0 [[texture(0)]],
                                  texture2d<float> state1 [[texture(1)]],
                                  sampler smp [[sampler(0)]]) {
    uchar4 a = verts[vid];
    // Texel centres. The reference samples the boundary; with nearest filtering
    // that is a rounding coin-flip which can pair a particle's current position
    // with its neighbour's previous one — a stroke across the whole map.
    float2 tc = (float2(a.xy) + 0.5) * u.block.xy + u.block.zw;
    float4 t0 = state0.sample(smp, tc);
    float4 t1 = state1.sample(smp, tc);

    SegmentOut out;
    // An all-zero texel is the update pass's way of saying "dead".
    out.discardFlag = step(0.025, t0.r + t0.g + t0.b + t0.a);

    float2 posA = fract(t0.ba + t0.rg / 255.5) * 2.0 - 1.0;
    float2 posB = fract(t1.ba + t1.rg / 255.5) * 2.0 - 1.0;
    float2 dirF = posA - posB;
    float2 dirFN = normalize(dirF);
    float d = length(dirF);
    float2 dirRN = float2(dirFN.y, -dirFN.x);
    float2 pos = mix(posB, posA, float(a.w) * 0.003921569);
    pos += dirRN * (float2(float(a.z)) * u.lineW.xy + u.lineW.zw);
    // Two ways a segment carries no information, and both must go.
    //
    // `d > 0.5`: wrapped across an edge this frame, so the quad would span the
    // screen. `d < 1e-5`: the two ends are the same point, which happens on the
    // frame after a rebirth when both state textures hold the identical fresh
    // random position — `normalize(0)` is NaN and a NaN clip coordinate
    // rasterises to whatever the driver feels like. The reference carries this
    // same guard, but only under its WAVES define.
    if (d > 0.5 || d < 1e-5) { pos.x += 10.0; }

    out.position = float4(pos, 0.0, 1.0);
    out.edge = u.edge.z * float(a.z) + u.edge.w;
    return out;
  }

  // Grey, feathered across the width. The reference has no speed term: wind
  // speed changes a segment's length, not its brightness.
  fragment float4 segmentFragment(SegmentOut in [[stage_in]],
                                  constant Uniforms &u [[buffer(0)]]) {
    if (in.discardFlag <= 0.0) { discard_fragment(); }
    float aa = clamp(u.fld.z - abs(in.edge), 0.0, 1.0);
    // Premultiplied: the generations overlap constantly, and straight alpha
    // would let the newest erase the older ones instead of adding to them.
    return u.colour * float4(aa);
  }
  """
}
