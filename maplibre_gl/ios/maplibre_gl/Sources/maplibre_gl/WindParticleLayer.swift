import Foundation
import Metal
import QuartzCore
import MapLibre
import simd

/// Uniforms shared by every pass.
///
/// **Hand-packed so there is no implicit padding anywhere**, and the Metal
/// declaration in `WindParticleShaders` is field-for-field identical. Metal and
/// Swift agree on the alignment of `float2` (8) and `float4` (16), but they are
/// free to insert padding where a declaration order forces it — and a struct
/// that drifts by four bytes does not fail to compile or bind. It reads the
/// next field along, which shows up as a field blowing at the wrong speed, or
/// as nothing drawing at all, with no error anywhere.
///
/// The layout: eleven `float4`s and nothing else, in the order
/// `WindParticleShaders` declares them. One member size means there is no
/// padding for the two declarations to disagree about — which is the only kind
/// of drift Metal will not report.
struct WindUniforms {
  var disp = SIMD4<Float>()    // update: (2ox, 2oy, -ox, -oy)
  var block = SIMD4<Float>()   // draw: (1/edge, 1/edge, 0, blockIndex/blocks)
  var lineW = SIMD4<Float>()   // draw: (aw*2/255, ah*2/255, -aw, -ah)
  var edge = SIMD4<Float>()    // draw: (0, 0, halfLen*2/255, -halfLen)
  var colour = SIMD4<Float>()  // draw: this generation's alpha, four channels
  var curve = SIMD4<Float>()   // project: (speedMul, speedMin, cPow, -)
  var cam0 = SIMD4<Float>()    // (centreX, centreY, halfWidth, halfHeight)
  var cam1 = SIMD4<Float>()    // (cos(-bearing), sin(-bearing), worldPx, lon0)
  var fld = SIMD4<Float>()     // (lat0, latSpan, halfLen, fade or gain)
  var misc = SIMD4<Float>()    // (shiftX, shiftY, lift, -)
  var range = SIMD4<Float>()   // (minU, minV, maxU, maxV)
}

/// One WND1 payload waiting to be uploaded on the render thread.
struct WindFieldUpload {
  let bytes: Data
  let planeOffset: Int
  let width: Int
  let height: Int
  let lat0: Float
  let lon0: Float
  let dLat: Float
  let dLon: Float
  let uMin: Float
  let uMax: Float
  let vMin: Float
  let vMax: Float
}

/// The zoom-dependent curves, as their endpoints — the same shape the Android
/// layer takes, and interpolated by the same rules, so the numbers keep one
/// home in `wind_particle_sim.dart`.
struct WindTuning {
  var zoomLo: Double = 3
  var zoomHi: Double = 7
  var particlesLo: Double = 6400
  var particlesHi: Double = 1024
  var lineWidthZ3: Double = 1
  var lineWidthZ4: Double = 1.2
  var lineWidthZ5: Double = 1.6
  var lineWidthZ6: Double = 1.8
  var lineWidthZ7: Double = 2
  var particleWidth: Double = 1.3
  var speedFactorLo: Double = 0.2
  var speedFactorHi: Double = 0.0151
  var fadeOpacityLo: Double = 0.97
  var fadeOpacityHi: Double = 0.97
  var densityCalm: Double = 0.5
  var densityStrong: Double = 5.5
  var speedScale: Double = 32
  var pixelRatio: Double = 1

  private func fraction(_ zoom: Double) -> Double {
    let f = (zoom - zoomLo) / (zoomHi - zoomLo)
    return f < 0 ? 0 : (f > 1 ? 1 : f)
  }

  private func lin(_ lo: Double, _ hi: Double, _ zoom: Double) -> Double {
    let f = fraction(zoom)
    return lo * (1 - f) + hi * f
  }

  private func log_(_ lo: Double, _ hi: Double, _ zoom: Double) -> Double {
    let f = fraction(zoom)
    return exp(log(lo) * (1 - f) + log(hi) * f)
  }

  func particleCount(_ zoom: Double) -> Int {
    let edge = max(1, Int(( log_(particlesLo, particlesHi, zoom)).squareRoot().rounded()))
    return edge * edge
  }

  func lineWidth(_ zoom: Double) -> Float {
    let z = max(zoomLo, min(zoomHi, zoom))
    let lo = Int(floor(z))
    let hi = Int(ceil(z))
    let a = lineWidthAt(lo)
    if lo == hi { return Float(a) }
    let b = lineWidthAt(hi)
    return Float(a * (1 - (z - Double(lo))) + b * (z - Double(lo)))
  }

  private func lineWidthAt(_ zoom: Int) -> Double {
    switch zoom {
    case 3: return lineWidthZ3
    case 4: return lineWidthZ4
    case 5: return lineWidthZ5
    case 6: return lineWidthZ6
    default: return lineWidthZ7
    }
  }

  func speedFactor(_ zoom: Double) -> Float {
    Float(0.0001 * log_(speedFactorLo, speedFactorHi, zoom))
  }

  func fadeOpacity(_ zoom: Double) -> Float {
    Float(lin(fadeOpacityLo, fadeOpacityHi, zoom))
  }
}

/// GPU wind-particle field drawn inside the map's own Metal pass.
///
/// The Android twin of this file explains at length why the particles are not
/// a Flutter widget: HCPP leaks a full-screen buffer per presented Flutter
/// frame, and an animated overlay took a Pixel 9 from 394 MB to 8 GB of GPU
/// memory in sixteen seconds. iOS never had that leak — but the field is map
/// content, and drawing it here means it pans, zooms and rotates in the same
/// pass as the map rather than chasing it a frame behind. One design, two
/// platforms, one set of numbers.
///
/// ## What Metal makes different
///
/// `MLNCustomStyleLayer` hands over a `MTLRenderCommandEncoder` that is already
/// inside the map's render pass. You cannot begin another pass on it, and the
/// simulation needs three of its own (advect, fade, draw) before the result can
/// be composited. So this class keeps its own `MTLCommandQueue`, encodes the
/// offscreen work into its own command buffer, and draws only the final quad
/// into the map's encoder.
///
/// That leaves an ordering question the API gives no way to answer directly:
/// two queues have no guaranteed order, so compositing a trail texture the
/// other queue wrote *this frame* is a read of memory that may not be written
/// yet. The fix is the ping-pong that was already there for other reasons —
/// the composite samples the buffer written on the **previous** frame, so a
/// whole vsync separates the write from the read. It costs one frame of
/// latency on a trail effect, which is invisible, and it avoids a
/// `waitUntilCompleted` that would stall the pipeline every frame. If this ever
/// does tear, the rigorous fix is an `MTLSharedEvent` signalled by the
/// offscreen buffer, not a smaller ping-pong.
class WindParticleLayer: MLNCustomStyleLayer {
  private static let stateEdge = 256
  private static let blocks = 16
  private static let blockFrames = 8
  private static let blockRows = stateEdge / blocks
  private static let lifetime = blocks * blockFrames
  private static let particlesPerBlock = stateEdge * blockRows
  /// How much of the accumulated trail reaches the map.
  ///
  /// The reference's own number is 0.44, which with a -0.1 lift caps a stroke
  /// at 0.34 alpha — calibrated for Windy's dark basemap. DPIP composites over
  /// a saturated ECMWF ramp and the same grey barely registers, so both are
  /// raised. Kept identical to `COMPOSITE_GAIN` / `COMPOSITE_LIFT` in
  /// `WindParticleLayer.java`; these two are the brightness knob.
  private static let compositeGain: Float = 0.62
  private static let compositeLift: Float = -0.05

  /// `glSpeedPx`: the reference's speed unit, screen pixels per second.
  private static let speedPx = 100.0

  /// `glCountMul` for wind, times the halving the reference applies off
  /// desktop. The reference's own value is 1 and it then halves the count on
  /// anything that is not a desktop browser — a rule from the era of browsers
  /// on phones, which leaves 500-odd particles on a modern handset and reads as
  /// scattered rather than as a flow. 2 cancels the halving and leaves Windy's
  /// desktop density. The count is linear in this; it is the one knob.
  ///
  /// Kept identical to `GL_COUNT_MUL` in `WindParticleLayer.java`.
  private static let glCountMul = 2.0

  /// `zoom2speed`, transcribed. Wind is slowed at the widest zooms, where a
  /// degree is a handful of pixels and true speed reads as a flicker.
  ///
  /// `WindTuning.speedScale` and `speedFactor` are deliberately not used for
  /// this. They belong to the Dart fallback renderer, whose particles move in
  /// fractions of the global grid — 32 there is not 32 pixels per second here.
  private static let zoomToSpeed: [Double] = [
    0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  ]

  private static func zoomSpeed(_ zoom: Double) -> Double {
    zoomToSpeed[max(0, min(zoomToSpeed.count - 1, Int(zoom.rounded())))]
  }

  /// Fade-in over the first 20 % of a life, out over the last 30 %. The two
  /// windows straddle the rebirth boundary, which is what makes the cycle
  /// seamless: the block being reseeded is already invisible when it is.
  private static let alphaLut: [Float] = {
    var lut = [Float](repeating: 1, count: lifetime)
    let fadeInEnd = Int((0.2 * Double(lifetime)).rounded())
    let fadeOutFrom = lifetime - Int((0.3 * Double(lifetime)).rounded())
    for i in 0..<lifetime {
      if i < fadeInEnd {
        lut[i] = Float(pow(Double(i) / Double(fadeInEnd), 0.9))
      } else if i >= fadeOutFrom {
        lut[i] = Float(
          pow(Double(lifetime - i) / Double(lifetime - fadeOutFrom), 0.8))
      }
    }
    return lut
  }()

  // ------------------------------------------------------------------- inputs

  private var tuning = WindTuning()
  private var pendingField: WindFieldUpload?
  private var playing = false

  /// Drives the redraw request. See `startClock`.
  private var clock: CADisplayLink?

  // ------------------------------------------------------------------- device

  private var device: MTLDevice?
  private var queue: MTLCommandQueue?
  private var updatePipeline: MTLRenderPipelineState?
  private var fadePipeline: MTLRenderPipelineState?
  private var segmentPipeline: MTLRenderPipelineState?
  private var compositePipeline: MTLRenderPipelineState?
  private var projectPipeline: MTLRenderPipelineState?
  private var stateSampler: MTLSamplerState?
  private var windSampler: MTLSamplerState?
  private var projSampler: MTLSamplerState?

  private var stateTex: [MTLTexture] = []
  private var trailTex: [MTLTexture] = []
  private var windTex: MTLTexture?
  private var projTex: MTLTexture?
  private var reseedBuffer: MTLBuffer?
  private var vertexBuffer: MTLBuffer?
  private var indexBuffer: MTLBuffer?

  private var stateFront = 0
  private var trailFront = 0
  private var trailSize = CGSize.zero

  // -------------------------------------------------------------------- field

  private var windRange = SIMD4<Float>()
  private var fieldLat0: Float = 0
  private var fieldLatSpan: Float = 0
  private var fieldLon0: Float = 0
  private var windWidth = 0
  private var windHeight = 0
  private var fieldReady = false

  // -------------------------------------------------------------------- clock

  private var stateBlock = 0
  private var blockFrame = 0.0
  private var lastFrameTime = 0.0
  private var frame: UInt64 = 0

  /// The camera the surviving trail pixels were drawn for.
  private var haveTrailCam = false
  private var trailZoom = 0.0
  private var trailLat = 0.0
  private var trailLng = 0.0
  private var trailBearing = 0.0
  private var trailsDirty = true
  private var trailShift = SIMD2<Float>()

  // ------------------------------------------------------------------- public

  func setTuning(_ value: WindTuning) {
    tuning = value
  }

  func setField(_ upload: WindFieldUpload) {
    pendingField = upload
    // A new forecast is new weather: streaks laid down by the previous one do
    // not describe what is about to animate.
    trailsDirty = true
    setNeedsDisplay()
  }

  func setPlaying(_ value: Bool) {
    guard playing != value else { return }
    playing = value
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
    // Clearing here on top of that was measured as a flicker on Android, where
    // `playing` follows map gestures: a tap or a short drag wiped a second of
    // accumulated streaks, which is the entire visible field.
    lastFrameTime = 0
    value ? startClock() : stopClock()
    setNeedsDisplay()
  }

  /// Asks for the next frame, once per vsync.
  ///
  /// **Not from inside `draw`.** Calling `setNeedsDisplay()` there is a redraw
  /// that schedules a redraw: nothing throttles it, the render thread never
  /// yields, and the map — the whole app, since the platform view owns that
  /// thread — stops responding. A display link is the same request made by the
  /// clock that already governs when a frame is worth drawing.
  private func startClock() {
    guard clock == nil else { return }
    let link = CADisplayLink(target: self, selector: #selector(tick))
    link.add(to: .main, forMode: .common)
    clock = link
  }

  private func stopClock() {
    clock?.invalidate()
    clock = nil
  }

  @objc private func tick() {
    setNeedsDisplay()
  }

  override func willMove(from mapView: MLNMapView) {
    // The link retains its target, so a layer removed while playing would keep
    // firing — and keep asking a detached layer to draw.
    stopClock()
    super.willMove(from: mapView)
  }

  // ------------------------------------------------------------------- render

  override func draw(
    in mapView: MLNMapView, with context: MLNStyleLayerDrawingContext
  ) {
    guard let encoder = renderEncoder else { return }
    guard context.size.width > 0, context.size.height > 0 else { return }

    if device == nil, !makeDevice(encoder: encoder) { return }
    if let upload = pendingField {
      pendingField = nil
      uploadField(upload)
    }
    guard fieldReady, playing else { return }

    let scale = mapView.contentScaleFactor
    let physical = CGSize(
      width: context.size.width * scale, height: context.size.height * scale)
    ensureTrailTextures(physical)
    guard !trailTex.isEmpty else { return }

    var u = uniforms(for: context, mapView: mapView)
    resolveTrailCamera(context: context, uniforms: &u)

    // Advance the float state at the tuned 60 Hz cadence. Render callbacks may
    // arrive at 120 Hz; the intervening map frames reuse the latest trail.
    // Two clocks, as the reference keeps them, and mixing them up is visible.
    //
    // Particles move by the *real* time since the last drawn frame. Quantising
    // that into whole 60 Hz steps lays down a quad twice as long as its
    // neighbours whenever two steps accumulate, and a double-length quad's flat
    // end is a rectangle nothing else covers. The block clock still advances in
    // whole frames, because the fade curve is a 128-entry table indexed by one.
    let now = ProcessInfo.processInfo.systemUptime
    var dt = 1.0 / 60.0
    if lastFrameTime != 0 {
      dt = max(0, min(0.1, now - lastFrameTime))
    }
    lastFrameTime = now

    // Offscreen work uses our own queue. A skipped simulation callback still
    // composites below because MapLibre supplied a fresh map framebuffer.
    encodeOffscreen(uniforms: &u, context: context, dt: dt)
    frame &+= 1

    composite(encoder: encoder, uniforms: &u)

    // One line, on the first frame that reaches the screen, and then rarely.
    //
    // The Android twin has had this since the day a shader failure and a stale
    // install turned out to look identical in the log. Here it answers the
    // questions a screenshot cannot: whether `context.size` arrives in points
    // or pixels, what the trail texture was actually allocated at, and where
    // the projection thinks the viewport is — the three numbers every
    // "everything is in the wrong half" bug so far has come down to.
    let ms = dt * 1000
    elapsedMin = min(elapsedMin, ms)
    elapsedMax = max(elapsedMax, ms)
    elapsedSum += ms
    elapsedN += 1

    if frame == 1 || frame % 600 == 0 {
      let wall = ProcessInfo.processInfo.systemUptime
      let fps = lastLogTime == 0 ? 0 : 600 / (wall - lastLogTime)
      lastLogTime = wall
      NSLog(
        "WindParticleLayer: frame=%llu fps=%.1f ctx=%.0fx%.0f scale=%.1f "
          + "trail=%.0fx%.0f half=%.1fx%.1f zoom=%.3f perBlock=%d total=%d "
          + "field=%dx%d fade=%.4f spacing[min=%.2f max=%.2f mean=%.2f]ms n=%d",
        frame, fps, context.size.width, context.size.height,
        mapView.contentScaleFactor, trailSize.width, trailSize.height,
        u.cam0.z, u.cam0.w, context.zoomLevel, lastPerBlock,
        lastPerBlock * Self.blocks, windWidth, windHeight, lastFade,
        elapsedMin, elapsedMax, elapsedSum / Double(max(1, elapsedN)), elapsedN)
      elapsedMin = .greatestFiniteMagnitude
      elapsedMax = 0
      elapsedSum = 0
      elapsedN = 0
    }
  }

  /// Kept only so the diagnostic line above can report it.
  private var lastPerBlock = 0

  /// The same counters the Android layer reports, so the two can be compared
  /// as numbers rather than as screenshots. Deposit rate and callback spacing
  /// are the two that decide how a stroke looks: how far a particle travels
  /// between the quads that draw it, and how evenly those quads land.
  private var lastLogTime = 0.0
  private var elapsedMin = Double.greatestFiniteMagnitude
  private var elapsedMax = 0.0
  private var elapsedSum = 0.0
  private var elapsedN = 0
  private var lastFade: Float = 0

  // ------------------------------------------------------------------- passes

  private func encodeOffscreen(uniforms u: inout WindUniforms,
                               context: MLNStyleLayerDrawingContext,
                               dt: Double) {
    guard let queue, let buffer = queue.makeCommandBuffer(),
          let updatePipeline, let fadePipeline, let segmentPipeline,
          let projectPipeline, let stateSampler, let windSampler,
          let projSampler, let windTex, let projTex,
          let vertexBuffer, let indexBuffer
    else { return }

    let t = tuning
    let zoom = context.zoomLevel
    // getAmount(width, height, zoom), transcribed. Proportional to the
    // viewport's area in logical points rather than to a constant, so a tablet
    // gets more particles than a phone at the same zoom:
    //
    //     amount(w, h, z) = min(15000, w · h / (50 · 1.6^(z-2)))
    //
    // then halved off desktop, and divided by the 65,536-particle pool.
    let amount = min(
      15000.0,
      (context.size.width * context.size.height
        / (50 * pow(1.6, zoom - 2))).rounded())
    let relativeAmount =
      amount * 0.5 / Double(Self.stateEdge * Self.stateEdge) * Self.glCountMul
    let perBlock = max(
      1,
      min(Self.particlesPerBlock,
          Int((relativeAmount * Double(Self.particlesPerBlock)).rounded())))
    lastPerBlock = perBlock

    // The block clock. One block is reseeded every `blockFrames` frames, so a
    // particle's age is known and `alphaLut` can fade it.
    // In frames of real time, not in callbacks.
    //
    // The reference advances this by `max(1, round(elapsed * 60))`, which is
    // correct only while the callback rate is the 60 Hz it assumes. A 120 Hz
    // panel then forces a whole frame per callback: the 128-frame lifetime
    // completes in 1.2 s instead of 2.13 s, every particle is reborn 1.77x too
    // often, and no trail lives long enough to become one. Advancing by the
    // real elapsed time costs nothing — the value is only ever read as an index
    // into the fade table and against the rebirth threshold.
    blockFrame += dt * 60
    // The block the cursor is leaving is the one reborn, and the cursor then
    // moves past it. Reseeding the block the cursor moves *to* instead puts the
    // newest generation one slot away from where the draw loop looks for it,
    // and the whole fade curve is applied off by one block — the youngest
    // strokes drawn at the dimmest alpha.
    var reseedTarget = -1
    if blockFrame >= Double(Self.blockFrames) {
      blockFrame -= Double(Self.blockFrames)
      reseedTarget = stateBlock
      stateBlock = (stateBlock + 1) % Self.blocks
    }
    // --- the global grid, redrawn viewport-aligned
    //
    // Once a frame, not once per vertex. After this the particles are in screen
    // space and every remaining pass is the reference's, unmodified.
    let projPass = MTLRenderPassDescriptor()
    projPass.colorAttachments[0].texture = projTex
    projPass.colorAttachments[0].loadAction = .dontCare
    projPass.colorAttachments[0].storeAction = .store
    if let enc = buffer.makeRenderCommandEncoder(descriptor: projPass) {
      enc.setRenderPipelineState(projectPipeline)
      enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
      enc.setFragmentTexture(windTex, index: 0)
      enc.setFragmentSamplerState(windSampler, index: 0)
      enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
      enc.endEncoding()
    }

    // The stored 0..1 direction becomes a displacement of at most `o`
    // normalised units, where `o` is how far the reference's speed unit carries
    // a particle this frame. glSpeedPx is pixels per second at full strength,
    // so the viewport's own size is the denominator — which is why the speed no
    // longer has to be retuned per zoom level.
    let seconds = dt
    let timeScale = Self.speedPx * Self.zoomSpeed(zoom) * t.pixelRatio
    let ox = Float(seconds * timeScale / Double(trailSize.width))
    let oy = Float(seconds * timeScale / Double(trailSize.height))
    u.disp = SIMD4(2 * ox, 2 * oy, -ox, -oy)

    // --- advect every particle into the back state texture
    let statePass = MTLRenderPassDescriptor()
    statePass.colorAttachments[0].texture = stateTex[1 - stateFront]
    statePass.colorAttachments[0].loadAction = .dontCare
    statePass.colorAttachments[0].storeAction = .store
    if let enc = buffer.makeRenderCommandEncoder(descriptor: statePass) {
      enc.setRenderPipelineState(updatePipeline)
      enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
      enc.setFragmentTexture(stateTex[stateFront], index: 0)
      enc.setFragmentTexture(projTex, index: 1)
      enc.setFragmentSamplerState(stateSampler, index: 0)
      enc.setFragmentSamplerState(projSampler, index: 1)
      enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
      enc.endEncoding()
    }
    stateFront = 1 - stateFront
    if reseedTarget >= 0 { reseedBlock(reseedTarget, in: buffer) }

    // --- fade the surviving streaks into the back trail texture, then draw
    //     this frame's sixteen generations over them
    // Clearing the *other* buffer first, in its own pass and before anything
    // else is encoding.
    //
    // Both have to go, not just the one about to be drawn into: the swap at the
    // end of the frame would otherwise hand the ghosts straight back. The
    // reason it is here rather than beside the branch below is Metal's rule
    // that one command buffer may have only one live encoder — opening a second
    // while the trail pass was already encoding is an API violation, and on a
    // device it does not report and return, it wedges the render thread. That
    // is what "iOS completely frozen" was.
    let wasDirty = trailsDirty
    if wasDirty {
      clearTrail(other: trailFront, in: buffer)
      trailsDirty = false
    }

    let trailPass = MTLRenderPassDescriptor()
    trailPass.colorAttachments[0].texture = trailTex[1 - trailFront]
    trailPass.colorAttachments[0].storeAction = .store
    if wasDirty {
      trailPass.colorAttachments[0].loadAction = .clear
      trailPass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
    } else {
      trailPass.colorAttachments[0].loadAction = .dontCare
    }

    guard let enc = buffer.makeRenderCommandEncoder(descriptor: trailPass) else {
      buffer.commit()
      return
    }
    if !wasDirty {
      // Keyed to real time, not to the callback count: a 120 Hz panel delivers
      // roughly twice the callbacks and a per-callback 0.97 would halve the
      // trail length against a 60 Hz device. `dt * 60` restores the intended
      // per-second decay on any refresh rate.
      u.fld.w = pow(t.fadeOpacity(zoom), Float(dt * 60))
      lastFade = u.fld.w
      u.misc = SIMD4(trailShift.x, trailShift.y, 0, 0)
      enc.setRenderPipelineState(fadePipeline)
      enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
      enc.setFragmentTexture(trailTex[trailFront], index: 0)
      enc.setFragmentSamplerState(stateSampler, index: 0)
      enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    // Line width in NDC. Wider at high zoom, where there are fewer particles —
    // which is what keeps the apparent density even across the range.
    let widthFactor = max(1, t.lineWidth(zoom) * Float(t.particleWidth * t.pixelRatio))
    let aw = (widthFactor + 1) / Float(trailSize.width)
    let ah = (widthFactor + 1) / Float(trailSize.height)
    let halfLen = max(1, widthFactor * 0.8)
    u.fld.z = halfLen
    u.lineW = SIMD4(aw * 2 / 255, ah * 2 / 255, -aw, -ah)
    u.edge = SIMD4(0, 0, halfLen * 2 / 255, -halfLen)

    enc.setRenderPipelineState(segmentPipeline)
    enc.setVertexBuffer(vertexBuffer, offset: 0, index: 1)
    enc.setVertexTexture(stateTex[stateFront], index: 0)
    enc.setVertexTexture(stateTex[1 - stateFront], index: 1)
    enc.setVertexSamplerState(stateSampler, index: 0)

    // The reference walks the blocks in storage order and walks the fade curve
    // backwards alongside them, so the block reseeded most recently — the one
    // just below the write cursor — lands at age ~0 and every earlier block is
    // another `blockFrames` older.
    var age = Int(blockFrame)
    for b in 0..<Self.blocks {
      let block = (stateBlock - 1 - b + 2 * Self.blocks) % Self.blocks
      u.block = SIMD4(1 / Float(Self.stateEdge), 1 / Float(Self.stateEdge), 0,
                      Float(block) / Float(Self.blocks))
      let a = Self.alphaLut[age % Self.lifetime]
      u.colour = SIMD4(a, a, a, a)
      enc.setVertexBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
      enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
      enc.drawIndexedPrimitives(
        type: .triangle,
        indexCount: perBlock * 6,
        indexType: .uint16,
        indexBuffer: indexBuffer,
        indexBufferOffset: 0)
      age += Self.blockFrames
    }
    enc.endEncoding()
    buffer.commit()
    trailFront = 1 - trailFront
  }

  private func clearTrail(other index: Int, in buffer: MTLCommandBuffer) {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = trailTex[index]
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
    pass.colorAttachments[0].storeAction = .store
    buffer.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
  }

  /// The one pass that lands in the map's own encoder.
  private func composite(encoder: MTLRenderCommandEncoder,
                         uniforms u: inout WindUniforms) {
    guard let compositePipeline, let stateSampler, !trailTex.isEmpty else { return }
    u.fld.w = Self.compositeGain
    u.misc = SIMD4(0, 0, Self.compositeLift, 0)
    encoder.setRenderPipelineState(compositePipeline)
    encoder.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
    // The buffer written on the previous frame — see the class comment.
    encoder.setFragmentTexture(trailTex[trailFront], index: 0)
    encoder.setFragmentSamplerState(stateSampler, index: 0)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
  }

  // -------------------------------------------------------------------- camera

  private func uniforms(for context: MLNStyleLayerDrawingContext,
                        mapView: MLNMapView) -> WindUniforms {
    var u = WindUniforms()
    let zoom = context.zoomLevel
    let world = 512 * pow(2.0, zoom)
    let cx = (context.centerCoordinate.longitude + 180) / 360 * world
    let cy = Self.mercatorY(context.centerCoordinate.latitude) * world
    let r = -context.direction * Double.pi / 180

    u.cam0 = SIMD4(Float(cx), Float(cy),
                   Float(context.size.width / 2), Float(context.size.height / 2))
    u.cam1 = SIMD4(Float(cos(r)), Float(sin(r)), Float(world), fieldLon0)
    u.fld = SIMD4(fieldLat0, fieldLatSpan, 0, 0)
    u.range = SIMD4(windRange.x, windRange.z, windRange.y, windRange.w)
    // (speedMul, speedMin, cPow) = (1/glMaxSpeedParam,
    //  glMinSpeedParam/glMaxSpeedParam, glSpeedCurvePowParam/2).
    u.curve = SIMD4(1.0 / 30.0, 1.5 / 30.0, 0.35, 0)
    return u
  }

  /// Decides whether the surviving trails can be re-anchored or must be
  /// dropped, and by how much to shift them if they can.
  ///
  /// A pan is recoverable: the whole image translated, and re-sampling it at
  /// the offset puts it back under the map it describes. A zoom or a rotation
  /// is not — they scale and turn the image, and a texture offset cannot
  /// express either. Panning is both the common case and the one that looks
  /// worst when discarded.
  private func resolveTrailCamera(context: MLNStyleLayerDrawingContext,
                                  uniforms u: inout WindUniforms) {
    trailShift = SIMD2<Float>()
    let zoom = context.zoomLevel
    let lat = context.centerCoordinate.latitude
    let lng = context.centerCoordinate.longitude
    let bearing = context.direction

    if haveTrailCam {
      if abs(zoom - trailZoom) > 1e-4 || abs(bearing - trailBearing) > 0.05 {
        trailsDirty = true
      } else if abs(lat - trailLat) > 1e-9 || abs(lng - trailLng) > 1e-9 {
        // `projectField` puts a world point at half + (w - centre) rotated by
        // `rot`, so moving the centre by d moves every fixed point on screen by
        // -d, rotated the same way.
        let world = 512 * pow(2.0, zoom)
        let dcx = ((lng - trailLng) / 360) * world
        let dcy = (Self.mercatorY(lat) - Self.mercatorY(trailLat)) * world
        let r = -bearing * Double.pi / 180
        let dx = -dcx * cos(r) + dcy * sin(r)
        let dy = -dcx * sin(r) - dcy * cos(r)
        // The fade pass samples at `1 - tex`, where u maps to screen x and v to
        // screen y with y pointing up. Content that moved by (dx, dy) is read
        // back from (-dx, +dy).
        let shiftX = -dx / Double(u.cam0.z * 2)
        let shiftY = dy / Double(u.cam0.w * 2)
        if abs(shiftX) > 0.5 || abs(shiftY) > 0.5 {
          // More than half the viewport: almost nothing survives the move, and
          // what does is a thin band of stale streaks along one edge.
          trailsDirty = true
        } else {
          trailShift = SIMD2(Float(shiftX), Float(shiftY))
        }
      }
    }
    haveTrailCam = true
    trailZoom = zoom
    trailLat = lat
    trailLng = lng
    trailBearing = bearing
  }

  private static func mercatorY(_ lat: Double) -> Double {
    let clamped = max(-85.051129, min(85.051129, lat))
    let phi = clamped * Double.pi / 180
    return (1 - log(tan(Double.pi / 4 + phi / 2)) / Double.pi) / 2
  }

  // ------------------------------------------------------------------ resources

  private func makeDevice(encoder: MTLRenderCommandEncoder) -> Bool {
    let dev = encoder.device
    guard let q = dev.makeCommandQueue() else { return false }
    let library: MTLLibrary
    do {
      library = try dev.makeLibrary(source: WindParticleShaders.source, options: nil)
    } catch {
      NSLog("WindParticleLayer: shader compile failed: \\(error)")
      return false
    }
    guard
      let quadV = library.makeFunction(name: "quadVertex"),
      let updateV = library.makeFunction(name: "updateVertex"),
      let updateF = library.makeFunction(name: "updateFragment"),
      let screenF = library.makeFunction(name: "screenFragment"),
      let segV = library.makeFunction(name: "segmentVertex"),
      let segF = library.makeFunction(name: "segmentFragment"),
      let projectF = library.makeFunction(name: "windProjectFragment")
    else {
      NSLog("WindParticleLayer: a shader function is missing")
      return false
    }

    func pipeline(_ v: MTLFunction, _ f: MTLFunction, _ format: MTLPixelFormat,
                  premultipliedBlend: Bool) -> MTLRenderPipelineState? {
      let d = MTLRenderPipelineDescriptor()
      d.vertexFunction = v
      d.fragmentFunction = f
      d.colorAttachments[0].pixelFormat = format
      if premultipliedBlend {
        // The generations overlap constantly, and straight alpha would let the
        // newest erase the older ones instead of adding to them.
        d.colorAttachments[0].isBlendingEnabled = true
        d.colorAttachments[0].sourceRGBBlendFactor = .one
        d.colorAttachments[0].sourceAlphaBlendFactor = .one
        d.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        d.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
      }
      return try? dev.makeRenderPipelineState(descriptor: d)
    }

    updatePipeline = pipeline(updateV, updateF, .rgba8Unorm, premultipliedBlend: false)
    projectPipeline = pipeline(updateV, projectF, .rgba8Unorm, premultipliedBlend: false)
    fadePipeline = pipeline(quadV, screenF, .rgba8Unorm, premultipliedBlend: false)
    segmentPipeline = pipeline(segV, segF, .rgba8Unorm, premultipliedBlend: true)
    // The map's drawable format is not exposed anywhere on this API, so both
    // plausible orders are tried. Guessing one and shipping it would fail as a
    // pipeline that silently never builds — which looks exactly like a layer
    // that was never added.
    compositePipeline =
      pipeline(quadV, screenF, .bgra8Unorm, premultipliedBlend: true)
      ?? pipeline(quadV, screenF, .rgba8Unorm, premultipliedBlend: true)

    guard updatePipeline != nil, fadePipeline != nil, segmentPipeline != nil,
          compositePipeline != nil, projectPipeline != nil
    else {
      NSLog("WindParticleLayer: pipeline creation failed")
      return false
    }

    let nearest = MTLSamplerDescriptor()
    nearest.minFilter = .nearest
    nearest.magFilter = .nearest
    nearest.sAddressMode = .clampToEdge
    nearest.tAddressMode = .clampToEdge
    stateSampler = dev.makeSamplerState(descriptor: nearest)

    let wind = MTLSamplerDescriptor()
    // Linear, not nearest. The projection pass resamples this grid at one
    // screen pixel per texel and ECMWF's cells are degrees wide; nearest would
    // hand the particles a staircase and they would move in squares.
    wind.minFilter = .linear
    wind.magFilter = .linear
    // Longitude is cyclic; latitude is not. Wrapping the second walks a
    // particle off the pole onto the other pole.
    wind.sAddressMode = .repeat
    wind.tAddressMode = .clampToEdge
    windSampler = dev.makeSamplerState(descriptor: wind)

    // The reprojected field is in screen space, so it wraps nowhere.
    let proj = MTLSamplerDescriptor()
    proj.minFilter = .linear
    proj.magFilter = .linear
    proj.sAddressMode = .clampToEdge
    proj.tAddressMode = .clampToEdge
    projSampler = dev.makeSamplerState(descriptor: proj)

    guard makeStateTextures(dev), makeGeometry(dev) else { return false }
    device = dev
    queue = q
    return true
  }

  private func makeStateTextures(_ dev: MTLDevice) -> Bool {
    // Eight bits a channel, matching Android, because the position encoding is
    // designed around that quantisation: `rg` is the low byte and `ba` the
    // high, and `ba + rg / 255.5` inverts `fract(pos * 255 + 0.25/255)` only
    // when the values have actually been rounded to bytes. Stored as float the
    // pair stops round-tripping and every particle drifts a little each frame.
    //
    // Private, and rebirth reaches them through a blit rather than a CPU
    // write — see `reseedBlock` for why the ordering matters.
    let d = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm, width: Self.stateEdge, height: Self.stateEdge,
      mipmapped: false)
    d.usage = [.shaderRead, .renderTarget]
    d.storageMode = .private
    var seed = [UInt8](repeating: 0, count: Self.stateEdge * Self.stateEdge * 4)
    for i in 0..<seed.count { seed[i] = UInt8.random(in: 0...255) }

    // A private texture cannot be written from the CPU, so the initial seed
    // goes through a shared staging texture and a blit that is waited on once,
    // at start-up, where a stall costs nothing.
    let staging = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm, width: Self.stateEdge, height: Self.stateEdge,
      mipmapped: false)
    staging.usage = [.shaderRead]
    staging.storageMode = .shared
    guard let stage = dev.makeTexture(descriptor: staging) else { return false }
    stage.replace(
      region: MTLRegionMake2D(0, 0, Self.stateEdge, Self.stateEdge),
      mipmapLevel: 0, withBytes: seed, bytesPerRow: Self.stateEdge * 4)

    stateTex = []
    guard let q = dev.makeCommandQueue(), let buf = q.makeCommandBuffer(),
          let blit = buf.makeBlitCommandEncoder()
    else { return false }
    for _ in 0..<2 {
      guard let t = dev.makeTexture(descriptor: d) else { return false }
      blit.copy(
        from: stage, sourceSlice: 0, sourceLevel: 0,
        sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
        sourceSize: MTLSize(width: Self.stateEdge, height: Self.stateEdge, depth: 1),
        to: t, destinationSlice: 0, destinationLevel: 0,
        destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
      stateTex.append(t)
    }
    blit.endEncoding()
    buf.commit()
    buf.waitUntilCompleted()

    // Rebirth's staging buffer, allocated once and refilled in place.
    reseedBuffer = dev.makeBuffer(
      length: Self.stateEdge * Self.blockRows * 4, options: .storageModeShared)
    return reseedBuffer != nil
  }

  /// Rebirth, the reference's way: overwrite one block's rows of both state
  /// textures with uniform random bytes.
  ///
  /// Uniform in the packed encoding is uniform on screen, because the packing
  /// is linear. No density weighting, no view test, no rejection loop — the
  /// previous version needed all three only because a respawn drawn in field
  /// space mostly landed off-screen.
  ///
  /// Both textures, and that is the point of doing it here rather than in the
  /// update shader. A particle whose front state is new and whose back state is
  /// its previous life gets a segment drawn between two unrelated points: the
  /// streak from one edge of the map to the other. Writing both leaves the
  /// segment zero-length for one frame instead, which `segmentVertex` discards.
  ///
  /// Encoded as a blit into the frame's own command buffer rather than written
  /// straight to the texture. A CPU `replace` on a shared texture races the GPU
  /// twice over: the update pass encoded moments earlier has not run yet, and
  /// the previous frame's buffer may still be reading. It happened to look
  /// right — the update would then read the fresh noise and advect it, so the
  /// two ends differed and no NaN appeared — but "happened to" is the whole
  /// problem, and it also put the rebirth one frame out of step with Android.
  private func reseedBlock(_ block: Int, in buffer: MTLCommandBuffer) {
    guard stateTex.count == 2, let reseedBuffer else { return }
    let bytes = Self.stateEdge * Self.blockRows * 4
    let dst = reseedBuffer.contents().bindMemory(to: UInt8.self, capacity: bytes)
    for i in 0..<bytes { dst[i] = UInt8.random(in: 0...255) }
    guard let blit = buffer.makeBlitCommandEncoder() else { return }
    for t in stateTex {
      blit.copy(
        from: reseedBuffer, sourceOffset: 0,
        sourceBytesPerRow: Self.stateEdge * 4, sourceBytesPerImage: bytes,
        sourceSize: MTLSize(width: Self.stateEdge, height: Self.blockRows, depth: 1),
        to: t, destinationSlice: 0, destinationLevel: 0,
        destinationOrigin: MTLOrigin(x: 0, y: block * Self.blockRows, z: 0))
    }
    blit.endEncoding()
  }

  private func makeGeometry(_ dev: MTLDevice) -> Bool {
    // One block's worth of quads, reused for all sixteen. Four bytes a vertex:
    // the texel it reads and which corner of the quad it is.
    var verts = [UInt8]()
    var inds = [UInt16]()
    verts.reserveCapacity(Self.particlesPerBlock * 16)
    inds.reserveCapacity(Self.particlesPerBlock * 6)
    let corners: [(UInt8, UInt8)] = [(0, 0), (255, 0), (255, 255), (0, 255)]
    for i in 0..<Self.particlesPerBlock {
      let x = UInt8(i % Self.stateEdge)
      let y = UInt8(i / Self.stateEdge)
      for c in corners {
        verts.append(x); verts.append(y); verts.append(c.0); verts.append(c.1)
      }
      let base = UInt16(i * 4)
      inds.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }
    guard
      let vb = dev.makeBuffer(bytes: verts, length: verts.count, options: .storageModeShared),
      let ib = dev.makeBuffer(
        bytes: inds, length: inds.count * 2, options: .storageModeShared)
    else { return false }
    vertexBuffer = vb
    indexBuffer = ib
    return true
  }

  private func ensureTrailTextures(_ size: CGSize) {
    guard let device else { return }
    let w = Int(size.width.rounded())
    let h = Int(size.height.rounded())
    guard w > 0, h > 0 else { return }
    if trailSize == size, trailTex.count == 2 { return }

    let d = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    d.usage = [.shaderRead, .renderTarget]
    d.storageMode = .private
    trailTex = []
    for _ in 0..<2 {
      guard let t = device.makeTexture(descriptor: d) else { return }
      trailTex.append(t)
    }
    // The reprojected field shares the trails' resolution: after it, a
    // particle's stored position is its position on screen.
    let pd = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    pd.usage = [.shaderRead, .renderTarget]
    pd.storageMode = .private
    projTex = device.makeTexture(descriptor: pd)

    trailSize = size
    trailsDirty = true
  }

  /// Interleaves the two quantised planes into the two channels the wind
  /// texture carries. Done here, once per forecast frame, rather than in Dart:
  /// the wire then carries the untouched WND1 body and the parser stays in one
  /// place.
  private func uploadField(_ upload: WindFieldUpload) {
    guard let device else { return }
    let n = upload.width * upload.height
    guard upload.bytes.count >= upload.planeOffset + n * 2 else {
      NSLog("WindParticleLayer: wind field truncated")
      return
    }
    var texels = [UInt8](repeating: 0, count: n * 2)
    upload.bytes.withUnsafeBytes { raw in
      let base = raw.baseAddress!.advanced(by: upload.planeOffset)
        .assumingMemoryBound(to: UInt8.self)
      for i in 0..<n {
        texels[i * 2] = base[i]
        texels[i * 2 + 1] = base[n + i]
      }
    }
    let d = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rg8Unorm, width: upload.width, height: upload.height,
      mipmapped: false)
    d.usage = [.shaderRead]
    d.storageMode = .shared
    guard let t = device.makeTexture(descriptor: d) else { return }
    t.replace(
      region: MTLRegionMake2D(0, 0, upload.width, upload.height),
      mipmapLevel: 0, withBytes: texels, bytesPerRow: upload.width * 2)
    windTex = t
    windWidth = upload.width
    windHeight = upload.height
    windRange = SIMD4(upload.uMin, upload.uMax, upload.vMin, upload.vMax)
    fieldLat0 = upload.lat0
    fieldLatSpan = upload.dLat * Float(upload.height)
    fieldLon0 = upload.lon0
    fieldReady = true
  }
}
