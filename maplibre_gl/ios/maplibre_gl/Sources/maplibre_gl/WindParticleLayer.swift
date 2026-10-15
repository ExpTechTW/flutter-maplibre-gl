import Foundation
import Metal
import MetalKit
import QuartzCore
import MapLibre
import simd

/// Uniforms shared by every pass.
///
/// **Hand-packed so there is no implicit padding anywhere**, and the Metal
/// declaration in `WindParticleShaders` is field-for-field identical. Metal and
/// Swift agree on the alignment of `float4`, but a struct that mixes member
/// sizes is free to pad differently on each side — and a struct that drifts by
/// four bytes does not fail to compile or bind. It reads the next field along,
/// which shows up as a field blowing at the wrong speed, or as nothing drawing
/// at all, with no error anywhere. Fifteen `float4`s and nothing else.
struct WindUniforms {
  var disp = SIMD4<Float>()    // update: byte b moves by (b·mul + add) per axis
  var block = SIMD4<Float>()   // segment: (1/edge, 1/edge, 0, block/blocks)
  var line = SIMD4<Float>()    // segment: (w·2/255, h·2/255, -w, -h)
  var edge = SIMD4<Float>()    // segment: (0, 0, l·2/255, -l)
  var colour = SIMD4<Float>()  // segment: this generation's alpha, four times
  var seg = SIMD4<Float>()     // segment: (feather length, -, -, -)
  var range = SIMD4<Float>()   // project: (uMin, vMin, uMax, vMax)
  var grid = SIMD4<Float>()    // project: (lon0, lat0, latSpan, -)
  var curve = SIMD4<Float>()   // project: (speedMul, speedMin, cPow, -)
  var texel = SIMD4<Float>()   // project: (half texel u, half texel v, -, -)
  var cam0 = SIMD4<Float>()    // project: (centreX, centreY, halfWidth, halfHeight)
  var cam1 = SIMD4<Float>()    // project: (cos(-bearing), sin(-bearing), worldPx, -)
  var shift = SIMD4<Float>()   // fade: (shiftX, shiftY, fade, -)
  var affine0 = SIMD4<Float>() // composite: (origin.x, origin.y, xAxis.x, xAxis.y)
  var affine1 = SIMD4<Float>() // composite: (yAxis.x, yAxis.y, gain, lift)
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

/// GPU wind-particle field drawn inside the map's own Metal pass, the way
/// Windy's `gl-particles` draws it.
///
/// The Android twin, `WindParticleLayer.java`, carries the long version of the
/// pipeline; the two are kept pass for pass and number for number. In short:
/// everything happens in a *reference frame* — the viewport as it was when the
/// camera last came to rest. The global grid is resampled under that viewport
/// once per settle; particles are 16-bit positions in it, advected on the GPU;
/// the newest segment of each is drawn into a trail texture that decays by
/// 0.97 a frame; and the trails are composited as a quad placed wherever that
/// viewport now is. A pan or a pinch carries the field with the map. When the
/// camera settles the reference follows it: shifted after a pan, cleared and
/// regrown after a zoom or a rotation.
///
/// ## What Metal makes different
///
/// `MLNCustomStyleLayer` hands over a `MTLRenderCommandEncoder` that is already
/// inside the map's render pass. You cannot begin another pass on it, and the
/// simulation needs several of its own before the result can be composited. So
/// this class keeps its own `MTLCommandQueue`, encodes the offscreen work into
/// its own command buffer, and draws only the final quad into the map's
/// encoder.
///
/// ### Borrowing someone else's encoder
///
/// * **The pipeline must describe the map's attachments, not ours.** Colour
///   format comes from `MLNMapView.backendResource()`'s MTKView; the pass also
///   carries a `.depth32Float_stencil8` attachment, and a pipeline that omits
///   it can never be valid there. An invalid pipeline is not a slow path — the
///   draw is dropped, and nothing on screen says why.
/// * **The depth/stencil *state* is inherited.** The map stencils its own
///   geometry; whatever mask the previous layer left is still bound. Depth
///   always, depth write off, stencil unconfigured.
///
/// Both are what MapLibre's own
/// `platform/darwin/app/PluginLayerExampleMetalRendering.mm` does.
///
/// ### Ordering across two queues
///
/// Two queues have no guaranteed order, so compositing a trail texture the
/// other queue wrote *this frame* is a read of memory that may not be written
/// yet. Hence two trail textures where Android fades one in place: the
/// composite samples the one written on the **previous** deposit, so a whole
/// frame sits between the write and the read, and the fade is a pass from one
/// into the other. It also means the composite places the trails by the
/// reference they were drawn in, which after a settle is not yet the current
/// one — `shownRef` below.
class WindParticleLayer: MLNCustomStyleLayer {
  // The reference's numbers; see the Android twin for where each comes from.
  private static let stateEdge = 256
  private static let blocks = 16
  private static let blockFrames = 8
  private static let blockRows = stateEdge / blocks
  private static let lifetime = blocks * blockFrames
  private static let particlesPerBlock = stateEdge * blockRows
  private static let amountMax = 15000.0
  private static let countMul = 0.5
  private static let lineWidth: [Double] = [
    0.6, 0.6, 0.6, 1, 1.2, 1.6, 1.8, 2, 2.2, 2.4, 2.4, 2.4, 2.4, 2.6, 2.8, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
  ]
  private static let particleWidth = 1.3
  private static let zoomToSpeed: [Double] = [
    0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
  ]
  private static let speedPx = 100.0
  private static let curve = SIMD4<Float>(1.0 / 30.0, 1.5 / 30.0, 0.35, 0)
  private static let fade = 0.97
  private static let compositeGain: Float = 0.44
  private static let compositeLift: Float = -0.1
  private static let alphaRate = 1.8
  private static let trailScaleDpr: CGFloat = 1.5
  private static let trailScale: CGFloat = 0.8
  private static let trailMax = 2048
  private static let settleSeconds = 0.1
  private static let step = 1.0 / 60.0
  /// A stored velocity byte b is (b − 128) / 127 of full speed.
  private static let dispMul: Float = 255.0 / 127.0
  private static let dispAdd: Float = 128.0 / 127.0

  /// In over the first 20 % of a life (power 0.9), out over the last 30 % (0.8).
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

  /// The reference indexes its tables by Leaflet zoom, one above MapLibre's.
  private static func windyZoom(_ zoom: Double) -> Int { Int((zoom + 1).rounded()) }

  private static func at(_ table: [Double], _ index: Int) -> Double {
    table[max(0, min(table.count - 1, index))]
  }

  // ------------------------------------------------------------------- inputs

  private var pendingField: WindFieldUpload?
  private var playing = false
  private var restart = false

  /// Drives the redraw request. See `startClock`.
  private var clock: CADisplayLink?

  // ------------------------------------------------------------------- device

  private var device: MTLDevice?
  private var queue: MTLCommandQueue?
  private var projectPipeline: MTLRenderPipelineState?
  private var updatePipeline: MTLRenderPipelineState?
  private var stateShiftPipeline: MTLRenderPipelineState?
  private var fadePipeline: MTLRenderPipelineState?
  private var segmentPipeline: MTLRenderPipelineState?
  private var compositePipeline: MTLRenderPipelineState?
  /// Depth always, no write, no stencil test — the state the map's own
  /// encoder is *not* guaranteed to be in. See `composite`.
  private var passThroughDepth: MTLDepthStencilState?
  private var nearestSampler: MTLSamplerState?
  private var linearSampler: MTLSamplerState?
  private var fieldSampler: MTLSamplerState?

  private var stateTex: [MTLTexture] = []
  private var trailTex: [MTLTexture] = []
  private var velTex: MTLTexture?
  private var fieldU: MTLTexture?
  private var fieldV: MTLTexture?
  private var reseedBuffer: MTLBuffer?
  private var vertexBuffer: MTLBuffer?
  private var indexBuffer: MTLBuffer?

  private var stateFront = 0
  private var trailFront = 0
  private var physicalSize = CGSize.zero
  private var trailWidth = 0
  private var trailHeight = 0

  /// Kept for the life of the layer. Building a pass descriptor per frame is
  /// an allocation on the render thread.
  private let targetPass = MTLRenderPassDescriptor()

  // -------------------------------------------------------------------- field

  private var fieldReady = false
  private var fieldWidth = 0
  private var fieldHeight = 0
  private var range = SIMD4<Float>()
  private var grid = SIMD4<Float>()

  // ------------------------------------------------------------------- camera

  /// One frame's camera with its projection inputs. Logical points throughout.
  private struct Camera: Equatable {
    var lat = 0.0
    var lng = 0.0
    var zoom = 0.0
    var bearing = 0.0
    var width = 0.0
    var height = 0.0
    var world = 0.0
    var cx = 0.0
    var cy = 0.0
    var cos = 1.0
    var sin = 0.0

    init() {}

    init(_ c: MLNStyleLayerDrawingContext) {
      lat = c.centerCoordinate.latitude
      lng = c.centerCoordinate.longitude
      zoom = c.zoomLevel
      bearing = c.direction
      width = Double(c.size.width)
      height = Double(c.size.height)
      world = 512 * Darwin.pow(2.0, zoom)
      cx = (lng + 180) / 360 * world
      cy = WindParticleLayer.mercatorY(lat) * world
      let r = -bearing * Double.pi / 180
      cos = Darwin.cos(r)
      sin = Darwin.sin(r)
    }

    static func == (a: Camera, b: Camera) -> Bool {
      a.lat == b.lat && a.lng == b.lng && a.zoom == b.zoom && a.bearing == b.bearing
        && a.width == b.width && a.height == b.height
    }

    /// Same scale, rotation and size: the two differ by a translation only.
    func sameFrame(_ o: Camera) -> Bool {
      abs(zoom - o.zoom) < 1e-6 && abs(bearing - o.bearing) < 1e-6
        && width == o.width && height == o.height
    }
  }

  private var cam = Camera()
  private var lastCam = Camera()
  /// The reference frame the simulation runs in.
  private var ref = Camera()
  /// The reference the trail texture being composited was drawn in — one
  /// deposit behind `ref`, see "Ordering across two queues".
  private var shownRef = Camera()
  private var haveRef = false
  private var haveShown = false
  private var stillTime = 0.0

  private var needClear = true
  private var velocityDirty = true
  private var shiftPending = false
  private var shift = SIMD2<Float>()

  // -------------------------------------------------------------------- clock

  private var stateBlock = 0
  private var blockFrame = 0.0
  private var lastFrameTime = 0.0
  private var depositTimer = 0.0
  /// Windy's global alpha: 0 when playback starts, up to 1 at `alphaRate`.
  private var alpha = 0.0
  private var frame: UInt64 = 0
  private var lastPerBlock = 0
  private var deposits = 0
  private var settles = 0

  // ------------------------------------------------------------------- public

  func setField(_ upload: WindFieldUpload) {
    pendingField = upload
    setNeedsDisplay()
  }

  /// Starts or stops the frame clock. A new playback starts from a fresh
  /// reference and fades in: the camera may have gone anywhere meanwhile.
  func setPlaying(_ value: Bool) {
    guard playing != value else { return }
    playing = value
    if value { restart = true }
    value ? startClock() : stopClock()
    setNeedsDisplay()
  }

  /// Asks for a map frame at up to 60 Hz — Windy's `requestAnimationFrame`.
  ///
  /// **Not from inside `draw`.** Calling `setNeedsDisplay()` there is a redraw
  /// that schedules a redraw: nothing throttles it, the render thread never
  /// yields, and the map — the whole app, since the platform view owns that
  /// thread — stops responding.
  ///
  /// Capped at 60 on ProMotion panels: the field advances at 60 Hz whatever
  /// the panel does, so asking for 120 would render the whole map twice per
  /// change.
  private func startClock() {
    guard clock == nil else { return }
    let link = CADisplayLink(target: self, selector: #selector(tick))
    if #available(iOS 15.0, *) {
      link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
    } else {
      link.preferredFramesPerSecond = 60
    }
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

    if device == nil, !makeDevice(mapView: mapView, encoder: encoder) { return }
    if let upload = pendingField {
      pendingField = nil
      uploadField(upload)
    }
    guard fieldReady, playing else { return }

    if restart {
      restart = false
      haveRef = false
      haveShown = false
      lastFrameTime = 0
      depositTimer = 0
      alpha = 0
    }
    cam = Camera(context)
    ensureTargets(scale: mapView.contentScaleFactor)
    guard trailTex.count == 2, velTex != nil else { return }

    let now = ProcessInfo.processInfo.systemUptime
    var elapsed = Self.step
    if lastFrameTime != 0 {
      elapsed = max(0, min(0.1, now - lastFrameTime))
    }
    lastFrameTime = now
    depositTimer += elapsed

    // Two rates, as on Android: the simulation deposits at 60 Hz, the
    // composite runs on every map frame.
    var advanced = false
    if depositTimer >= Self.step * 0.75 || !haveRef {
      let dt = min(depositTimer, 0.1)
      depositTimer = 0
      settleCamera(dt: dt)
      advanced = simulate(dt: dt)
    }

    // Reads the texture the deposit above did *not* write. The flip follows,
    // so the next frame reads what was just encoded.
    composite(encoder: encoder)
    if advanced {
      trailFront = 1 - trailFront
      shownRef = ref
      haveShown = true
    }

    frame &+= 1
    if frame == 1 || frame % 600 == 0 {
      NSLog(
        "WindParticleLayer: frame=%llu zoom=%.3f ctx=%.0fx%.0f scale=%.1f trail=%ldx%ld "
          + "perBlock=%ld total=%ld field=%ldx%ld deposit=%ld settle=%ld",
        frame, context.zoomLevel, context.size.width, context.size.height,
        mapView.contentScaleFactor, trailWidth, trailHeight, lastPerBlock,
        lastPerBlock * Self.blocks, fieldWidth, fieldHeight, deposits, settles)
      deposits = 0
      settles = 0
    }
  }

  /// Windy's `moveend` and `zoomend`, inferred from consecutive frames: a
  /// camera that has held still for `settleSeconds` is at rest, and if it is
  /// not where the reference is, the reference follows it. See the Android
  /// twin's `settleCamera` for why nothing happens while it moves.
  private func settleCamera(dt: Double) {
    if cam == lastCam {
      stillTime += dt
    } else {
      lastCam = cam
      stillTime = 0
    }
    if !haveRef {
      lastCam = cam
      ref = cam
      haveRef = true
      needClear = true
      velocityDirty = true
      shiftPending = false
    } else if stillTime >= Self.settleSeconds, cam != ref {
      settles += 1
      if cam.sameFrame(ref), !needClear {
        // A pan: whole trail texels, so the copy is exact rather than a blur.
        let o = refToClip(ref, 0, 0)
        shift = SIMD2(
          Float((Double(o.x + 1) * 0.5 * Double(trailWidth)).rounded() / Double(trailWidth)),
          Float((Double(o.y + 1) * 0.5 * Double(trailHeight)).rounded() / Double(trailHeight)))
        shiftPending = true
      } else {
        needClear = true
        shiftPending = false
      }
      ref = cam
      velocityDirty = true
    }
  }

  /// One deposit: project (if the reference moved) → advect, and fade → draw
  /// into the back trail texture. Returns whether a trail frame was encoded.
  private func simulate(dt: Double) -> Bool {
    guard let queue, let buffer = queue.makeCommandBuffer(),
          let projectPipeline, let updatePipeline, let stateShiftPipeline,
          let fadePipeline, let segmentPipeline, let nearestSampler,
          let linearSampler, let fieldSampler, let velTex, let fieldU, let fieldV,
          let vertexBuffer, let indexBuffer
    else { return false }

    let z = Self.windyZoom(ref.zoom)
    let amount = min(
      Self.amountMax,
      (ref.width * ref.height / (50 * pow(1.6, Double(z) - 2))).rounded())
    let relativeAmount = amount * Self.countMul / Double(Self.stateEdge * Self.stateEdge)
    let perBlock = max(
      1, min(Self.particlesPerBlock,
             Int((relativeAmount * Double(Self.particlesPerBlock)).rounded())))
    let columns = min(Self.stateEdge, Int(floor(relativeAmount * Double(Self.stateEdge) + 1)))
    let scale = Double(physicalSize.width) / ref.width
    let widthFactor = Float(max(1, Self.at(Self.lineWidth, z) * Self.particleWidth * scale))
    let timeScale = Self.speedPx * Self.at(Self.zoomToSpeed, z) * scale
    lastPerBlock = perBlock
    deposits += 1

    blockFrame += dt * 60
    var reseed = -1
    if blockFrame >= Double(Self.blockFrames) {
      blockFrame -= Double(Self.blockFrames)
      reseed = stateBlock
      stateBlock = (stateBlock + 1) % Self.blocks
    }

    var u = WindUniforms()

    // --- velocity, once per settle
    if velocityDirty {
      velocityDirty = false
      u.range = range
      u.grid = grid
      u.curve = Self.curve
      u.texel = SIMD4(0.5 / Float(fieldWidth), 0.5 / Float(fieldHeight), 0, 0)
      u.cam0 = SIMD4(Float(ref.cx), Float(ref.cy), Float(ref.width / 2), Float(ref.height / 2))
      u.cam1 = SIMD4(Float(ref.cos), Float(ref.sin), Float(ref.world), 0)
      if let enc = beginPass(buffer, velTex, load: .dontCare) {
        enc.setRenderPipelineState(projectPipeline)
        enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
        enc.setFragmentTexture(fieldU, index: 0)
        enc.setFragmentTexture(fieldV, index: 1)
        enc.setFragmentSamplerState(fieldSampler, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
      }
    }

    // --- particles follow a panned reference: into the back texture and back,
    //     so the next segment joins two positions in the same frame
    if shiftPending {
      for (src, dst, s) in [
        (stateFront, 1 - stateFront, shift), (1 - stateFront, stateFront, SIMD2<Float>()),
      ] {
        u.shift = SIMD4(s.x, s.y, 0, 0)
        if let enc = beginPass(buffer, stateTex[dst], load: .dontCare) {
          enc.setRenderPipelineState(stateShiftPipeline)
          enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
          enc.setFragmentTexture(stateTex[src], index: 0)
          enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
          enc.endEncoding()
        }
      }
    }

    // --- trails: last deposit's, faded (and shifted after a pan) into the back
    //     texture, then this deposit's segments over them. The reference blends
    //     the segments in and *then* fades; drawing them with a `fade` source
    //     factor over the already-faded trails is the same sum.
    let fade = Float(pow(Self.fade, dt * 60))
    let clear = needClear || !haveShown
    let back = trailTex[1 - trailFront]
    guard let enc = beginPass(buffer, back, load: clear ? .clear : .dontCare) else {
      buffer.commit()
      return false
    }
    if !clear {
      u.shift = SIMD4(shiftPending ? shift.x : 0, shiftPending ? shift.y : 0, fade, 0)
      enc.setRenderPipelineState(fadePipeline)
      enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
      enc.setFragmentTexture(trailTex[trailFront], index: 0)
      enc.setFragmentSamplerState(nearestSampler, index: 0)
      enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
    needClear = false
    shiftPending = false

    let aw = (widthFactor + 1) / Float(physicalSize.width)
    let ah = (widthFactor + 1) / Float(physicalSize.height)
    let len = max(1, widthFactor * 0.8)
    u.line = SIMD4(aw * 2 / 255, ah * 2 / 255, -aw, -ah)
    u.edge = SIMD4(0, 0, len * 2 / 255, -len)
    u.seg = SIMD4(len, 0, 0, 0)
    enc.setRenderPipelineState(segmentPipeline)
    enc.setBlendColor(red: fade, green: fade, blue: fade, alpha: fade)
    enc.setVertexBuffer(vertexBuffer, offset: 0, index: 1)
    enc.setVertexTexture(stateTex[stateFront], index: 0)
    enc.setVertexTexture(stateTex[1 - stateFront], index: 1)
    enc.setVertexSamplerState(nearestSampler, index: 0)
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
        type: .triangle, indexCount: perBlock * 6, indexType: .uint16,
        indexBuffer: indexBuffer, indexBufferOffset: 0)
      age += Self.blockFrames
    }
    enc.endEncoding()

    // --- advect the drawn columns into the back state texture
    let ox = Float(dt * timeScale / Double(physicalSize.width))
    let oy = Float(dt * timeScale / Double(physicalSize.height))
    u.disp = SIMD4(ox * Self.dispMul, oy * Self.dispMul, -ox * Self.dispAdd, -oy * Self.dispAdd)
    if let enc = beginPass(buffer, stateTex[1 - stateFront], load: .load) {
      enc.setViewport(MTLViewport(
        originX: 0, originY: 0, width: Double(columns), height: Double(Self.stateEdge),
        znear: 0, zfar: 1))
      enc.setRenderPipelineState(updatePipeline)
      enc.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
      enc.setFragmentTexture(stateTex[stateFront], index: 0)
      enc.setFragmentTexture(velTex, index: 1)
      enc.setFragmentSamplerState(linearSampler, index: 1)
      enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
      enc.endEncoding()
    }
    stateFront = 1 - stateFront
    if reseed >= 0 { reseedBlock(reseed, in: buffer) }
    buffer.commit()

    alpha = min(1, alpha + dt * Self.alphaRate)
    return true
  }

  private func beginPass(_ buffer: MTLCommandBuffer, _ texture: MTLTexture,
                         load: MTLLoadAction) -> MTLRenderCommandEncoder? {
    targetPass.colorAttachments[0].texture = texture
    targetPass.colorAttachments[0].loadAction = load
    targetPass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
    targetPass.colorAttachments[0].storeAction = .store
    return buffer.makeRenderCommandEncoder(descriptor: targetPass)
  }

  /// The one pass that lands in the map's own encoder: the trail texture as a
  /// quad wherever its reference viewport now is.
  private func composite(encoder: MTLRenderCommandEncoder) {
    guard haveShown, let compositePipeline, let linearSampler, trailTex.count == 2
    else { return }
    let o = refToClip(shownRef, 0, 0)
    let x = refToClip(shownRef, 1, 0)
    let y = refToClip(shownRef, 0, 1)
    var u = WindUniforms()
    u.affine0 = SIMD4(o.x, o.y, x.x - o.x, x.y - o.y)
    u.affine1 = SIMD4(y.x - o.x, y.y - o.y, Self.compositeGain * Float(alpha),
                      Self.compositeLift)
    encoder.setRenderPipelineState(compositePipeline)
    if let passThroughDepth { encoder.setDepthStencilState(passThroughDepth) }
    encoder.setVertexBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
    encoder.setFragmentBytes(&u, length: MemoryLayout<WindUniforms>.stride, index: 0)
    encoder.setFragmentTexture(trailTex[trailFront], index: 0)
    encoder.setFragmentSamplerState(linearSampler, index: 0)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
  }

  /// A point of `r`'s viewport (particle space: 0..1, y up) in this frame's
  /// clip space. The map has no pitch, so this is affine and three points place
  /// the quad. The world shift keeps a pan across the antimeridian from
  /// throwing the field a world away.
  private func refToClip(_ r: Camera, _ px: Double, _ py: Double) -> SIMD2<Float> {
    let s = cam.world / r.world
    let wrap = cam.world * ((r.cx * s - cam.cx) / cam.world).rounded()
    let dx = (px * 2 - 1) * r.width / 2
    let dy = (1 - py * 2) * r.height / 2
    let ex = (r.cx + dx * r.cos + dy * r.sin) * s - cam.cx - wrap
    let ey = (r.cy - dx * r.sin + dy * r.cos) * s - cam.cy
    return SIMD2(
      Float((ex * cam.cos - ey * cam.sin) / (cam.width / 2)),
      Float(-(ex * cam.sin + ey * cam.cos) / (cam.height / 2)))
  }

  fileprivate static func mercatorY(_ lat: Double) -> Double {
    let clamped = max(-85.051129, min(85.051129, lat))
    let phi = clamped * Double.pi / 180
    return (1 - log(tan(Double.pi / 4 + phi / 2)) / Double.pi) / 2
  }

  // ------------------------------------------------------------------ resources

  private func makeDevice(mapView: MLNMapView,
                         encoder: MTLRenderCommandEncoder) -> Bool {
    let dev = encoder.device
    guard let q = dev.makeCommandQueue() else { return false }
    let library: MTLLibrary
    do {
      library = try dev.makeLibrary(source: WindParticleShaders.source, options: nil)
    } catch {
      NSLog("WindParticleLayer: shader compile failed: \(error)")
      return false
    }
    guard
      let fullV = library.makeFunction(name: "fullVertex"),
      let projectF = library.makeFunction(name: "projectFragment"),
      let updateF = library.makeFunction(name: "updateFragment"),
      let stateShiftF = library.makeFunction(name: "stateShiftFragment"),
      let fadeF = library.makeFunction(name: "fadeFragment"),
      let segV = library.makeFunction(name: "segmentVertex"),
      let segF = library.makeFunction(name: "segmentFragment"),
      let compV = library.makeFunction(name: "compositeVertex"),
      let compF = library.makeFunction(name: "compositeFragment")
    else {
      NSLog("WindParticleLayer: a shader function is missing")
      return false
    }

    enum Blend { case none, premultiplied, premultipliedTimesConstant }

    // `depthStencil` is set only for the composite: the offscreen passes render
    // into this class's own colour-only textures, and declaring an attachment
    // they do not have would invalidate them the same way omitting it
    // invalidates the composite.
    func pipeline(_ v: MTLFunction, _ f: MTLFunction, _ format: MTLPixelFormat,
                  _ blend: Blend, depthStencil: MTLPixelFormat? = nil)
      -> MTLRenderPipelineState?
    {
      let d = MTLRenderPipelineDescriptor()
      d.vertexFunction = v
      d.fragmentFunction = f
      d.colorAttachments[0].pixelFormat = format
      if let depthStencil {
        d.depthAttachmentPixelFormat = depthStencil
        d.stencilAttachmentPixelFormat = depthStencil
      }
      if blend != .none {
        // Premultiplied: the generations overlap constantly, and straight
        // alpha would let the newest erase the older ones.
        let src: MTLBlendFactor = blend == .premultiplied ? .one : .blendAlpha
        d.colorAttachments[0].isBlendingEnabled = true
        d.colorAttachments[0].sourceRGBBlendFactor = src
        d.colorAttachments[0].sourceAlphaBlendFactor = src
        d.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        d.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
      }
      return try? dev.makeRenderPipelineState(descriptor: d)
    }

    projectPipeline = pipeline(fullV, projectF, .rgba8Unorm, .none)
    updatePipeline = pipeline(fullV, updateF, .rgba8Unorm, .none)
    stateShiftPipeline = pipeline(fullV, stateShiftF, .rgba8Unorm, .none)
    fadePipeline = pipeline(fullV, fadeF, .rgba8Unorm, .none)
    segmentPipeline = pipeline(segV, segF, .rgba8Unorm, .premultipliedTimesConstant)
    // The composite lands in the map's own render pass, so its pipeline has to
    // describe that pass exactly — `backendResource` carries the MTKView the
    // map draws into, and MapLibre's plugin-layer example pins the
    // depth/stencil pair it attaches.
    let resource = mapView.backendResource()
    let drawableFormat = resource.mtkView?.colorPixelFormat ?? .bgra8Unorm
    compositePipeline = pipeline(
      compV, compF, drawableFormat, .premultiplied, depthStencil: .depth32Float_stencil8)

    // And the encoder's depth/stencil *state* is whatever the layer drawn
    // before this one left behind. The map stencils its own geometry, so
    // inheriting that mask is how a correct quad ends up invisible.
    let ds = MTLDepthStencilDescriptor()
    ds.depthCompareFunction = .always
    ds.isDepthWriteEnabled = false
    passThroughDepth = dev.makeDepthStencilState(descriptor: ds)

    guard projectPipeline != nil, updatePipeline != nil, stateShiftPipeline != nil,
          fadePipeline != nil, segmentPipeline != nil, compositePipeline != nil
    else {
      NSLog(
        "WindParticleLayer: pipeline creation failed (drawable=%ld)",
        drawableFormat.rawValue)
      return false
    }

    func sampler(_ filter: MTLSamplerMinMagFilter, repeatS: Bool) -> MTLSamplerState? {
      let d = MTLSamplerDescriptor()
      d.minFilter = filter
      d.magFilter = filter
      d.sAddressMode = repeatS ? .repeat : .clampToEdge
      d.tAddressMode = .clampToEdge
      return dev.makeSamplerState(descriptor: d)
    }
    nearestSampler = sampler(.nearest, repeatS: false)
    linearSampler = sampler(.linear, repeatS: false)
    // Longitude is cyclic; latitude is not. Wrapping the second walks a
    // particle off the pole onto the other pole.
    fieldSampler = sampler(.linear, repeatS: true)

    guard makeStateTextures(dev), makeGeometry(dev) else { return false }
    device = dev
    queue = q
    return true
  }

  private func makeStateTextures(_ dev: MTLDevice) -> Bool {
    // Eight bits a channel, matching Android: `ba + rg / 255.5` inverts
    // `fract(pos * 255 + 0.25/255)` only when the values have actually been
    // rounded to bytes. Stored as float the pair stops round-tripping and
    // every particle drifts a little each frame.
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

  /// Rebirth, the reference's way: one block's rows of both state textures
  /// overwritten with uniform noise. Both, so the first segment of the new life
  /// has zero length and is culled instead of joining it to the last one.
  ///
  /// Encoded as a blit into the frame's own command buffer rather than written
  /// straight to the texture, so it is ordered after this frame's update.
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
    // One block's quads, reused for all sixteen: (texel x, row, side, end).
    // Column-major, the reference's order, so a partial draw is whole columns
    // and only those columns need advecting.
    var verts = [UInt8]()
    var inds = [UInt16]()
    verts.reserveCapacity(Self.particlesPerBlock * 16)
    inds.reserveCapacity(Self.particlesPerBlock * 6)
    let corners: [(UInt8, UInt8)] = [(0, 0), (255, 0), (255, 255), (0, 255)]
    for i in 0..<Self.particlesPerBlock {
      let x = UInt8(i / Self.blockRows)
      let y = UInt8(i % Self.blockRows)
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

  /// Trails at 0.8× the screen on dense displays, as the reference keeps them;
  /// the velocity texture at one texel per point, the reference's tile mosaic
  /// resolution. A new size is a new reference frame.
  private func ensureTargets(scale: CGFloat) {
    guard let device else { return }
    let size = CGSize(width: CGFloat(cam.width) * scale, height: CGFloat(cam.height) * scale)
    if size == physicalSize, trailTex.count == 2 { return }

    let ts = scale > Self.trailScaleDpr ? Self.trailScale : 1
    let tw = max(1, min(Self.trailMax, Int((size.width * ts).rounded())))
    let th = max(1, min(Self.trailMax, Int((size.height * ts).rounded())))
    let vw = max(1, min(Self.trailMax, Int(cam.width.rounded())))
    let vh = max(1, min(Self.trailMax, Int(cam.height.rounded())))

    func target(_ w: Int, _ h: Int) -> MTLTexture? {
      let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
      d.usage = [.shaderRead, .renderTarget]
      d.storageMode = .private
      return device.makeTexture(descriptor: d)
    }
    trailTex = []
    for _ in 0..<2 {
      guard let t = target(tw, th) else { return }
      trailTex.append(t)
    }
    velTex = target(vw, vh)
    trailWidth = tw
    trailHeight = th
    physicalSize = size
    haveRef = false
    haveShown = false
  }

  /// The two quantised planes as two R8 textures, straight out of the WND1
  /// body: no interleaving pass over a million cells per forecast frame.
  private func uploadField(_ upload: WindFieldUpload) {
    guard let device else { return }
    let n = upload.width * upload.height
    guard upload.width > 0, upload.height > 0,
          upload.bytes.count >= upload.planeOffset + n * 2
    else {
      NSLog("WindParticleLayer: wind field truncated")
      return
    }
    if fieldU?.width != upload.width || fieldU?.height != upload.height {
      let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm, width: upload.width, height: upload.height,
        mipmapped: false)
      d.usage = [.shaderRead]
      d.storageMode = .shared
      fieldU = device.makeTexture(descriptor: d)
      fieldV = device.makeTexture(descriptor: d)
    }
    guard let fieldU, let fieldV else { return }
    upload.bytes.withUnsafeBytes { raw in
      guard let base = raw.baseAddress?.advanced(by: upload.planeOffset) else { return }
      let region = MTLRegionMake2D(0, 0, upload.width, upload.height)
      fieldU.replace(region: region, mipmapLevel: 0, withBytes: base,
                     bytesPerRow: upload.width)
      fieldV.replace(region: region, mipmapLevel: 0, withBytes: base.advanced(by: n),
                     bytesPerRow: upload.width)
    }
    fieldWidth = upload.width
    fieldHeight = upload.height
    range = SIMD4(upload.uMin, upload.vMin, upload.uMax, upload.vMax)
    grid = SIMD4(upload.lon0, upload.lat0, upload.dLat * Float(upload.height), 0)
    fieldReady = true
    // Resampled under the same reference. The trails carry over, as the
    // reference's do: new data is the same view a forecast step later.
    velocityDirty = true
  }
}
