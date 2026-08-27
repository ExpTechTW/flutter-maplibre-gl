// A MapLibre custom layer host, so the wind particles are drawn by the map's
// own renderer instead of by a callback this plugin re-arms behind its back.
//
// ## Why this exists
//
// MapLibre Android exposes no Java API for drawing into the map's GL pass. The
// plugin previously hooked `MapLibreSurfaceView.surfaceRedrawNeededAsync`,
// which is not that: it is `SurfaceHolder.Callback2`'s "your surface needs
// redrawing" entry point, and calling it *requests a redraw* as well as
// registering a one-shot callback. Two consequences, both measured on a
// Pixel 9:
//
//   * The layer became the map's frame driver and drove it past the panel —
//     169 fps against 120 Hz, so ~49 frames a second were rendered and
//     discarded at irregular moments.
//   * Callback spacing ranged from 0.46 ms to 50 ms against a 16.6 ms mean.
//     Frames the map started on its own, after our one-shot was consumed and
//     before the next was registered, carried no particles at all.
//
// `CustomLayerHost` is the supported path and the one iOS already uses —
// `MLNCustomStyleLayer` is the same core mechanism with an Objective-C face.
// The renderer calls `render` once per map frame, inside its own pass, with
// the GL context current. There is nothing to re-arm and no second driver.
//
// ## ABI, and why linking is not needed
//
// MapLibre calls us; we call nothing of MapLibre's. So this translation unit
// links against JNI and liblog only — never libmaplibre.so — and the entire
// contract is the vtable layout of the abstract class below. It is transcribed
// from `include/mbgl/style/layers/custom_layer.hpp` at tag `android-v13.3.0`,
// which is the version `build.gradle` pins.
//
// **Two things must be re-checked on every MapLibre bump**: the order of the
// virtual functions, and the fields of `CustomLayerRenderParameters`. Upstream
// has moved this before — the namespace is `mln::` on newer branches — and a
// mismatch is not a link error. It is a call through the wrong vtable slot.
#include <jni.h>

#include <array>
#include <cstddef>

namespace mbgl {
namespace style {

// Transcribed, not included: the AAR ships no headers.
struct CustomLayerRenderParameters {
    double width;
    double height;
    double latitude;
    double longitude;
    double zoom;
    double bearing;
    double pitch;
    double fieldOfView;
    std::array<double, 16> projectionMatrix;
};

class CustomLayerHost {
public:
    virtual ~CustomLayerHost() = default;
    virtual void initialize() = 0;
    virtual void render(const CustomLayerRenderParameters&) = 0;
    virtual void contextLost() = 0;
    virtual void deinitialize() = 0;
};

}  // namespace style
}  // namespace mbgl

namespace {

JavaVM* g_vm = nullptr;

/// The render thread is one MapLibre created, and it is a Java thread, so it is
/// normally already attached. `initialize` runs there first and attaches if it
/// is not — permanently, because detaching between frames would cost a JNI
/// attach on every single one.
JNIEnv* renderEnv() {
    if (g_vm == nullptr) return nullptr;
    JNIEnv* env = nullptr;
    jint status = g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return nullptr;
    } else if (status != JNI_OK) {
        return nullptr;
    }
    return env;
}

/// Forwards the map's own render callbacks to the Java layer, which owns every
/// GL object and all of the drawing. Only the timing hook moved down here; the
/// renderer above it is unchanged.
class WindParticleHost final : public mbgl::style::CustomLayerHost {
public:
    WindParticleHost(JNIEnv* env, jobject layer) {
        peer_ = env->NewGlobalRef(layer);
        jclass cls = env->GetObjectClass(layer);
        onInitialize_ = env->GetMethodID(cls, "onNativeInitialize", "()V");
        onRender_ = env->GetMethodID(cls, "onNativeRender", "(DDDDDD)V");
        onContextLost_ = env->GetMethodID(cls, "onNativeContextLost", "()V");
        onDeinitialize_ = env->GetMethodID(cls, "onNativeDeinitialize", "()V");
        env->DeleteLocalRef(cls);
    }

    ~WindParticleHost() override {
        JNIEnv* env = renderEnv();
        if (env != nullptr && peer_ != nullptr) {
            env->DeleteGlobalRef(peer_);
        }
        peer_ = nullptr;
    }

    void initialize() override { callVoid(onInitialize_); }

    void render(const mbgl::style::CustomLayerRenderParameters& p) override {
        JNIEnv* env = renderEnv();
        if (env == nullptr || peer_ == nullptr || onRender_ == nullptr) return;
        // The camera as the renderer has it for *this* frame — the thing the
        // old path could only approximate with a snapshot pushed from the
        // platform thread, which was stale whenever the map was moving.
        env->CallVoidMethod(peer_, onRender_, p.width, p.height, p.latitude,
                            p.longitude, p.zoom, p.bearing);
        clearPending(env);
    }

    void contextLost() override { callVoid(onContextLost_); }

    void deinitialize() override { callVoid(onDeinitialize_); }

private:
    void callVoid(jmethodID method) {
        JNIEnv* env = renderEnv();
        if (env == nullptr || peer_ == nullptr || method == nullptr) return;
        env->CallVoidMethod(peer_, method);
        clearPending(env);
    }

    /// An exception left pending would surface at the next JNI call, somewhere
    /// unrelated and impossible to attribute. The Java side already logs and
    /// swallows its own failures; this is the backstop for anything that got
    /// past it.
    static void clearPending(JNIEnv* env) {
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
    }

    jobject peer_ = nullptr;
    jmethodID onInitialize_ = nullptr;
    jmethodID onRender_ = nullptr;
    jmethodID onContextLost_ = nullptr;
    jmethodID onDeinitialize_ = nullptr;
};

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

/// Returns a `CustomLayerHost*` as a jlong, which is exactly what
/// `org.maplibre.android.style.layers.CustomLayer(String, long)` wants.
/// Ownership passes to MapLibre, which deletes it through the base class —
/// hence the virtual destructor.
JNIEXPORT jlong JNICALL
Java_org_maplibre_maplibregl_WindParticleLayer_nativeCreateHost(
    JNIEnv* env, jclass, jobject layer) {
    return reinterpret_cast<jlong>(new WindParticleHost(env, layer));
}

}  // extern "C"
