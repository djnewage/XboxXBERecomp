// OpenXR on the Direct3D 12 renderer: see vr.h. The loader (openxr_loader.dll beside the exe) is loaded by hand, so a PC
// without it or without a runtime plays as before.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define XR_NO_PROTOTYPES
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_D3D12
#include "../../thirdparty/openxr/include/openxr/openxr.h"
#include "../../thirdparty/openxr/include/openxr/openxr_platform.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include "app/app.h"
#include "core/log.h"
#include "gpu/ext.h"
#include "gpu/shaders.h"
#include "gpu/state.h"
#include "vr/vr.h"

namespace vr {
namespace {
using Microsoft::WRL::ComPtr;

constexpr float kPi = 3.14159265358979f;
constexpr float kScreenDistance = 2.5f, kScreenWidth = 3.2f;  // metres: the screen menus and movies show on
constexpr uint64_t kRetryMs = 3000;
constexpr uint32_t kMostImages = 16;

#define XR_FUNCTIONS(X)                \
    X(xrDestroyInstance)               \
    X(xrGetInstanceProperties)         \
    X(xrPollEvent)                     \
    X(xrGetSystem)                     \
    X(xrGetSystemProperties)           \
    X(xrCreateSession)                 \
    X(xrDestroySession)                \
    X(xrBeginSession)                  \
    X(xrEndSession)                    \
    X(xrCreateReferenceSpace)          \
    X(xrDestroySpace)                  \
    X(xrLocateSpace)                   \
    X(xrEnumerateSwapchainFormats)     \
    X(xrCreateSwapchain)               \
    X(xrDestroySwapchain)              \
    X(xrEnumerateSwapchainImages)      \
    X(xrAcquireSwapchainImage)         \
    X(xrWaitSwapchainImage)            \
    X(xrReleaseSwapchainImage)         \
    X(xrWaitFrame)                     \
    X(xrBeginFrame)                    \
    X(xrEndFrame)                      \
    X(xrGetD3D12GraphicsRequirementsKHR)
#define XR_DECLARE(name) PFN_##name name = nullptr;
XR_FUNCTIONS(XR_DECLARE)
#undef XR_DECLARE
PFN_xrGetInstanceProcAddr xrGetInstanceProcAddr = nullptr;

// what the title drew a frame with
struct Pose {
    bool world = false;  // a view of the world at these angles, else a picture for the screen
    float yaw = 0, pitch = 0;  // radians
    XrVector3f position{0, 0, 0};
};

std::atomic<bool> g_failed{false}, g_running{false};

// the head as last located, for the title (g_lock)
std::mutex g_lock;
bool g_head_valid = false;
Pose g_head;
Pose g_title;              // what the title last asked for: its frames are drawn with it until it says otherwise
uint64_t g_title_at = 0;    // when (GetTickCount64): a title that stops saying has no such view any more
std::deque<Pose> g_poses;  // one per title frame presented, in order

// --- the renderer's thread from here ---
ID3D12Device* g_device = nullptr;
ID3D12CommandQueue* g_queue = nullptr;
XrInstance g_instance = XR_NULL_HANDLE;
XrSystemId g_system = XR_NULL_SYSTEM_ID;
XrSession g_session = XR_NULL_HANDLE;
XrSpace g_local = XR_NULL_HANDLE, g_view = XR_NULL_HANDLE;
uint64_t g_retry_at = 0;
bool g_begun = false, g_should_render = false, g_drew = false;
XrTime g_time = 0;
Pose g_shown;  // the pose of the frame being shown

XrSwapchain g_swap = XR_NULL_HANDLE;
uint32_t g_swap_w = 0, g_swap_h = 0;
DXGI_FORMAT g_swap_format = DXGI_FORMAT_UNKNOWN;
ComPtr<ID3D12DescriptorHeap> g_rtv_heap;
uint32_t g_rtv_inc = 0;
std::vector<XrSwapchainImageD3D12KHR> g_images;

// the scene's projection this frame: the perspective projection most vertices were drawn with
struct Seen {
    float x, y;  // the tangents of the half angles
    uint64_t vertices;
};
std::vector<Seen> g_seen;
float g_tan_x = 0, g_tan_y = 0;

const char kCopyHlsl[] = R"(
cbuffer B : register(b1) { float4 opts; float4 unused; };
Texture2D img : register(t0);
SamplerState ls : register(s0);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = t;
    return o;
}
// the console's picture is gamma encoded; an sRGB target wants it linear
float4 psvr(O i) : SV_Target {
    float3 c = img.SampleLevel(ls, i.uv, 0).rgb;
    if (opts.x > 0) c = c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
    return float4(c, 1);
}
)";

bool Srgb(DXGI_FORMAT f) { return f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB; }

XrQuaternionf Mul(const XrQuaternionf& a, const XrQuaternionf& b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
// yaw about +y (left positive), then pitch about +x (up positive): OpenXR looks along -z
XrQuaternionf Level(float yaw, float pitch) {
    XrQuaternionf y{0, std::sin(yaw / 2), 0, std::cos(yaw / 2)}, x{std::sin(pitch / 2), 0, 0, std::cos(pitch / 2)};
    return Mul(y, x);
}
void Angles(const XrQuaternionf& q, float& yaw, float& pitch) {
    // where -z points
    float fx = -2 * (q.x * q.z + q.w * q.y), fy = -2 * (q.y * q.z - q.w * q.x), fz = -(1 - 2 * (q.x * q.x + q.y * q.y));
    yaw = std::atan2(-fx, -fz);
    pitch = std::asin(std::clamp(fy, -1.0f, 1.0f));
}

void DropSwapchain() {
    if (g_swap != XR_NULL_HANDLE) xrDestroySwapchain(g_swap);
    g_swap = XR_NULL_HANDLE;
    g_images.clear();
    g_swap_w = g_swap_h = 0;
}

void DropSession(const char* why) {
    LOGR("vr: the session ends (%s)", why);
    DropSwapchain();
    if (g_view != XR_NULL_HANDLE) xrDestroySpace(g_view);
    if (g_local != XR_NULL_HANDLE) xrDestroySpace(g_local);
    if (g_session != XR_NULL_HANDLE) xrDestroySession(g_session);
    g_view = g_local = XR_NULL_HANDLE;
    g_session = XR_NULL_HANDLE;
    g_system = XR_NULL_SYSTEM_ID;
    g_begun = g_drew = false;
    g_running = false;
    {
        std::lock_guard lock(g_lock);
        g_head_valid = false;
        g_poses.clear();
    }
    g_retry_at = GetTickCount64() + kRetryMs;
}

void Fail(const char* why) {
    LOGR("vr: off for this run (%s)", why);
    g_failed = true;
}

bool CreateInstance() {
    HMODULE loader = LoadLibraryW((app::ExeDir() / L"openxr_loader.dll").c_str());
    if (!loader) loader = LoadLibraryW(L"openxr_loader.dll");
    if (!loader) return Fail("openxr_loader.dll is not beside the game"), false;
    xrGetInstanceProcAddr = reinterpret_cast<PFN_xrGetInstanceProcAddr>(GetProcAddress(loader, "xrGetInstanceProcAddr"));
    PFN_xrCreateInstance create = nullptr;
    if (xrGetInstanceProcAddr) xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrCreateInstance", reinterpret_cast<PFN_xrVoidFunction*>(&create));
    if (!create) return Fail("the OpenXR loader has no xrCreateInstance"), false;

    const char* extension = XR_KHR_D3D12_ENABLE_EXTENSION_NAME;
    XrInstanceCreateInfo info{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strncpy(info.applicationInfo.applicationName, app::kGame.log ? app::kGame.log : "xbe", XR_MAX_APPLICATION_NAME_SIZE - 1);
    info.applicationInfo.applicationVersion = 1;
    std::strncpy(info.applicationInfo.engineName, "XboxXBERecomp", XR_MAX_ENGINE_NAME_SIZE - 1);
    info.applicationInfo.apiVersion = XR_API_VERSION_1_0;
    info.enabledExtensionCount = 1;
    info.enabledExtensionNames = &extension;
    XrResult r = create(&info, &g_instance);
    if (XR_FAILED(r)) {
        LOGR("vr: xrCreateInstance failed (%d): is an OpenXR runtime (SteamVR) installed and set as the active one?", int(r));
        return Fail("no OpenXR runtime"), false;
    }
    bool all = true;
#define XR_RESOLVE(name)                                                                                   \
    xrGetInstanceProcAddr(g_instance, #name, reinterpret_cast<PFN_xrVoidFunction*>(&name));                \
    if (!name) LOGR("vr: the runtime has no " #name), all = false;
    XR_FUNCTIONS(XR_RESOLVE)
#undef XR_RESOLVE
    if (!all) return Fail("the runtime lacks a function"), false;
    XrInstanceProperties props{XR_TYPE_INSTANCE_PROPERTIES};
    if (XR_SUCCEEDED(xrGetInstanceProperties(g_instance, &props)))
        LOGR("vr: OpenXR runtime %s %u.%u.%u", props.runtimeName, unsigned(XR_VERSION_MAJOR(props.runtimeVersion)),
             unsigned(XR_VERSION_MINOR(props.runtimeVersion)), unsigned(XR_VERSION_PATCH(props.runtimeVersion)));
    return true;
}

// a headset may be plugged in or switched on later: tried again every few seconds
bool CreateSession() {
    XrSystemGetInfo get{XR_TYPE_SYSTEM_GET_INFO};
    get.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XrResult r = xrGetSystem(g_instance, &get, &g_system);
    if (XR_FAILED(r)) {
        static bool said = false;
        if (!said) LOGR("vr: no headset yet (%d), looking again every %u s", int(r), unsigned(kRetryMs / 1000));
        said = true;
        return false;
    }
    XrSystemProperties sys{XR_TYPE_SYSTEM_PROPERTIES};
    if (XR_SUCCEEDED(xrGetSystemProperties(g_instance, g_system, &sys))) LOGR("vr: headset %s", sys.systemName);

    XrGraphicsRequirementsD3D12KHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_D3D12_KHR};
    if (XR_FAILED(r = xrGetD3D12GraphicsRequirementsKHR(g_instance, g_system, &req))) {
        LOGR("vr: xrGetD3D12GraphicsRequirementsKHR failed (%d)", int(r));
        return false;
    }
    LUID ours = g_device->GetAdapterLuid();
    if (std::memcmp(&ours, &req.adapterLuid, sizeof(LUID)) != 0)
        LOGR("vr: the headset is on another graphics adapter than the game's: the session may fail");

    XrGraphicsBindingD3D12KHR binding{XR_TYPE_GRAPHICS_BINDING_D3D12_KHR};
    binding.device = g_device;
    binding.queue = g_queue;
    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
    sci.next = &binding;
    sci.systemId = g_system;
    if (XR_FAILED(r = xrCreateSession(g_instance, &sci, &g_session))) {
        LOGR("vr: xrCreateSession failed (%d)", int(r));
        g_session = XR_NULL_HANDLE;
        return false;
    }
    XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    space.poseInReferenceSpace.orientation = {0, 0, 0, 1};
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    XrResult a = xrCreateReferenceSpace(g_session, &space, &g_local);
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    XrResult b = xrCreateReferenceSpace(g_session, &space, &g_view);
    if (XR_FAILED(a) || XR_FAILED(b)) {
        LOGR("vr: xrCreateReferenceSpace failed (%d, %d)", int(a), int(b));
        DropSession("no reference space");
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = kMostImages;
    if (!g_rtv_heap) g_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_rtv_heap));
    g_rtv_inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    LOGR("vr: session made on the Direct3D 12 device");
    return true;
}

void PollEvents() {
    XrEventDataBuffer e{XR_TYPE_EVENT_DATA_BUFFER};
    while (g_session != XR_NULL_HANDLE && xrPollEvent(g_instance, &e) == XR_SUCCESS) {
        if (e.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            auto& s = *reinterpret_cast<XrEventDataSessionStateChanged*>(&e);
            LOGR("vr: session state %d", int(s.state));
            if (s.state == XR_SESSION_STATE_READY) {
                XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
                begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                XrResult r = xrBeginSession(g_session, &begin);
                if (XR_SUCCEEDED(r)) g_running = true;
                else LOGR("vr: xrBeginSession failed (%d)", int(r));
            } else if (s.state == XR_SESSION_STATE_STOPPING) {
                xrEndSession(g_session);
                g_running = false;
                g_begun = false;
                std::lock_guard lock(g_lock);
                g_head_valid = false;
            } else if (s.state == XR_SESSION_STATE_EXITING || s.state == XR_SESSION_STATE_LOSS_PENDING) {
                DropSession("the runtime asked");
            }
        } else if (e.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
            DropSession("the runtime is going");
            Fail("the OpenXR runtime went away");
        }
        e = {XR_TYPE_EVENT_DATA_BUFFER};
    }
}

bool EnsureSwapchain(uint32_t w, uint32_t h) {
    if (g_swap != XR_NULL_HANDLE && w == g_swap_w && h == g_swap_h) return true;
    DropSwapchain();
    uint32_t n = 0;
    xrEnumerateSwapchainFormats(g_session, 0, &n, nullptr);
    std::vector<int64_t> formats(n);
    if (n) xrEnumerateSwapchainFormats(g_session, n, &n, formats.data());
    g_swap_format = DXGI_FORMAT_UNKNOWN;
    for (DXGI_FORMAT want : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM,
                             DXGI_FORMAT_B8G8R8A8_UNORM})
        if (g_swap_format == DXGI_FORMAT_UNKNOWN && std::find(formats.begin(), formats.end(), int64_t(want)) != formats.end())
            g_swap_format = want;
    if (g_swap_format == DXGI_FORMAT_UNKNOWN) return LOGR("vr: the runtime offers no 8 bit colour format"), false;
    XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    ci.format = g_swap_format;
    ci.sampleCount = 1;
    ci.width = w;
    ci.height = h;
    ci.faceCount = ci.arraySize = ci.mipCount = 1;
    XrResult r = xrCreateSwapchain(g_session, &ci, &g_swap);
    if (XR_FAILED(r)) {
        LOGR("vr: xrCreateSwapchain %ux%u failed (%d)", w, h, int(r));
        g_swap = XR_NULL_HANDLE;
        return false;
    }
    xrEnumerateSwapchainImages(g_swap, 0, &n, nullptr);
    n = std::min(n, kMostImages);
    g_images.assign(n, {XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR});
    xrEnumerateSwapchainImages(g_swap, n, &n, reinterpret_cast<XrSwapchainImageBaseHeader*>(g_images.data()));
    for (uint32_t i = 0; i < n; i++) {
        D3D12_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = g_swap_format;
        rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        auto h0 = g_rtv_heap->GetCPUDescriptorHandleForHeapStart();
        h0.ptr += size_t(i) * g_rtv_inc;
        g_device->CreateRenderTargetView(g_images[i].texture, &rv, h0);
    }
    g_swap_w = w, g_swap_h = h;
    LOGR("vr: swap chain %ux%u, format %d, %u images", w, h, int(g_swap_format), n);
    return true;
}

// the head now, for the frame the title builds next
void LocateHead() {
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    bool ok = XR_SUCCEEDED(xrLocateSpace(g_view, g_local, g_time, &loc)) &&
              (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) && (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT);
    std::lock_guard lock(g_lock);
    g_head_valid = ok;
    if (!ok) return;
    g_head.world = true;
    Angles(loc.pose.orientation, g_head.yaw, g_head.pitch);
    g_head.position = loc.pose.position;
}
}  // namespace

bool Enabled() { return app::Get().vr && !g_failed; }
bool Running() { return g_running; }

bool Head(float& yaw, float& pitch) {
    std::lock_guard lock(g_lock);
    if (!g_running || !g_head_valid) return false;
    g_title = g_head;
    g_title_at = GetTickCount64();
    yaw = g_head.yaw * 180 / kPi;
    pitch = g_head.pitch * 180 / kPi;
    return true;
}

void Flat() {
    std::lock_guard lock(g_lock);
    g_title = {};
}

void TitlePresented() {
    if (!g_running) return;
    std::lock_guard lock(g_lock);
    if (GetTickCount64() - g_title_at > 250) g_title = {};
    if (g_poses.size() < 8) g_poses.push_back(g_title);
}

void Start(ID3D12Device* device, ID3D12CommandQueue* queue) {
    g_device = device;
    g_queue = queue;
    if (!Enabled()) return;
    LOGR("vr: on (the vr setting)");
    if (!CreateInstance()) return;
    CreateSession();
    g_retry_at = GetTickCount64() + kRetryMs;
}

// what the frame was drawn with, logged now and then until the projection is understood
struct Diag {
    uint64_t fixed = 0, pre = 0, program = 0, no_matrix = 0, world = 0, screen = 0, frames = 0;
    uint64_t most_fixed = 0, most_program = 0;
    float proj[16] = {}, comp[16] = {};
    int k = -1;
} g_diag;

void NoteDraw(const gpu::State& s, uint32_t count) {
    if (!g_running) return;
    if (s.vs_handle & 1) {
        int k = s.vs ? gpu::MatrixProgram(s.vs->program) : -1;
        if (k < 0 || k + 3 >= 192) g_diag.no_matrix += count;
        else {
            g_diag.program += count;
            if (count > g_diag.most_program) {
                g_diag.most_program = count, g_diag.k = k;
                std::memcpy(g_diag.comp, s.vsc[size_t(k)].data(), 64);
            }
        }
    } else if ((s.vs_handle & 0xE) == 0x4) {
        g_diag.pre += count;
    } else {
        g_diag.fixed += count;
        if (count > g_diag.most_fixed) g_diag.most_fixed = count, std::memcpy(g_diag.proj, s.transforms[3].data(), 64);
    }
    // the fixed function with a perspective projection (D3DTS_PROJECTION, kept at 3): w is the view's z
    if ((s.vs_handle & 1) || (s.vs_handle & 0xE) == 0x4) return;
    const float* m = s.transforms[3].data();
    if (!(m[0] > 0.05f) || !(m[5] > 0.05f) || std::fabs(std::fabs(m[11]) - 1) > 1e-3f || std::fabs(m[15]) > 1e-3f) return;
    float x = 1 / m[0], y = 1 / m[5];
    for (Seen& seen : g_seen)
        if (std::fabs(seen.x - x) < 1e-4f && std::fabs(seen.y - y) < 1e-4f) {
            seen.vertices += count;
            return;
        }
    if (g_seen.size() < 16) g_seen.push_back({x, y, count});
}

void Frame(gpu::Context& ctx, ID3D12Resource* frame, int format, uint32_t width, uint32_t height) {
    // the frame's projection, whether or not it is shown
    const Seen* most = nullptr;
    for (const Seen& seen : g_seen)
        if (!most || seen.vertices > most->vertices) most = &seen;
    if (most && (std::fabs(most->x - g_tan_x) > 1e-3f || std::fabs(most->y - g_tan_y) > 1e-3f)) {
        g_tan_x = most->x, g_tan_y = most->y;
        LOGR("vr: the scene's field of view %.1f by %.1f degrees", 2 * std::atan(g_tan_x) * 180 / kPi, 2 * std::atan(g_tan_y) * 180 / kPi);
    }
    g_seen.clear();
    {
        std::lock_guard lock(g_lock);
        if (!g_poses.empty()) {
            g_shown = g_poses.front();
            g_poses.pop_front();
        }
    }
    (g_shown.world ? g_diag.world : g_diag.screen)++;
    if (++g_diag.frames >= 400) {
        const Diag& d = g_diag;
        LOGR("vr: 400 frames, %llu world %llu screen; vertices fixed %llu, pretransformed %llu, programs %llu (c%d), programs without a matrix %llu",
             d.world, d.screen, d.fixed, d.pre, d.program, d.k, d.no_matrix);
        if (d.most_fixed)
            LOGR("vr:  projection %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f", d.proj[0], d.proj[1], d.proj[2], d.proj[3],
                 d.proj[4], d.proj[5], d.proj[6], d.proj[7], d.proj[8], d.proj[9], d.proj[10], d.proj[11], d.proj[12], d.proj[13], d.proj[14], d.proj[15]);
        if (d.most_program)
            LOGR("vr:  program matrix %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f", d.comp[0], d.comp[1], d.comp[2], d.comp[3],
                 d.comp[4], d.comp[5], d.comp[6], d.comp[7], d.comp[8], d.comp[9], d.comp[10], d.comp[11], d.comp[12], d.comp[13], d.comp[14], d.comp[15]);
        g_diag = {};
    }
    g_drew = false;
    if (!g_begun || !g_should_render || !frame || !EnsureSwapchain(width, height)) return;
    uint32_t index = 0;
    XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(g_swap, &acquire, &index))) return;
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    if (XR_FAILED(xrWaitSwapchainImage(g_swap, &wait)) || index >= g_images.size()) {
        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(g_swap, &release);
        return;
    }
    auto rtv = g_rtv_heap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += size_t(index) * g_rtv_inc;
    ID3D12Resource* const src[3] = {frame, nullptr, nullptr};
    const DXGI_FORMAT fmt[3] = {DXGI_FORMAT(format), DXGI_FORMAT(format), DXGI_FORMAT(format)};
    const float p[8] = {Srgb(g_swap_format) ? 1.0f : 0.0f};
    ctx.Pass(kCopyHlsl, "psvr", g_swap_format, rtv, width, height, src, fmt, p, -1);
    ctx.Unbind();
    g_drew = true;
}

void Submit() {
    if (!g_begun) return;
    g_begun = false;
    XrCompositionLayerProjectionView views[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
    XrCompositionLayerProjection world{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerQuad screen{XR_TYPE_COMPOSITION_LAYER_QUAD};
    const XrCompositionLayerBaseHeader* layers[1] = {};
    uint32_t count = 0;
    if (g_drew) {
        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(g_swap, &release);
        XrSwapchainSubImage image{g_swap, {{0, 0}, {int32_t(g_swap_w), int32_t(g_swap_h)}}, 0};
        float aspect = float(g_swap_w) / float(g_swap_h);
        if (g_shown.world) {
            // one picture for both eyes, drawn level from the head's place: no depth yet
            float ty = g_tan_y > 0 ? g_tan_y : std::tan(30 * kPi / 180), tx = g_tan_x > 0 ? g_tan_x : ty * aspect;
            for (auto& v : views) {
                v.pose.orientation = Level(g_shown.yaw, g_shown.pitch);
                v.pose.position = g_shown.position;
                v.fov = {-std::atan(tx), std::atan(tx), std::atan(ty), -std::atan(ty)};
                v.subImage = image;
            }
            world.space = g_local;
            world.viewCount = 2;
            world.views = views;
            layers[count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&world);
        } else {
            screen.space = g_local;
            screen.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            screen.subImage = image;
            screen.pose.orientation = {0, 0, 0, 1};
            screen.pose.position = {0, 0, -kScreenDistance};
            screen.size = {kScreenWidth, kScreenWidth / aspect};
            layers[count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&screen);
        }
    }
    XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
    end.displayTime = g_time;
    end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    end.layerCount = count;
    end.layers = layers;
    XrResult r = xrEndFrame(g_session, &end);
    if (XR_FAILED(r)) LOG_FIRST(8, "vr: xrEndFrame failed (%d)", int(r));
    g_drew = false;
}

void Next() {
    if (!Enabled() || g_instance == XR_NULL_HANDLE) return;
    if (g_session == XR_NULL_HANDLE) {
        if (GetTickCount64() < g_retry_at) return;
        g_retry_at = GetTickCount64() + kRetryMs;
        if (!CreateSession()) return;
    }
    PollEvents();
    if (!g_running || g_session == XR_NULL_HANDLE) return;
    XrFrameState state{XR_TYPE_FRAME_STATE};
    XrResult r = xrWaitFrame(g_session, nullptr, &state);
    if (XR_SUCCEEDED(r)) r = xrBeginFrame(g_session, nullptr);
    if (XR_FAILED(r)) {
        LOG_FIRST(8, "vr: the frame did not begin (%d)", int(r));
        return;
    }
    g_begun = true;
    g_should_render = state.shouldRender;
    g_time = state.predictedDisplayTime;
    LocateHead();
}

}  // namespace vr
