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
#include "core/input.h"
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
    X(xrLocateViews)                   \
    X(xrStringToPath)                  \
    X(xrCreateActionSet)               \
    X(xrCreateAction)                  \
    X(xrSuggestInteractionProfileBindings) \
    X(xrAttachSessionActionSets)       \
    X(xrSyncActions)                   \
    X(xrGetActionStateFloat)           \
    X(xrGetActionStateBoolean)         \
    X(xrGetActionStateVector2f)        \
    X(xrCreateActionSpace)             \
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
    bool world = false;  // a view of the world, else a picture for the screen
    float yaw = 0, pitch = 0;  // radians: the angles the title drew it at (level), the head's or the aiming hand's
    XrQuaternionf head{0, 0, 0, 1};  // where the head looked then: the picture is turned to it by the renderer
    bool hand = false;               // the aiming hand gave the angles: where it was and how it was turned
    XrQuaternionf hand_q{0, 0, 0, 1};
    XrVector3f hand_at{0, 0, 0};
    XrVector3f position{0, 0, 0};
    bool ahead = false;  // vr::Ahead: the title's own camera, levelled by g_level
    uint64_t at = 0;  // when the head was read for it (GetTickCount64), for the log
};

std::atomic<bool> g_failed{false}, g_running{false};
std::atomic<float> g_units{0};   // a title's own units a metre (SetUnits), 0 the setting's
std::atomic<int> g_layout{0};    // how the hand controllers' sticks play (Layout)

// the head as last located, for the title (g_lock)
std::mutex g_lock;
bool g_head_valid = false;
Pose g_head;
XrVector3f g_origin{0, 0, 0};  // where the head is when the player stands at the title's camera (recentred: both sticks in)
float g_screen_yaw = 0;        // the way the screen for menus and movies faces from there
std::atomic<float> g_turn_pending{0};  // degrees of step turning the title has not taken yet, left positive
XrQuaternionf g_hand_q{0, 0, 0, 1};
XrVector3f g_hand_at{0, 0, 0};
bool g_hand_valid = false;  // the right hand's aim is tracked
float g_hand_yaw = 0, g_hand_pitch = 0;
// What each of the title's frames is drawn with, by its number (the count of presents before it): the title writes its
// frame's when it asks for the angles, before it draws; the renderer reads the one of the frame it is drawing, by its
// own count of presents. A frame the title does not ask in again is drawn with the one before's.
constexpr uint64_t kFrames = 16;
Pose g_frames[kFrames];
uint64_t g_title_frame = 0, g_render_frame = 0;
uint64_t g_title_at = 0;    // when the title last asked (GetTickCount64): one that stops asking has no such view any more

// --- the renderer's thread from here ---
ID3D12Device* g_device = nullptr;
ID3D12CommandQueue* g_queue = nullptr;
XrInstance g_instance = XR_NULL_HANDLE;
XrSystemId g_system = XR_NULL_SYSTEM_ID;
XrSession g_session = XR_NULL_HANDLE;
XrSpace g_local = XR_NULL_HANDLE, g_view = XR_NULL_HANDLE;
uint64_t g_retry_at = 0;
bool g_begun = false, g_should_render = false;  // the renderer's: a waited frame is there for this present, and is to be drawn
XrTime g_time = 0;                              // its display time
// The headset's frames are waited for on the title's thread (TitleWait, at each Swap) and begun and ended on the
// renderer's (Submit), which so never stands still: a title that waits for the renderer in the middle of a frame (a
// buffer it locks again, twenty times a frame in a vent of Agent Under Fire) waited out the headset's wait with it.
struct Waited {
    XrTime time;
    bool render;
};
std::deque<Waited> g_waited;  // g_lock
XrTime g_title_time = 0;      // the title's thread: the frame it is building
Pose g_shown;  // the pose of the frame being shown

// a swap chain an eye: the left one alone shows a frame drawn once
struct Chain {
    XrSwapchain swap = XR_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    std::vector<XrSwapchainImageD3D12KHR> images;
    bool drew = false;
} g_chain[2];
DXGI_FORMAT g_swap_format = DXGI_FORMAT_UNKNOWN;
ComPtr<ID3D12DescriptorHeap> g_rtv_heap;
uint32_t g_rtv_inc = 0;
std::atomic<float> g_half_ipd{0.0315f};  // metres, half the distance between the eyes

// the scene's projection this frame: the perspective projection most vertices were drawn with
struct Seen {
    float x, y;  // the tangents of the half angles
    uint64_t vertices;
    uint32_t biggest = 0;  // a program's: its largest draw, whose matrix rows (x, y and w) these are
    float rows[9] = {};
};
// the last frame had a perspective in it: one without (a loading picture, a fade) is shown on the screen, whatever the
// title asked, and its 2D is drawn as it comes
std::atomic<bool> g_had_3d{true};
// a camera the title keeps itself (vr::Ahead: a car's) is not level: the turn from its axes to level ones with the same
// heading, 3x3 by rows, from the frame before's matrix (g_lock)
float g_level[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
int g_up_axis = -1;
std::vector<Seen> g_seen, g_seen_programs;  // the fixed function's projections, and vertex programs' (a title with no other)
float g_tan_x = 0, g_tan_y = 0;
float g_tan_try_x = 0, g_tan_try_y = 0;  // the last frames' own, taken once it has held a few frames
int g_tan_held = 0;

// what the headset was given, for the log: every five seconds, and when it changes
struct Given {
    uint32_t frames = 0, world = 0, screen = 0, none = 0, idle = 0, bad = 0;
} g_given;
uint64_t g_given_at = 0;
int g_given_kind = -1;

// The hand controllers play controller 1 (input::Poll takes Controls): the left stick walks, the right one turns (the head
// looks up and down) and jumps pushed up, the triggers are the triggers, the right hand's A and B the console's, its grip
// Black; the left hand's A is White, its B Start, its grip X, its trackpad the d-pad.
XrActionSet g_actions = XR_NULL_HANDLE;
XrAction g_trigger, g_grip, g_a, g_b, g_stick, g_click, g_pad, g_pad_force, g_aim;
XrSpace g_aim_space = XR_NULL_HANDLE;  // the right hand's aim
XrPath g_hand[2];
float g_controls[input::kControls] = {};  // g_lock
bool g_controls_live = false;

void Angles(const XrQuaternionf& q, float& yaw, float& pitch);
XrPath Path(const char* text) {
    XrPath p = XR_NULL_PATH;
    xrStringToPath(g_instance, text, &p);
    return p;
}

void CreateActions() {
    XrActionSetCreateInfo set{XR_TYPE_ACTION_SET_CREATE_INFO};
    std::strcpy(set.actionSetName, "game");
    std::strcpy(set.localizedActionSetName, "Game");
    if (XR_FAILED(xrCreateActionSet(g_instance, &set, &g_actions))) {
        g_actions = XR_NULL_HANDLE;
        LOGR("vr: no action set, the hand controllers are not read");
        return;
    }
    g_hand[0] = Path("/user/hand/left");
    g_hand[1] = Path("/user/hand/right");
    const auto make = [](XrAction& a, XrActionType type, const char* name, const char* shown) {
        XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO};
        info.actionType = type;
        std::strcpy(info.actionName, name);
        std::strcpy(info.localizedActionName, shown);
        info.countSubactionPaths = 2;
        info.subactionPaths = g_hand;
        xrCreateAction(g_actions, &info, &a);
    };
    make(g_trigger, XR_ACTION_TYPE_FLOAT_INPUT, "trigger", "Trigger");
    make(g_grip, XR_ACTION_TYPE_FLOAT_INPUT, "grip", "Grip");
    make(g_a, XR_ACTION_TYPE_BOOLEAN_INPUT, "a", "Lower button");
    make(g_b, XR_ACTION_TYPE_BOOLEAN_INPUT, "b", "Upper button");
    make(g_stick, XR_ACTION_TYPE_VECTOR2F_INPUT, "stick", "Stick");
    make(g_click, XR_ACTION_TYPE_BOOLEAN_INPUT, "stick_click", "Stick click");
    make(g_pad, XR_ACTION_TYPE_VECTOR2F_INPUT, "pad", "Trackpad");
    make(g_pad_force, XR_ACTION_TYPE_FLOAT_INPUT, "pad_force", "Trackpad press");
    make(g_aim, XR_ACTION_TYPE_POSE_INPUT, "aim", "Aim");
    const auto suggest = [](const char* profile, std::vector<XrActionSuggestedBinding> b) {
        XrInteractionProfileSuggestedBinding sb{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        sb.interactionProfile = Path(profile);
        sb.countSuggestedBindings = uint32_t(b.size());
        sb.suggestedBindings = b.data();
        XrResult r = xrSuggestInteractionProfileBindings(g_instance, &sb);
        if (XR_FAILED(r)) LOGR("vr: the bindings for %s were not taken (%d)", profile, int(r));
    };
    suggest("/interaction_profiles/valve/index_controller",
            {{g_trigger, Path("/user/hand/left/input/trigger/value")}, {g_trigger, Path("/user/hand/right/input/trigger/value")},
             {g_grip, Path("/user/hand/left/input/squeeze/force")}, {g_grip, Path("/user/hand/right/input/squeeze/force")},
             {g_a, Path("/user/hand/left/input/a/click")}, {g_a, Path("/user/hand/right/input/a/click")},
             {g_b, Path("/user/hand/left/input/b/click")}, {g_b, Path("/user/hand/right/input/b/click")},
             {g_stick, Path("/user/hand/left/input/thumbstick")}, {g_stick, Path("/user/hand/right/input/thumbstick")},
             {g_click, Path("/user/hand/left/input/thumbstick/click")}, {g_click, Path("/user/hand/right/input/thumbstick/click")},
             {g_pad, Path("/user/hand/left/input/trackpad")}, {g_pad, Path("/user/hand/right/input/trackpad")},
             {g_pad_force, Path("/user/hand/left/input/trackpad/force")}, {g_pad_force, Path("/user/hand/right/input/trackpad/force")},
             {g_aim, Path("/user/hand/right/input/aim/pose")}});
    suggest("/interaction_profiles/oculus/touch_controller",
            {{g_trigger, Path("/user/hand/left/input/trigger/value")}, {g_trigger, Path("/user/hand/right/input/trigger/value")},
             {g_grip, Path("/user/hand/left/input/squeeze/value")}, {g_grip, Path("/user/hand/right/input/squeeze/value")},
             {g_a, Path("/user/hand/left/input/x/click")}, {g_a, Path("/user/hand/right/input/a/click")},
             {g_b, Path("/user/hand/left/input/y/click")}, {g_b, Path("/user/hand/right/input/b/click")},
             {g_stick, Path("/user/hand/left/input/thumbstick")}, {g_stick, Path("/user/hand/right/input/thumbstick")},
             {g_click, Path("/user/hand/left/input/thumbstick/click")}, {g_click, Path("/user/hand/right/input/thumbstick/click")},
             {g_aim, Path("/user/hand/right/input/aim/pose")}});
}

void AttachActions() {
    if (g_actions == XR_NULL_HANDLE) return;
    XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attach.countActionSets = 1;
    attach.actionSets = &g_actions;
    XrResult r = xrAttachSessionActionSets(g_session, &attach);
    if (XR_FAILED(r)) LOGR("vr: the hand controllers' actions were not attached (%d)", int(r));
    else LOGR("vr: the hand controllers play controller 1");
    XrActionSpaceCreateInfo space{XR_TYPE_ACTION_SPACE_CREATE_INFO};
    space.action = g_aim;
    space.subactionPath = g_hand[1];
    space.poseInActionSpace.orientation = {0, 0, 0, 1};
    if (XR_FAILED(xrCreateActionSpace(g_session, &space, &g_aim_space))) g_aim_space = XR_NULL_HANDLE;
}

float Float(XrAction a, int hand) {
    XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
    get.action = a;
    get.subactionPath = g_hand[hand];
    XrActionStateFloat st{XR_TYPE_ACTION_STATE_FLOAT};
    return XR_SUCCEEDED(xrGetActionStateFloat(g_session, &get, &st)) && st.isActive ? st.currentState : 0;
}
bool Bool(XrAction a, int hand) {
    XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
    get.action = a;
    get.subactionPath = g_hand[hand];
    XrActionStateBoolean st{XR_TYPE_ACTION_STATE_BOOLEAN};
    return XR_SUCCEEDED(xrGetActionStateBoolean(g_session, &get, &st)) && st.isActive && st.currentState;
}
XrVector2f Vec(XrAction a, int hand, bool* active = nullptr) {
    XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
    get.action = a;
    get.subactionPath = g_hand[hand];
    XrActionStateVector2f st{XR_TYPE_ACTION_STATE_VECTOR2F};
    bool ok = XR_SUCCEEDED(xrGetActionStateVector2f(g_session, &get, &st)) && st.isActive;
    if (active) *active = ok;
    return ok ? st.currentState : XrVector2f{0, 0};
}

void ReadControls() {
    using namespace input;
    float v[kControls] = {};
    bool live = false, recentre = false;
    XrVector2f walk{0, 0};
    float turn_stick = 0, aim_y = 0;
    XrActiveActionSet active{g_actions, XR_NULL_PATH};
    XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
    sync.countActiveActionSets = 1;
    sync.activeActionSets = &active;
    if (g_actions != XR_NULL_HANDLE && xrSyncActions(g_session, &sync) == XR_SUCCESS) {
        bool left_on = false, right_on = false;
        XrVector2f l = Vec(g_stick, 0, &left_on), r = Vec(g_stick, 1, &right_on);
        live = left_on || right_on;
        walk = l;
        // the right stick turns: smoothly through the title's own stick, or by steps (Settings::vr_snap_turn degrees), one a push
        const float snap = float(app::Get().vr_snap_turn);
        static int pushed = 0;
        const int now = r.x < -0.6f ? -1 : r.x > 0.6f ? 1 : std::fabs(r.x) < 0.3f ? 0 : pushed;
        if (snap > 0) {
            if (now && now != pushed) g_turn_pending = g_turn_pending + (now < 0 ? snap : -snap);
        } else {
            turn_stick = r.x;
        }
        pushed = now;
        v[kY] = r.y > 0.7f;  // jump
        aim_y = r.y;
        // both sticks pressed in: the place and the way faced now are the middle again
        static bool both = false;
        const bool pressed = Bool(g_click, 0) && Bool(g_click, 1);
        recentre = pressed && !both;
        both = pressed;
        v[kLT] = Float(g_trigger, 0), v[kRT] = Float(g_trigger, 1);
        v[kX] = Float(g_grip, 0) > 0.35f;      // crouch
        v[kBlack] = Float(g_grip, 1) > 0.35f;  // alt. fire
        v[kA] = Bool(g_a, 1), v[kB] = Bool(g_b, 1);
        v[kWhite] = Bool(g_a, 0), v[kStart] = Bool(g_b, 0);
        v[kLThumb] = Bool(g_click, 0);
        v[kRThumb] = Bool(g_click, 1) || r.y < -0.7f;  // pressed in, or pulled back (a title's spare button: the next gadget in Agent Under Fire)
        if (Float(g_pad_force, 0) > 0.3f) {
            XrVector2f d = Vec(g_pad, 0);
            if (std::fabs(d.x) > std::fabs(d.y)) v[d.x < 0 ? kLeft : kRight] = 1;
            else v[d.y > 0 ? kUp : kDown] = 1;
        }
    }
    // where the right hand points
    XrSpaceLocation aim{XR_TYPE_SPACE_LOCATION};
    bool aimed = g_aim_space != XR_NULL_HANDLE && XR_SUCCEEDED(xrLocateSpace(g_aim_space, g_local, g_title_time, &aim)) &&
                 (aim.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) && (aim.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT);
    std::lock_guard lock(g_lock);
    std::memcpy(g_controls, v, sizeof(v));
    g_controls_live = live;
    if (aimed && !g_hand_valid) LOGR("vr: the right hand aims");
    g_hand_valid = aimed;
    if (aimed) Angles(aim.pose.orientation, g_hand_yaw, g_hand_pitch);
    if (aimed) g_hand_q = aim.pose.orientation, g_hand_at = aim.pose.position;
    // Walking goes where the head faces. The title walks along its camera, which is the aiming hand's when that aims: the
    // stick is turned by the angle between them. Its sticks (this renderer's titles so far, as Agent Under Fire's first
    // setup): the left one forward and back and turning, the right one's x stepping aside.
    float side = walk.x, ahead = walk.y;
    if (g_layout == 1) {
        // a title steered with the left stick (driving): the sticks as they are, the right one's x the turning stick's read
        g_controls[kLSUp] = std::max(walk.y, 0.0f), g_controls[kLSDown] = std::max(-walk.y, 0.0f);
        g_controls[kLSLeft] = std::max(-walk.x, 0.0f), g_controls[kLSRight] = std::max(walk.x, 0.0f);
        g_controls[kRSLeft] = std::max(-turn_stick, 0.0f), g_controls[kRSRight] = std::max(turn_stick, 0.0f);
        g_controls[kRSUp] = std::max(aim_y, 0.0f), g_controls[kRSDown] = std::max(-aim_y, 0.0f);
        g_controls[kY] = 0;        // not a jump there
        g_controls[kRThumb] = Bool(g_click, 1);
        if (recentre && g_head_valid) {
            g_origin = g_head.position;
            g_screen_yaw = g_head.yaw;
            LOGR("vr: recentred");
        }
        return;
    }
    if (app::Get().vr_hand_aim && g_hand_valid && g_head_valid) {
        const float a = g_head.yaw - g_hand_yaw;
        side = walk.x * std::cos(a) - walk.y * std::sin(a);
        ahead = walk.x * std::sin(a) + walk.y * std::cos(a);
    }
    g_controls[kLSUp] = std::max(ahead, 0.0f), g_controls[kLSDown] = std::max(-ahead, 0.0f);
    g_controls[kRSLeft] = std::max(-side, 0.0f), g_controls[kRSRight] = std::max(side, 0.0f);
    g_controls[kLSLeft] = std::max(-turn_stick, 0.0f), g_controls[kLSRight] = std::max(turn_stick, 0.0f);
    if (recentre && g_head_valid) {
        g_origin = g_head.position;
        g_screen_yaw = g_head.yaw;
        LOGR("vr: recentred");
    }
}

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

void DropSwapchain(Chain& c) {
    if (c.swap != XR_NULL_HANDLE) xrDestroySwapchain(c.swap);
    c = {};
}

void DropSession(const char* why) {
    LOGR("vr: the session ends (%s)", why);
    for (Chain& c : g_chain) DropSwapchain(c);
    if (g_view != XR_NULL_HANDLE) xrDestroySpace(g_view);
    if (g_local != XR_NULL_HANDLE) xrDestroySpace(g_local);
    if (g_session != XR_NULL_HANDLE) xrDestroySession(g_session);
    g_view = g_local = g_aim_space = XR_NULL_HANDLE;
    g_session = XR_NULL_HANDLE;
    g_system = XR_NULL_SYSTEM_ID;
    g_begun = false;
    g_running = false;
    {
        std::lock_guard lock(g_lock);
        g_head_valid = g_hand_valid = false;
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
    CreateActions();
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
    hd.NumDescriptors = 2 * kMostImages;
    if (!g_rtv_heap) g_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_rtv_heap));
    g_rtv_inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    LOGR("vr: session made on the Direct3D 12 device");
    AttachActions();
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
                g_waited.clear();
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

bool EnsureSwapchain(int eye, uint32_t w, uint32_t h) {
    Chain& c = g_chain[eye];
    if (c.swap != XR_NULL_HANDLE && w == c.w && h == c.h) return true;
    DropSwapchain(c);
    uint32_t n = 0;
    xrEnumerateSwapchainFormats(g_session, 0, &n, nullptr);
    std::vector<int64_t> formats(n);
    if (n) xrEnumerateSwapchainFormats(g_session, n, &n, formats.data());
    g_swap_format = DXGI_FORMAT_UNKNOWN;
    // BGRA first: the renderer's frames are B8G8R8A8, which copies into it as it is (Frame)
    for (DXGI_FORMAT want : {DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM,
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
    XrResult r = xrCreateSwapchain(g_session, &ci, &c.swap);
    if (XR_FAILED(r)) {
        LOGR("vr: xrCreateSwapchain %ux%u failed (%d)", w, h, int(r));
        c.swap = XR_NULL_HANDLE;
        return false;
    }
    xrEnumerateSwapchainImages(c.swap, 0, &n, nullptr);
    n = std::min(n, kMostImages);
    c.images.assign(n, {XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR});
    xrEnumerateSwapchainImages(c.swap, n, &n, reinterpret_cast<XrSwapchainImageBaseHeader*>(c.images.data()));
    for (uint32_t i = 0; i < n; i++) {
        D3D12_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = g_swap_format;
        rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        auto h0 = g_rtv_heap->GetCPUDescriptorHandleForHeapStart();
        h0.ptr += size_t(uint32_t(eye) * kMostImages + i) * g_rtv_inc;
        g_device->CreateRenderTargetView(c.images[i].texture, &rv, h0);
    }
    c.w = w, c.h = h;
    LOGR("vr: swap chain %d %ux%u, format %d, %u images", eye, w, h, int(g_swap_format), n);
    return true;
}

// the head now, for the frame the title builds next
void LocateHead() {
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    bool ok = XR_SUCCEEDED(xrLocateSpace(g_view, g_local, g_title_time, &loc)) &&
              (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) && (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT);
    std::lock_guard lock(g_lock);
    g_head_valid = ok;
    if (!ok) return;
    g_head.world = true;
    Angles(loc.pose.orientation, g_head.yaw, g_head.pitch);
    g_head.head = loc.pose.orientation;
    g_head.at = GetTickCount64();
    g_head.position = loc.pose.position;
    XrViewLocateInfo info{XR_TYPE_VIEW_LOCATE_INFO};
    info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    info.displayTime = g_title_time;
    info.space = g_view;
    XrViewState vs{XR_TYPE_VIEW_STATE};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    uint32_t n = 0;
    if (XR_SUCCEEDED(xrLocateViews(g_session, &info, &vs, 2, &n, views)) && n == 2 && (vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT)) {
        float dx = views[1].pose.position.x - views[0].pose.position.x, dy = views[1].pose.position.y - views[0].pose.position.y,
              dz = views[1].pose.position.z - views[0].pose.position.z;
        float half = std::sqrt(dx * dx + dy * dy + dz * dz) / 2;
        if (half > 0.02f && half < 0.045f) g_half_ipd = half;
    }
}
}  // namespace

bool Enabled() { return app::Get().vr && !g_failed; }
bool Running() { return g_running; }
float Units() { return g_units > 0 ? g_units.load() : float(app::Get().vr_units); }
void SetUnits(float units) { g_units = units; }
void Layout(int layout) { g_layout = layout; }
bool Stereo() { return g_running; }
float EyeOffset(int eye) { return (eye ? 1.0f : -1.0f) * g_half_ipd * Units(); }

bool Controls(float* v) {
    std::lock_guard lock(g_lock);
    if (!g_running || !g_controls_live) return false;
    std::memcpy(v, g_controls, sizeof(g_controls));
    return true;
}

bool Head(float& yaw, float& pitch) {
    std::lock_guard lock(g_lock);
    if (!g_running || !g_head_valid) return false;
    Pose pose = g_head;
    // the gun in the right hand aims, when it is tracked: the title turns to it, the picture stays with the head
    if (app::Get().vr_hand_aim && g_hand_valid) {
        pose.yaw = g_hand_yaw, pose.pitch = g_hand_pitch;
        pose.hand = true, pose.hand_q = g_hand_q, pose.hand_at = g_hand_at;
    }
    g_frames[g_title_frame % kFrames] = pose;
    g_title_at = GetTickCount64();
    yaw = pose.yaw * 180 / kPi;
    pitch = pose.pitch * 180 / kPi;
    return true;
}

void Pitch(float up) {
    std::lock_guard lock(g_lock);
    Pose& pose = g_frames[g_title_frame % kFrames];
    if (pose.world) pose.pitch = up * kPi / 180;
}

void Hold() {
    std::lock_guard lock(g_lock);
    Pose& pose = g_frames[g_title_frame % kFrames];
    if (!g_running || !g_head_valid || !pose.world) return;
    pose.head = g_head.head;
    pose.position = g_head.position;
    pose.at = g_head.at;
    g_title_at = GetTickCount64();
}

void Ahead() {
    std::lock_guard lock(g_lock);
    if (!g_running || !g_head_valid) return;
    Pose pose = g_head;
    pose.yaw = g_screen_yaw, pose.pitch = 0;
    pose.ahead = true;
    g_frames[g_title_frame % kFrames] = pose;
    g_title_at = GetTickCount64();
}

void NoteProgram(float tan_x, float tan_y, uint32_t count, const float* m) {
    if (!g_running) return;
    const auto keep = [&](Seen& seen) {
        if (count <= seen.biggest) return;
        seen.biggest = count;
        for (int i = 0; i < 3; i++) seen.rows[i] = m[i], seen.rows[3 + i] = m[4 + i], seen.rows[6 + i] = m[12 + i];
    };
    for (Seen& seen : g_seen_programs)
        if (std::fabs(seen.x - tan_x) < 1e-3f && std::fabs(seen.y - tan_y) < 1e-3f) {
            seen.vertices += count;
            keep(seen);
            return;
        }
    if (g_seen_programs.size() < 16) {
        g_seen_programs.push_back({tan_x, tan_y, count});
        keep(g_seen_programs.back());
    }
}

void Flat() {
    std::lock_guard lock(g_lock);
    g_frames[g_title_frame % kFrames] = {};
}

uint32_t TitleFrame() {
    std::lock_guard lock(g_lock);
    return uint32_t(g_title_frame);
}
void RenderFrame(uint32_t frame) { g_render_frame = frame; }
uint32_t RenderHeight() { return Enabled() ? app::Get().vr_render_height : 0; }

void TitlePresented() {
    std::lock_guard lock(g_lock);
    Pose next = g_frames[g_title_frame % kFrames];
    if (!g_running || GetTickCount64() - g_title_at > 2000) next = {};  // (two seconds: a quarter was shorter than a hitch, and the view fell to the screen for a moment)
    g_frames[++g_title_frame % kFrames] = next;
}

namespace {
XrQuaternionf Conj(const XrQuaternionf& q) { return {-q.x, -q.y, -q.z, q.w}; }
// from the title's camera to the head
void Turn(const Pose& pose, float turn[9]) {
    const XrQuaternionf q = Mul(Conj(Level(pose.yaw, pose.pitch)), pose.head);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    turn[0] = 1 - 2 * (y * y + z * z), turn[1] = 2 * (x * y - z * w), turn[2] = 2 * (x * z + y * w);
    turn[3] = 2 * (x * y + z * w), turn[4] = 1 - 2 * (x * x + z * z), turn[5] = 2 * (y * z - x * w);
    turn[6] = 2 * (x * z - y * w), turn[7] = 2 * (y * z + x * w), turn[8] = 1 - 2 * (x * x + y * y);
    if (pose.ahead) {
        // from the camera's axes to the level ones first
        float level[9], r[9];
        {
            std::lock_guard lock(g_lock);
            std::memcpy(level, g_level, sizeof(level));
        }
        std::memcpy(r, turn, sizeof(r));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) turn[i * 3 + j] = level[i * 3] * r[j] + level[i * 3 + 1] * r[3 + j] + level[i * 3 + 2] * r[6 + j];
    }
}
}  // namespace

bool Fov(float& tan_x, float& tan_y) {
    tan_x = g_tan_x, tan_y = g_tan_y;  // the renderer's own thread
    return tan_x > 0 && tan_y > 0;
}

bool Eye(int eye, float turn[9], float& offset) {
    Pose pose;
    {
        std::lock_guard lock(g_lock);
        pose = g_frames[g_render_frame % kFrames];
    }
    if (!g_running || !pose.world || !g_had_3d) return false;
    Turn(pose, turn);
    offset = EyeOffset(eye);
    return true;
}

bool EyeShift(int eye, float turn[9], float shift[3]) {
    Pose pose;
    XrVector3f origin;
    {
        std::lock_guard lock(g_lock);
        pose = g_frames[g_render_frame % kFrames];
        origin = g_origin;
    }
    if (!g_running || !pose.world || !g_had_3d) return false;
    Turn(pose, turn);
    // how far the head is from the middle, in the camera's axes, in the title's units
    const float d[3] = {pose.position.x - origin.x, pose.position.y - origin.y, pose.position.z - origin.z};
    const XrQuaternionf q = Level(pose.yaw, pose.pitch);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float g[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w),     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
    const float units = Units();
    float t[3];
    for (int i = 0; i < 3; i++) t[i] = (g[i] * d[0] + g[3 + i] * d[1] + g[6 + i] * d[2]) * units;
    // seen from the head's own axes, and the eye to one side there
    for (int i = 0; i < 3; i++) shift[i] = turn[i] * t[0] + turn[3 + i] * t[1] + turn[6 + i] * t[2];
    shift[0] += EyeOffset(eye);
    return true;
}

bool Held(float rot[9], float at[3]) {
    Pose pose;
    XrVector3f origin;
    {
        std::lock_guard lock(g_lock);
        pose = g_frames[g_render_frame % kFrames];
        origin = g_origin;
    }
    if (!g_running || !pose.world || !pose.hand) return false;
    const XrQuaternionf level = Level(pose.yaw, pose.pitch);
    {
        const XrQuaternionf q = Mul(Conj(level), pose.hand_q);
        const float x = q.x, y = q.y, z = q.z, w = q.w;
        rot[0] = 1 - 2 * (y * y + z * z), rot[1] = 2 * (x * y - z * w), rot[2] = 2 * (x * z + y * w);
        rot[3] = 2 * (x * y + z * w), rot[4] = 1 - 2 * (x * x + z * z), rot[5] = 2 * (y * z - x * w);
        rot[6] = 2 * (x * z - y * w), rot[7] = 2 * (y * z + x * w), rot[8] = 1 - 2 * (x * x + y * y);
    }
    const float x = level.x, y = level.y, z = level.z, w = level.w;
    const float g[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w),     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
    const float d[3] = {pose.hand_at.x - origin.x, pose.hand_at.y - origin.y, pose.hand_at.z - origin.z};
    const float units = Units();
    for (int i = 0; i < 3; i++) at[i] = (g[i] * d[0] + g[3 + i] * d[1] + g[6 + i] * d[2]) * units;
    return true;
}

float TakeTurn() { return g_turn_pending.exchange(0); }

bool TitleTurn(float turn[9]) {
    Pose pose;
    {
        std::lock_guard lock(g_lock);
        pose = g_frames[g_title_frame % kFrames];
    }
    if (!g_running || !pose.world || !g_had_3d) return false;
    Turn(pose, turn);
    return true;
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

void NoteDraw(const gpu::State& s, uint32_t count) {
    if (!g_running) return;
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

void Frame(gpu::Context& ctx, int eye, ID3D12Resource* frame, int format, uint32_t width, uint32_t height) {
    Chain& c = g_chain[eye];
    if (eye == 0) {
        // the frame's projection, whether or not it is shown
        // The fixed function's when the frame has any; a program's matrix may be no camera's at all (a small room's few
        // walls are outweighed by one), so those count only for a title that draws nothing else. A plausible one (33 to
        // 150 degrees), the same for a few frames.
        const Seen* most = nullptr;
        for (const Seen& seen : g_seen.empty() ? g_seen_programs : g_seen)
            if (!most || seen.vertices > most->vertices) most = &seen;
        if (most && most->x > 0.3f && most->x < 4 && most->y > 0.3f && most->y < 4) {
            if (std::fabs(most->x - g_tan_try_x) > 1e-3f || std::fabs(most->y - g_tan_try_y) > 1e-3f) {
                g_tan_try_x = most->x, g_tan_try_y = most->y;
                g_tan_held = 0;
            } else if (++g_tan_held == 5 && (std::fabs(most->x - g_tan_x) > 1e-3f || std::fabs(most->y - g_tan_y) > 1e-3f)) {
                g_tan_x = most->x, g_tan_y = most->y;
                LOGR("vr: the scene's field of view %.1f by %.1f degrees", 2 * std::atan(g_tan_x) * 180 / kPi, 2 * std::atan(g_tan_y) * 180 / kPi);
            }
        }
        g_had_3d = most != nullptr;
        // A program's rows are the camera's right, up and (w) forward in the world. The world's up is the axis the camera's
        // up lies along most; with it, the level frame of the same heading, seen from the camera's axes (x right, y up, z
        // towards the viewer): y is the world's up, z what is left of the camera's own z, x across them.
        if (most && g_seen.empty() && most->biggest) {
            const auto unit = [](const float* r, float* o) {
                const float l = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                for (int i = 0; i < 3; i++) o[i] = l > 0 ? r[i] / l : 0;
            };
            float right[3], up[3], ahead[3];
            unit(most->rows, right), unit(most->rows + 3, up), unit(most->rows + 6, ahead);
            int k = 0;
            for (int i = 1; i < 3; i++)
                if (std::fabs(up[i]) > std::fabs(up[k])) k = i;
            if (k != g_up_axis) LOGR("vr: the world's up is its %c axis (%s), from the camera's matrix", "xyz"[k], up[k] > 0 ? "positive" : "negative");
            const float s = up[k] > 0 ? 1.0f : -1.0f;
            float y[3] = {s * right[k], s * up[k], -s * ahead[k]}, z[3] = {0, 0, 1};
            unit(y, y);
            const float d = z[0] * y[0] + z[1] * y[1] + z[2] * y[2];
            for (int i = 0; i < 3; i++) z[i] -= d * y[i];
            if (std::sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]) > 0.2f) {
                unit(z, z);
                const float x[3] = {y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]};
                std::lock_guard lock(g_lock);
                g_up_axis = k;
                for (int i = 0; i < 3; i++) g_level[i * 3] = x[i], g_level[i * 3 + 1] = y[i], g_level[i * 3 + 2] = z[i];
            }
        }
        g_seen.clear();
        g_seen_programs.clear();
        std::lock_guard lock(g_lock);
        g_shown = g_frames[g_render_frame % kFrames];
        g_chain[0].drew = g_chain[1].drew = false;
    }
    if (eye == 0) {
        std::lock_guard lock(g_lock);
        g_begun = !g_waited.empty();
        if (g_begun) g_should_render = g_waited.front().render, g_time = g_waited.front().time;
    }
    if (!g_begun || !g_should_render || !frame || !EnsureSwapchain(eye, width, height)) return;
    uint32_t index = 0;
    XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(c.swap, &acquire, &index))) return;
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    if (XR_FAILED(xrWaitSwapchainImage(c.swap, &wait)) || index >= c.images.size()) {
        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(c.swap, &release);
        return;
    }
    // The frame is gamma encoded B8G8R8A8 and the image its sRGB twin: the bytes are copied as they are, which no state the
    // title's last draws left can spoil (the frame is a shader resource here, the image a render target, and both are
    // again after). Any other pair goes through the pass below.
    const D3D12_RESOURCE_DESC from = frame->GetDesc(), to = c.images[index].texture->GetDesc();
    if (g_swap_format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB && DXGI_FORMAT(format) == DXGI_FORMAT_B8G8R8A8_UNORM && from.Width == to.Width &&
        from.Height == to.Height && from.SampleDesc.Count == 1 && from.MipLevels == to.MipLevels && from.DepthOrArraySize == to.DepthOrArraySize) {
        D3D12_RESOURCE_STATES fs = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, is = D3D12_RESOURCE_STATE_RENDER_TARGET;
        ctx.Transition(frame, fs, D3D12_RESOURCE_STATE_COPY_SOURCE);
        ctx.Transition(c.images[index].texture, is, D3D12_RESOURCE_STATE_COPY_DEST);
        ctx.List()->CopyResource(c.images[index].texture, frame);
        ctx.Transition(c.images[index].texture, is, D3D12_RESOURCE_STATE_RENDER_TARGET);
        ctx.Transition(frame, fs, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        static bool said = false;
        if (!said) LOGR("vr: the frame is copied to the headset as it is");
        said = true;
        c.drew = true;
        return;
    }
    static bool said_pass = false;
    if (!said_pass) LOGR("vr: the frame goes to the headset through a pass (formats %d to %d, %llux%u to %llux%u, mips %u to %u)", format, int(g_swap_format),
                         (unsigned long long)from.Width, from.Height, (unsigned long long)to.Width, to.Height, unsigned(from.MipLevels), unsigned(to.MipLevels));
    said_pass = true;
    auto rtv = g_rtv_heap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += size_t(uint32_t(eye) * kMostImages + index) * g_rtv_inc;
    ID3D12Resource* const src[3] = {frame, nullptr, nullptr};
    const DXGI_FORMAT fmt[3] = {DXGI_FORMAT(format), DXGI_FORMAT(format), DXGI_FORMAT(format)};
    const float p[8] = {Srgb(g_swap_format) ? 1.0f : 0.0f};
    ctx.Pass(kCopyHlsl, "psvr", g_swap_format, rtv, width, height, src, fmt, p, -1);
    ctx.Unbind();
    c.drew = true;
}

namespace {
void Report(int kind, bool finite, XrResult result) {
    Given& g = g_given;
    g.frames++;
    (kind == 2 ? g.world : kind == 1 ? g.screen : kind == 0 ? g.none : g.idle)++;
    if (!finite) g.bad++;
    const uint64_t now = GetTickCount64();
    static XrResult last_result = XR_SUCCESS;
    if (kind != g_given_kind || result != last_result || now - g_given_at >= 5000) {
        static const char* const kKinds[] = {"not begun", "no layer", "the screen", "the world"};
        uint64_t title;
        {
            std::lock_guard lock(g_lock);
            title = g_title_frame;
        }
        LOGR("vr: %u frames: %u the world, %u the screen, %u no layer, %u not begun, %u with a value not finite; now %s, render %d, "
             "field of view %.1f by %.1f, xrEndFrame %d; the head read %llu ms before, frame %llu of the title's %llu, yaw %.0f pitch %.0f",
             g.frames, g.world, g.screen, g.none, g.idle, g.bad, kKinds[kind + 1], int(g_should_render),
             2 * std::atan(g_tan_x) * 180 / kPi, 2 * std::atan(g_tan_y) * 180 / kPi, int(result),
             (unsigned long long)(g_shown.at ? now - g_shown.at : 0), (unsigned long long)g_render_frame, (unsigned long long)title,
             g_shown.yaw * 180 / kPi, g_shown.pitch * 180 / kPi);
        g = {};
        g_given_at = now;
        g_given_kind = kind;
        last_result = result;
    }
}
}  // namespace

void Submit() {
    // every frame waited for is begun and ended, drawn or not: the title's next wait stands until this one is begun
    bool have;
    {
        std::lock_guard lock(g_lock);
        have = !g_waited.empty();
        if (have) {
            g_time = g_waited.front().time;
            g_waited.pop_front();
        }
    }
    if (!have || !g_running) {
        if (g_running) Report(-1, true, XR_SUCCESS);
        return;
    }
    g_begun = false;
    if (XrResult b = xrBeginFrame(g_session, nullptr); XR_FAILED(b)) {
        LOG_FIRST(8, "vr: the frame did not begin (%d)", int(b));
        for (Chain& c : g_chain)
            if (c.drew) {
                XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                xrReleaseSwapchainImage(c.swap, &release);
                c.drew = false;
            }
        return;
    }
    XrCompositionLayerProjectionView views[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
    XrCompositionLayerProjection world{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerQuad screen{XR_TYPE_COMPOSITION_LAYER_QUAD};
    const XrCompositionLayerBaseHeader* layers[1] = {};
    uint32_t count = 0;
    for (Chain& c : g_chain)
        if (c.drew) {
            XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            xrReleaseSwapchainImage(c.swap, &release);
        }
    if (g_chain[0].drew) {
        const Chain& left = g_chain[0];
        float aspect = float(left.w) / float(left.h);
        if (g_shown.world && g_had_3d) {
            // each eye's picture, drawn level from its place beside the head's; the left one for both when only it was drawn
            float ty = g_tan_y > 0 ? g_tan_y : std::tan(30 * kPi / 180), tx = g_tan_x > 0 ? g_tan_x : ty * aspect;
            bool both = g_chain[1].drew;
            for (int i = 0; i < 2; i++) {
                const Chain& c = g_chain[both ? i : 0];
                float side = both ? (i ? 1.0f : -1.0f) * g_half_ipd : 0;
                const XrQuaternionf& h = g_shown.head;
                // the head's right: its orientation turning +x
                const XrVector3f right{1 - 2 * (h.y * h.y + h.z * h.z), 2 * (h.x * h.y + h.z * h.w), 2 * (h.x * h.z - h.y * h.w)};
                views[i].pose.orientation = h;
                views[i].pose.position = {g_shown.position.x + side * right.x, g_shown.position.y + side * right.y,
                                          g_shown.position.z + side * right.z};
                views[i].fov = {-std::atan(tx), std::atan(tx), std::atan(ty), -std::atan(ty)};
                views[i].subImage = {c.swap, {{0, 0}, {int32_t(c.w), int32_t(c.h)}}, 0};
            }
            world.space = g_local;
            world.viewCount = 2;
            world.views = views;
            layers[count++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&world);
        } else {
            screen.space = g_local;
            screen.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            screen.subImage = {left.swap, {{0, 0}, {int32_t(left.w), int32_t(left.h)}}, 0};
            float yaw;
            XrVector3f at;
            {
                std::lock_guard lock(g_lock);
                yaw = g_screen_yaw, at = g_origin;
            }
            screen.pose.orientation = Level(yaw, 0);
            screen.pose.position = {at.x - kScreenDistance * std::sin(yaw), at.y, at.z - kScreenDistance * std::cos(yaw)};
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
    {
        const Pose& p = g_shown;
        const float values[] = {p.yaw, p.pitch, p.head.x, p.head.y, p.head.z, p.head.w, p.position.x, p.position.y, p.position.z, g_tan_x, g_tan_y};
        bool finite = true;
        for (float v : values) finite = finite && std::isfinite(v);
        Report(count == 0 ? 0 : g_shown.world && g_had_3d ? 2 : 1, finite, r);
    }
    g_chain[0].drew = g_chain[1].drew = false;
}

void Next() {
    if (!Enabled() || g_instance == XR_NULL_HANDLE) return;
    static uint64_t look = 0;
    if (GetTickCount64() >= look) {
        look = GetTickCount64() + 1000;
        app::ReloadVr();
    }
    if (g_session == XR_NULL_HANDLE) {
        if (GetTickCount64() < g_retry_at) return;
        g_retry_at = GetTickCount64() + kRetryMs;
        if (!CreateSession()) return;
    }
    PollEvents();
}

void TitleWait() {
    if (!g_running || g_session == XR_NULL_HANDLE) return;
    XrFrameState state{XR_TYPE_FRAME_STATE};
    XrResult r = xrWaitFrame(g_session, nullptr, &state);
    if (XR_FAILED(r)) {
        LOG_FIRST(8, "vr: no frame to wait for (%d)", int(r));
        return;
    }
    {
        std::lock_guard lock(g_lock);
        while (g_waited.size() >= 4) g_waited.pop_front();
        g_waited.push_back({state.predictedDisplayTime, state.shouldRender != XR_FALSE});
    }
    g_title_time = state.predictedDisplayTime;
    LocateHead();
    ReadControls();
}

}  // namespace vr
