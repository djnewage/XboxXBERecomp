#pragma once
// A headset through OpenXR (Settings::vr, Direct3D 12 only). The session lives on the renderer's device and thread: each
// present copies the finished frame into an OpenXR swap chain and waits for the headset's next frame, which then paces the
// title in place of the frame cap. The title's side asks for the head's angles when it has a camera the player turns, and
// that frame is shown as a view of the world from them (the compositor carries it to where the head is by then); any
// other frame (menus, movies, cut scenes) is shown on a screen standing in front of the player.
#include <cstdint>

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12Resource;

namespace gpu {
class Context;
struct State;
}

namespace vr {

bool Enabled();  // the setting is on and OpenXR has not failed for good
bool Running();  // a session is showing frames: the headset paces the title

// --- the title's thread ---
// The head's angles in degrees for the frame the title is about to build: yaw grows to the left, pitch upwards, both 0
// looking ahead as the session started. Asking marks that frame as a view of the world drawn at these angles (level, no
// roll). False without a headset to ask.
bool Head(float& yaw, float& pitch);
void TitlePresented();  // each Swap, before the renderer's Present

// --- the renderer's thread (gpu/d3d12.cpp) ---
void Start(ID3D12Device* device, ID3D12CommandQueue* queue);
void NoteDraw(const gpu::State& s, uint32_t count);  // each draw: the frame's projection is read from the scene's
// while the frame's command list is open, the finished frame a shader resource: copies it for the headset
void Frame(gpu::Context& ctx, ID3D12Resource* frame, int format, uint32_t width, uint32_t height);
void Submit();  // once that list has been handed to the queue
void Next();    // after the window's present: waits for the headset's next frame

}  // namespace vr
