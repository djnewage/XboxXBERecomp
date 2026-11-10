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
bool Stereo();   // the renderer draws everything twice, a picture an eye
// how far to the right of the head an eye is (0 left, 1 right), in the title's units (Settings::vr_units a metre)
float EyeOffset(int eye);
// The renderer's, for the frame it is drawing: false when it is not a view of the world (drawn as it comes, the same for
// both eyes). turn: the rotation from the title's camera to the head, 3x3 by rows, in the camera's own axes (x right, y
// up, z towards the viewer); offset: EyeOffset.
bool Eye(int eye, float turn[9], float& offset);
bool Fov(float& tan_x, float& tan_y);  // the tangents of the half angles the frames are drawn at, once known

// --- the title's thread ---
// The angles in degrees the title is to draw the frame it is about to build at: the aiming hand's when one is tracked
// (Settings::vr_hand_aim), else the head's; yaw grows to the left, pitch upwards, both 0 looking ahead as the session
// started. The renderer turns the picture from them to the head. Asking marks the frames from here as views of the world drawn at these angles
// (level, no roll), until Flat or a quarter of a second without asking. False without a headset to ask.
bool Head(float& yaw, float& pitch);
// the title's camera stands still (a level paused): its frames stay views of the world at the angles last asked for,
// seen from where the head is now
void Hold();
void Flat();            // the frames from here are not such a view (a menu, a cut scene): shown on the screen
// the hand controllers as controller 1's inputs, 0 to 1 each in input::Control order; false with none in the hands
bool Controls(float* v);
void TitlePresented();  // each Swap, before the renderer's Present

// --- the renderer's thread (gpu/d3d12.cpp) ---
void Start(ID3D12Device* device, ID3D12CommandQueue* queue);
void NoteDraw(const gpu::State& s, uint32_t count);  // each draw: the frame's projection is read from the scene's
// while the frame's command list is open, the finished frame a shader resource: copies it for the headset
void Frame(gpu::Context& ctx, int eye, ID3D12Resource* frame, int format, uint32_t width, uint32_t height);
void Submit();  // once that list has been handed to the queue
void Next();    // after the window's present: waits for the headset's next frame

}  // namespace vr
