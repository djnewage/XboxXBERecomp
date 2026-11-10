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
float Units();                 // the title's units in a metre: its own (SetUnits), else Settings::vr_units
void SetUnits(float units);    // a title whose units are known, from GameInfo::init
// how the hand controllers' sticks play: 0 a walker's (the left one walks where the head faces, the right one turns),
// 1 as they are (a driving game: the left one steers)
void Layout(int layout);
bool Stereo();   // the renderer draws everything twice, a picture an eye
// how far to the right of the head an eye is (0 left, 1 right), in the title's units (Settings::vr_units a metre)
float EyeOffset(int eye);
// The renderer's, for the frame it is drawing: false when it is not a view of the world (drawn as it comes, the same for
// both eyes). turn: the rotation from the title's camera to the head, 3x3 by rows, in the camera's own axes (x right, y
// up, z towards the viewer); offset: EyeOffset.
bool Eye(int eye, float turn[9], float& offset);
// the same for the scene: shift is how far the eye stands from the title's camera in the head's axes, the head's own
// movement from the middle included (leaning, ducking), in the title's units
bool EyeShift(int eye, float turn[9], float shift[3]);
// what the title draws as held (gpu::State::held) goes to the aiming hand: rot turns it from the camera's axes to the
// hand's (its roll), by rows; at is where the hand is from the camera, in the camera's axes and the title's units.
// False when no hand aims the frame.
bool Held(float rot[9], float at[3]);
bool Fov(float& tan_x, float& tan_y);
uint32_t RenderHeight();  // the lines an eye's picture is drawn with (Settings::vr_render_height), 0 without a headset  // the tangents of the half angles the frames are drawn at, once known

// --- the title's thread ---
// The angles in degrees the title is to draw the frame it is about to build at: the aiming hand's when one is tracked
// (Settings::vr_hand_aim), else the head's; yaw grows to the left, pitch upwards, both 0 looking ahead as the session
// started. The renderer turns the picture from them to the head. Asking marks the frames from here as views of the world drawn at these angles
// (level, no roll), until Flat or a quarter of a second without asking. False without a headset to ask.
bool Head(float& yaw, float& pitch);
// the title's camera stands still (a level paused): its frames stay views of the world at the angles last asked for,
// seen from where the head is now
// the turn from the title's camera to the head for the frame the title is building (as Eye's, for its own culling);
// false when that frame is not a view of the world
bool TitleTurn(float turn[9]);
// the frame the title is building is drawn at this pitch after all (degrees, upwards), not the one Head gave: a title
// whose camera did not follow
void Pitch(float up);
// degrees the title is to turn its camera by now, to the left positive: the right stick's steps (Settings::vr_snap_turn)
float TakeTurn();
// the title's camera looks ahead, level, the way the player faced when recentred, whatever the head does (a car's, a
// rail's): its frames are views of the world from there, turned to the head by the renderer
void Ahead();
// the right hand is tracked and is to aim (Settings::vr_hand_aim): what Head gives is its angles
bool HandAims();
// how far up a title's own camera looked in the last frame drawn, in degrees, from a vertex program's matrix (a title
// drawing through programs: Agent Under Fire's driving levels); false before any
bool CameraPitch(float& up);
// its heading likewise, in degrees growing to the left, without a jump at a full turn; from where it starts is arbitrary
bool CameraYaw(float& left);
void Yaw(float left);   // as Pitch: the frame the title is building is drawn at this yaw
float TurnStick();      // the right stick's x, -1 left to 1 right (0 while it turns by steps: TakeTurn)
// A title that knows its own camera: the frame it is building is a view of the world from it. level turns the camera's
// axes (x right, y up, z towards the viewer) to level ones of the same heading, 3x3 by rows (what Ahead finds from the
// renderer a frame late); the heading is the given yaw (degrees to the left, as Head's), or with own_yaw false the way
// the player faced when recentred.
void View(const float level[9], bool own_yaw, float yaw);
// the camera as the renderer last read it from a vertex program's matrix: its right, its up and the way it looks, in
// the world, three rows; false before any
bool SeenAxes(float axes[9]);
uint64_t Ticks();  // milliseconds, for a title's own pacing of what it asks
void Hold();
void Flat();            // the frames from here are not such a view (a menu, a cut scene): shown on the screen
// the hand controllers as controller 1's inputs, 0 to 1 each in input::Control order; false with none in the hands
bool Controls(float* v);
void TitlePresented();
// after the renderer's Present was asked for: waits for the headset's next frame, which paces the title, and reads the
// head and the hand controllers for it
void TitleWait();
uint32_t TitleFrame();  // the number of the frame the title is building: the count of its Swaps  // each Swap, before the renderer's Present

// --- the renderer's thread (gpu/d3d12.cpp) ---
void Start(ID3D12Device* device, ID3D12CommandQueue* queue);
// each draw and clear: the title's frame it is of (gpu::State::vr_frame), whose pose the calls below use
void RenderFrame(uint32_t frame);
void NoteDraw(const gpu::State& s, uint32_t count);
void NoteProgram(float tan_x, float tan_y, uint32_t count, const float* matrix);  // matrix: its four rows  // the same for a vertex program's perspective  // each draw: the frame's projection is read from the scene's
// while the frame's command list is open, the finished frame a shader resource: copies it for the headset
void Frame(gpu::Context& ctx, int eye, ID3D12Resource* frame, int format, uint32_t width, uint32_t height);
void Submit();  // once that list has been handed to the queue
void Next();    // after the window's present: the session's events, the settings again

}  // namespace vr
