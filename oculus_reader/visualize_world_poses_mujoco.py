"""Visualize the head and controller poses in the world frame with MuJoCo.

Each tracked body is drawn as an RGB coordinate triad (red = local x, green =
local y, blue = local z) carried by a MuJoCo mocap body; untracked bodies fade
out instead of freezing in place. The controllers leave blue trails. The look
and controls follow the CODA_deisgn glove viewers: a white MuJoCo Playground
style scene, a translucent readout in the top-left corner and a key help box
in the bottom-left.

Run it like `stream_world_poses.py`, whose world-frame conversion it reuses:

    python oculus_reader/visualize_world_poses_mujoco.py
    python oculus_reader/visualize_world_poses_mujoco.py --ip 10.0.32.101
    python oculus_reader/visualize_world_poses_mujoco.py --demo   # no headset

Left drag: orbit | Right drag: pan | Wheel: zoom
V: frame view | T: trails | C: clear trails | Space: pause | Esc: close
"""
import argparse
import collections
import struct
import time
import zlib
from pathlib import Path

import glfw
import mujoco
import numpy as np

from oculus_reader.stream_world_poses import quaternion_from_matrix, world_poses

BODIES = ('head', 'left', 'right')
CONTROLLERS = ('left', 'right')
# rgba of the local x/y/z axis of every triad
AXIS_COLORS = ('0.9 0.25 0.2', '0.2 0.75 0.35', '0.25 0.5 0.9')
# shared accent of the CODA_deisgn viewers (viewer_style.ACCENT_BLUE)
ACCENT_BLUE = (0.25, 0.60, 0.95, 1)
# stay below MuJoCo's 0.995 opaque cutoff so shadow passes don't brighten lines
TRAIL_RGBA = np.array((*ACCENT_BLUE[:3], 0.99), dtype=np.float32)
TRAIL_SECONDS = 5.0
HELP = ('Left drag: orbit | Right drag: pan | Wheel: zoom\n'
        'V: frame view | T: trails | C: clear trails | Space: pause | Esc: close')

# OpenXR LOCAL space is Y-up with -Z forward; MuJoCo is Z-up, so OpenXR
# (x, y, z) becomes (x, -z, y): up is +Z and forward is +Y. This is a proper rotation (determinant +1), so the triads keep their handedness.
WORLD_TO_MUJOCO = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])


def _triad_geoms(axis_length, radius, alpha=1.0):
    """XML for the three axis capsules and the little sphere at their origin."""
    geoms = ['<geom type="sphere" size="{:.4f}" rgba="0.55 0.55 0.58 {:.2f}"/>'
             .format(radius * 1.8, alpha)]
    for index, colour in enumerate(AXIS_COLORS):
        end = np.zeros(3)
        end[index] = axis_length
        geoms.append('<geom type="capsule" fromto="0 0 0 {:.4f} {:.4f} {:.4f}" '
                     'size="{:.4f}" rgba="{} {:.2f}"/>'
                     .format(end[0], end[1], end[2], radius, colour, alpha))
    return '\n      '.join(geoms)


def build_scene_xml(axis_length=0.12, floor_height=-1.3):
    """Build the scene: a floor, a static world-origin triad and one mocap body each."""
    bodies = []
    for name in BODIES:
        bodies.append(
            '<body name="{}" mocap="true" pos="0 0 0">\n      {}\n    </body>'
            .format(name, _triad_geoms(axis_length, 0.006)))
    # Appearance of the CODA_deisgn glove viewers, which take it from MuJoCo
    # Playground's LEAP Sim2Sim scene (Apache-2.0): white checker floor with
    # black grid lines, white sky, headlight plus a shadow casting spotlight.
    return """
<mujoco model="oculus_world_poses">
  <visual>
    <!-- from behind and above the user, looking forward (MuJoCo +Y) -->
    <global azimuth="75" elevation="-20" offwidth="1920" offheight="1080"/>
    <headlight ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="1 1 1"/>
    <quality shadowsize="8192"/>
  </visual>

  <!-- frame the free camera on the volume the head and hands move in -->
  <statistic center="0 0.15 -0.3" extent="1.1"/>

  <default>
    <geom contype="0" conaffinity="0"/>
  </default>

  <asset>
    <texture name="ground_texture" type="2d" builtin="checker" rgb1="1 1 1" rgb2="1 1 1"
             mark="edge" markrgb="0 0 0" width="300" height="300"/>
    <material name="ground_material" texture="ground_texture" texuniform="true"
              texrepeat="2 2" reflectance="0"/>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
  </asset>

  <worldbody>
    <light name="spotlight" mode="fixed" pos="3 2 4" dir="-3 -2 -5.3" cutoff="60"
           diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 {floor:.4f}"
          material="ground_material"/>
    <!-- the world origin: head pose at app start / last recentre -->
    <body name="world_origin" pos="0 0 0">
      {origin}
    </body>
    {bodies}
  </worldbody>
</mujoco>
""".format(floor=floor_height,
           origin=_triad_geoms(axis_length * 1.6, 0.004, alpha=0.35),
           bodies='\n    '.join(bodies))


def mujoco_pose(pose):
    """Convert a world-frame 4x4 pose into a MuJoCo (position, wxyz quaternion)."""
    position = WORLD_TO_MUJOCO @ pose[:3, 3]
    rotation = WORLD_TO_MUJOCO @ pose[:3, :3]
    # quaternion_from_matrix returns (x, y, z, w), MuJoCo wants (w, x, y, z)
    return position, np.roll(quaternion_from_matrix(rotation), 1)


class MujocoPoseScene:
    """Owns the model/data pair, places the mocap bodies and keeps the trails."""

    def __init__(self, axis_length=0.12, floor_height=-1.3):
        self.model = mujoco.MjModel.from_xml_string(
            build_scene_xml(axis_length, floor_height))
        self.data = mujoco.MjData(self.model)
        self.mocap_ids = {}
        self.geom_ids = {}
        for name in BODIES:
            body = self.model.body(name)
            self.mocap_ids[name] = int(body.mocapid[0])
            start = int(body.geomadr[0])
            self.geom_ids[name] = list(range(start, start + int(body.geomnum[0])))
        self.show_trail = True
        # (time, position or None); None breaks the line where tracking dropped
        self.trails = {name: collections.deque() for name in CONTROLLERS}
        mujoco.mj_forward(self.model, self.data)

    def clear_trails(self):
        for trail in self.trails.values():
            trail.clear()

    def update(self, poses, now):
        """Place every tracked body, and fade out the ones that are not tracked."""
        for name in BODIES:
            pose = poses.get(name)
            alpha = 1.0 if pose is not None else 0.0
            for geom_id in self.geom_ids[name]:
                self.model.geom_rgba[geom_id, 3] = alpha
            position = None
            if pose is not None:
                position, quat = mujoco_pose(pose)
                self.data.mocap_pos[self.mocap_ids[name]] = position
                self.data.mocap_quat[self.mocap_ids[name]] = quat
            if name in self.trails:
                trail = self.trails[name]
                trail.append((now, position))
                while trail and now - trail[0][0] > TRAIL_SECONDS:
                    trail.popleft()
        mujoco.mj_forward(self.model, self.data)

    def annotate(self, scene):
        """Draw the controller trails into the scene as line segments."""
        if not self.show_trail:
            return
        for trail in self.trails.values():
            previous = None
            for _, point in trail:
                if point is not None and previous is not None and \
                        np.linalg.norm(point - previous) > 1e-7:
                    if scene.ngeom >= scene.maxgeom:
                        return
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                                        np.zeros(3), np.eye(3).ravel(), TRAIL_RGBA)
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2, previous, point)
                    scene.ngeom += 1
                previous = point


def readout(poses, rate, paused, source):
    """Left/right columns of the top-left overlay table."""
    labels, values = ['Source', 'Status'], [source]
    if paused:
        values.append('Paused (Space to resume)')
    elif not poses:
        values.append('Waiting for the head pose (put the headset on)')
    else:
        values.append('Tracking  {:.0f} Hz'.format(rate))
    for name in BODIES:
        labels.append(name.capitalize())
        pose = poses.get(name)
        if pose is None:
            values.append('not tracked')
            continue
        x, y, z = pose[:3, 3]
        qx, qy, qz, qw = quaternion_from_matrix(pose)
        values.append('pos {:+.3f} {:+.3f} {:+.3f} m   quat {:+.3f} {:+.3f} {:+.3f} {:+.3f}'
                      .format(x, y, z, qx, qy, qz, qw))
    labels.append('Frame')
    values.append('OpenXR LOCAL: +Y up, -Z forward, origin = head at app start')
    return '\n'.join(labels), '\n'.join(values)


class RateMeter:
    """Samples per second, counted from changes of the reader's sample object."""

    def __init__(self):
        self.times = collections.deque()
        self.last = None

    def update(self, sample, now):
        if sample is not self.last and sample:
            self.times.append(now)
        self.last = sample
        while self.times and now - self.times[0] > 1.0:
            self.times.popleft()
        return len(self.times)


def demo_transforms(t):
    """Synthetic reader output: a head looking around, controllers drawing loops."""
    def transform(rotation, position):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = rotation, position
        return out

    def rot_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_x(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    head = transform(rot_y(0.3 * np.sin(0.4 * t)), [0.05 * np.sin(0.3 * t), 0.0, 0.0])
    transforms = {'h': head}
    for key, side in (('l', -1.0), ('r', 1.0)):
        world = transform(rot_x(-0.6 + 0.4 * np.sin(t + side)) @ rot_y(0.5 * np.cos(0.7 * t)),
                          [0.22 * side + 0.08 * np.cos(1.3 * t + side),
                           -0.35 + 0.08 * np.sin(2.6 * t + side),
                           -0.35 + 0.06 * np.sin(1.3 * t)])
        # the APK reports controllers relative to the head
        transforms[key] = np.linalg.inv(head) @ world
    # the left controller drops out for a moment every 6 s
    if (t % 6.0) > 5.0:
        del transforms['l']
    return transforms


def save_png(path, pixels):
    """Save a top-to-bottom H x W x 3 uint8 RGB array without Pillow."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width, _ = pixels.shape

    def chunk(kind, payload):
        return (struct.pack('!I', len(payload)) + kind + payload
                + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff))

    rows = b''.join(b'\0' + row.tobytes() for row in pixels)
    result = b'\x89PNG\r\n\x1a\n'
    result += chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
    result += chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result)


def frame_camera(model, camera):
    """Frame the scene from the model's default azimuth and elevation."""
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 1.2 * model.stat.extent


def run_viewer(scene, get_transforms, source, rate_hz=60.0, screenshot=None, screenshot_after=0.0):
    """GLFW render loop in the style of CODA_deisgn's viewer_ui.run_viewer."""
    model, data = scene.model, scene.data
    if not glfw.init():
        raise RuntimeError('Could not initialize GLFW; run from a desktop session')
    window, context = None, None
    try:
        window = glfw.create_window(1200, 800, 'Quest world poses - MuJoCo', None, None)
        if not window:
            raise RuntimeError('Could not create the MuJoCo window')
        glfw.set_window_size_limits(window, 720, 420, glfw.DONT_CARE, glfw.DONT_CARE)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mjv_scene = mujoco.MjvScene(model, maxgeom=max(3000, model.ngeom + 2000))
        option, camera = mujoco.MjvOption(), mujoco.MjvCamera()
        frame_camera(model, camera)
        last_cursor, camera_button = glfw.get_cursor_pos(window), None
        state = {'paused': False}

        def on_key(window, key, scancode, action, mods):
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_V:
                frame_camera(model, camera)
            elif key == glfw.KEY_T:
                scene.show_trail = not scene.show_trail
            elif key == glfw.KEY_C:
                scene.clear_trails()
            elif key == glfw.KEY_SPACE:
                state['paused'] = not state['paused']
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

        def on_button(window, button, action, mods):
            nonlocal camera_button, last_cursor
            last_cursor = glfw.get_cursor_pos(window)
            if action == glfw.PRESS:
                camera_button = button
            elif button == camera_button:
                camera_button = None

        def on_cursor(window, x, y):
            nonlocal last_cursor
            dx, dy = x - last_cursor[0], y - last_cursor[1]
            last_cursor = (x, y)
            if camera_button is None:
                return
            shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                     glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
            if camera_button == glfw.MOUSE_BUTTON_LEFT:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif camera_button == glfw.MOUSE_BUTTON_RIGHT:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM
            height = max(1, glfw.get_window_size(window)[1])
            mujoco.mjv_moveCamera(model, action, dx / height, dy / height, camera)

        def on_scroll(window, dx, dy):
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * dy, camera)

        glfw.set_key_callback(window, on_key)
        glfw.set_mouse_button_callback(window, on_button)
        glfw.set_cursor_pos_callback(window, on_cursor)
        glfw.set_scroll_callback(window, on_scroll)

        meter, poses, rate = RateMeter(), {}, 0.0
        opened = time.monotonic()
        while not glfw.window_should_close(window):
            start = time.monotonic()
            glfw.poll_events()
            if not state['paused']:
                transforms = get_transforms()
                rate = meter.update(transforms, start)
                poses = world_poses(transforms)
                scene.update(poses, start)
            window_size, framebuffer_size = glfw.get_window_size(window), glfw.get_framebuffer_size(window)
            if min(*window_size, *framebuffer_size) > 0:
                scale = framebuffer_size[0] / window_size[0]
                font_scale = int(np.clip(round(2 * scale) * 50, 100, 300))
                if context.fontScale != font_scale:
                    mujoco.mjr_changeFont(font_scale, context)
                viewport = mujoco.MjrRect(0, 0, *framebuffer_size)
                mujoco.mjv_updateScene(model, data, option, None, camera,
                                       mujoco.mjtCatBit.mjCAT_ALL, mjv_scene)
                scene.annotate(mjv_scene)
                mujoco.mjr_render(viewport, mjv_scene, context)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                   viewport, HELP, '', context)
                left, right = readout(poses, rate, state['paused'], source)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   viewport, left, right, context)
                if screenshot is not None and start - opened >= screenshot_after:
                    pixels = np.empty((framebuffer_size[1], framebuffer_size[0], 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(pixels, None, viewport, context)
                    save_png(screenshot, pixels[::-1])
                    print('Saved viewer screenshot: {}'.format(screenshot), flush=True)
                    screenshot = None
                glfw.swap_buffers(window)
            time.sleep(max(0.0, 1.0 / rate_hz - (time.monotonic() - start)))
    finally:
        if context is not None:
            context.free()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ip', default=None,
                        help='IP address of the headset for a network connection (default: USB)')
    parser.add_argument('--port', type=int, default=5555, help='port used with --ip')
    parser.add_argument('--rate', type=float, default=60.0, help='refresh rate in Hz')
    parser.add_argument('--axis-length', type=float, default=0.12,
                        help='length of each drawn coordinate axis in m')
    parser.add_argument('--floor', type=float, default=-1.3,
                        help='height of the floor relative to the world origin in m')
    parser.add_argument('--demo', action='store_true',
                        help='show synthetic motion instead of connecting to the headset')
    parser.add_argument('--screenshot', default=None,
                        help='save one PNG of the viewer (after --screenshot-after seconds)')
    parser.add_argument('--screenshot-after', type=float, default=3.0)
    args = parser.parse_args()

    scene = MujocoPoseScene(axis_length=args.axis_length, floor_height=args.floor)
    oculus_reader = None
    if args.demo:
        t0 = time.monotonic()
        last = {}

        def get_transforms():
            # a new dict per call stands in for a new headset sample
            last['sample'] = demo_transforms(time.monotonic() - t0)
            return last['sample']
        source = 'demo (synthetic motion, no headset)'
    else:
        from oculus_reader.reader import OculusReader
        oculus_reader = OculusReader(ip_address=args.ip, port=args.port)

        def get_transforms():
            return oculus_reader.get_transformations_and_buttons()[0]
        source = 'Quest over {}'.format('wifi ' + args.ip if args.ip else 'USB')

    print('Showing world-frame poses in MuJoCo. Close the window or press Esc to stop.')
    try:
        run_viewer(scene, get_transforms, source, rate_hz=args.rate,
                   screenshot=args.screenshot, screenshot_after=args.screenshot_after)
    except KeyboardInterrupt:
        pass
    finally:
        if oculus_reader is not None:
            oculus_reader.stop()


if __name__ == '__main__':
    main()
