"""Visualize the head and controller poses in the world frame with MuJoCo.

Each tracked body is drawn as an RGB coordinate triad (red = local x, green =
local y, blue = local z) carried by a MuJoCo mocap body, so the poses are shown
in a real 3D scene that can be orbited, zoomed and screenshotted with the
standard MuJoCo viewer controls. Untracked bodies fade out instead of freezing
in place.

Run it like `stream_world_poses.py`, whose world-frame conversion it reuses:

    python oculus_reader/visualize_world_poses_mujoco.py
    python oculus_reader/visualize_world_poses_mujoco.py --ip 10.0.32.101
"""
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from oculus_reader.reader import OculusReader
from oculus_reader.stream_world_poses import pose_lines, quaternion_from_matrix, world_poses

BODIES = ('head', 'left', 'right')
BODY_COLORS = {'head': '0.25 0.5 0.9', 'left': '0.2 0.7 0.3', 'right': '0.85 0.25 0.25'}
# rgba of the local x/y/z axis of every triad
AXIS_COLORS = ('0.9 0.25 0.2', '0.2 0.75 0.35', '0.25 0.5 0.9')

# OpenXR LOCAL space is Y-up with -Z forward, MuJoCo is Z-up with +X forward.
# This is a proper rotation (determinant +1), so the triads keep their handedness.
WORLD_TO_MUJOCO = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])


def _triad_geoms(axis_length, radius, alpha=1.0):
    """XML for the three axis capsules and the little sphere at their origin."""
    geoms = ['<geom type="sphere" size="{:.4f}" rgba="0.85 0.85 0.85 {:.2f}" '
             'contype="0" conaffinity="0"/>'.format(radius * 1.8, alpha)]
    for index, colour in enumerate(AXIS_COLORS):
        end = np.zeros(3)
        end[index] = axis_length
        geoms.append('<geom type="capsule" fromto="0 0 0 {:.4f} {:.4f} {:.4f}" '
                     'size="{:.4f}" rgba="{} {:.2f}" contype="0" conaffinity="0"/>'
                     .format(end[0], end[1], end[2], radius, colour, alpha))
    return '\n      '.join(geoms)


def build_scene_xml(axis_length=0.12, floor_height=-1.3):
    """Build the scene: a floor, a static world-origin triad and one mocap body each."""
    bodies = []
    for name in BODIES:
        bodies.append(
            '<body name="{}" mocap="true" pos="0 0 0">\n      {}\n    </body>'
            .format(name, _triad_geoms(axis_length, 0.006)))
    return """
<mujoco model="oculus_world_poses">
  <!-- ground, skybox and lighting taken from tidybot2's
       models/stanford_tidybot/scene.xml so the background matches that sim -->
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
  </visual>

  <!-- frame the free camera on the volume the head and hands move in -->
  <statistic center="0 0 -0.15" extent="1.6"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          pos="0 0 {floor:.4f}" contype="0" conaffinity="0"/>
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
    """Owns the model/data pair and pushes world-frame poses onto the mocap bodies."""

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

    def update(self, poses):
        """Place every tracked body, and fade out the ones that are not tracked."""
        for name in BODIES:
            pose = poses.get(name)
            alpha = 1.0 if pose is not None else 0.0
            for geom_id in self.geom_ids[name]:
                self.model.geom_rgba[geom_id, 3] = alpha
            if pose is None:
                continue
            position, quat = mujoco_pose(pose)
            self.data.mocap_pos[self.mocap_ids[name]] = position
            self.data.mocap_quat[self.mocap_ids[name]] = quat
        mujoco.mj_forward(self.model, self.data)


def camera_basis(cam):
    """Unit right/up vectors of a MuJoCo free camera, in world coordinates."""
    azimuth, elevation = np.radians(cam.azimuth), np.radians(cam.elevation)
    forward = -np.array([np.cos(elevation) * np.cos(azimuth),
                         np.cos(elevation) * np.sin(azimuth),
                         np.sin(elevation)])
    left = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(left)
    # looking straight down the world up axis leaves the horizontal axis undefined
    left = np.array([1.0, 0.0, 0.0]) if norm < 1e-6 else left / norm
    # cross(forward, world_up) comes out pointing screen-left for MuJoCo's
    # azimuth/elevation convention, so negate it to return a true screen-right
    return -left, np.cross(left, forward)


def hud_anchor(cam, offset_right=-0.62, offset_up=0.30):
    """Top-left corner of a readout that tracks the camera.

    Labels are drawn at a fixed pixel size but anchored in 3D, so the offsets
    are scaled by the camera distance to hold a roughly constant place on
    screen no matter how far the user has zoomed or orbited. The defaults are
    calibrated for a 16:9 window, where the block starts about 8% in from the
    left; a much wider window shifts it right, a much narrower one clips it.
    """
    right, up = camera_basis(cam)
    return np.asarray(cam.lookat) + cam.distance * (offset_right * right + offset_up * up)


def draw_readout(scn, lines, cam=None, anchor=(0.0, 0.0, 0.55), spacing=None):
    """Stack `lines` as billboarded text in the viewer, refreshed each sample.

    MuJoCo's passive viewer has no 2D overlay hook, so each line is carried by
    a fully transparent marker geom; only its label is drawn. The lines are
    spaced vertically so that long pose strings cannot overlap each other.

    Pass `cam` to pin the block to the camera like a HUD; otherwise it stays at
    the fixed world `anchor`. Geoms are appended at scn.ngeom, so the caller
    decides whether to clear the scene first. Never clear a scene that already
    holds the model's own geoms.
    """
    if cam is not None:
        origin = hud_anchor(cam)
        down = -camera_basis(cam)[1]
        step = (0.036 if spacing is None else spacing) * cam.distance
    else:
        origin = np.asarray(anchor, dtype=float)
        down = np.array([0.0, 0.0, -1.0])
        step = 0.085 if spacing is None else spacing

    for index, text in enumerate(lines):
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.001, 0.0, 0.0]),
                            origin + down * (index * step), np.eye(3).flatten(),
                            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        # mjvGeom.label is a fixed 100 byte buffer
        geom.label = text[:99]
        scn.ngeom += 1


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
    parser.add_argument('--labels', action='store_true', help='show body name labels')
    parser.add_argument('--no-readout', dest='readout', action='store_false',
                        help='hide the pose values drawn in the viewer')
    args = parser.parse_args()

    scene = MujocoPoseScene(axis_length=args.axis_length, floor_height=args.floor)
    oculus_reader = OculusReader(ip_address=args.ip, port=args.port)
    period = 1.0 / args.rate

    print('Showing world-frame poses in MuJoCo. Close the viewer window to stop.')
    try:
        with mujoco.viewer.launch_passive(scene.model, scene.data,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            if args.labels:
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY
            while viewer.is_running():
                transforms, _ = oculus_reader.get_transformations_and_buttons()
                poses = world_poses(transforms)
                with viewer.lock():
                    scene.update(poses)
                    if args.readout:
                        # user_scn holds only our own geoms, so it is safe to clear
                        viewer.user_scn.ngeom = 0
                        draw_readout(viewer.user_scn, pose_lines(poses), cam=viewer.cam)
                viewer.sync()
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        oculus_reader.stop()


if __name__ == '__main__':
    main()
