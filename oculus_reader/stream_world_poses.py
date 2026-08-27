"""Stream the head and controller poses in the world (tracking) frame.

The APK reports each controller relative to the head, plus the head pose itself
in the world frame under the key 'h'. The world-frame controller pose is
therefore T_world_head @ T_head_controller.

The world frame is OpenXR LOCAL space: gravity-aligned, +Y up, -Z forward, with
the origin at the head pose when the app started (or was last recentered).
"""
import argparse
import sys
import time

import numpy as np

from oculus_reader.reader import OculusReader

# reader key -> human readable name
CONTROLLERS = (('l', 'left'), ('r', 'right'))


def quaternion_from_matrix(matrix):
    """Return the (x, y, z, w) quaternion of a 4x4 homogeneous transform."""
    m = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def world_poses(transforms):
    """Convert one reader sample into world-frame 4x4 poses.

    Returns a dict with the keys 'head', 'left' and 'right' that are currently
    tracked. Returns an empty dict while the head pose is unavailable, since
    nothing can be placed in the world frame without it.
    """
    if not transforms or 'h' not in transforms:
        return {}
    head = transforms['h']
    poses = {'head': head}
    for key, name in CONTROLLERS:
        if key in transforms:
            poses[name] = head @ transforms[key]
    return poses


def format_pose(name, pose):
    position = pose[:3, 3]
    quat = quaternion_from_matrix(pose)
    return ('{:<6} pos [{:+.4f} {:+.4f} {:+.4f}]  quat [{:+.4f} {:+.4f} {:+.4f} {:+.4f}]'
            .format(name, *position, *quat))


def pose_lines(poses):
    """Render one sample as the lines shown by this script and by the viewers."""
    if not poses:
        return ['waiting for the head pose (put the headset on)']
    lines = [format_pose(name, poses[name])
             for name in ('head', 'left', 'right') if name in poses]
    for _, name in CONTROLLERS:
        if name not in poses:
            lines.append('{:<6} not tracked'.format(name))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ip', default=None,
                        help='IP address of the headset for a network connection (default: USB)')
    parser.add_argument('--port', type=int, default=5555, help='port used with --ip')
    parser.add_argument('--rate', type=float, default=20.0, help='print rate in Hz')
    args = parser.parse_args()

    oculus_reader = OculusReader(ip_address=args.ip, port=args.port)
    period = 1.0 / args.rate
    # redraw in place on a terminal, otherwise emit plain lines for piping to a file
    redraw = sys.stdout.isatty()
    previous_lines = 0

    print('Streaming world-frame poses. Press Ctrl+C to stop.')
    try:
        while True:
            time.sleep(period)
            transforms, _ = oculus_reader.get_transformations_and_buttons()
            lines = pose_lines(world_poses(transforms))
            if redraw and previous_lines:
                sys.stdout.write('\033[{}A\033[J'.format(previous_lines))
            print('\n'.join(lines))
            previous_lines = len(lines)
    except KeyboardInterrupt:
        pass
    finally:
        oculus_reader.stop()


if __name__ == '__main__':
    main()
