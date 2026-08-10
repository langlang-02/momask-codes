#!/usr/bin/env python3
"""Kinematically replay BONES-SEED / MoMask Unitree G1 trajectories in MuJoCo.

The script sets qpos directly and calls ``mj_forward``.  It deliberately does
not call ``mj_step`` or apply actuator controls, so the floor is visual only
and contact dynamics do not affect the reconstructed motion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_DIR = REPO_ROOT / "dataset" / "bones-seed_prepare"
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from g1_io import (  # noqa: E402
    G1_FEATURE_COLUMNS,
    G1_FEATURE_DIM,
    G1_JOINT_SLICE,
    G1_ROOT_ROTATION_SLICE,
    csv_values_to_model_features,
    read_g1_csv,
    rotation_6d_to_matrix,
)


DEFAULT_MJCF = (
    REPO_ROOT.parent
    / "unitree_mujoco"
    / "unitree_robots"
    / "g1"
    / "scene_29dof.xml"
)
ROOT_JOINT_NAME = "floating_base_joint"
JOINT_NAMES = tuple(column.removesuffix("_dof") for column in G1_FEATURE_COLUMNS[6:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinematic MuJoCo replay for 38-D native Unitree G1 trajectories."
    )
    parser.add_argument(
        "--motion",
        type=Path,
        required=True,
        help="Input .csv (BONES-SEED cm/degree, normally 120 Hz) or .npy (MoMask m/radian, normally 20 Hz).",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=DEFAULT_MJCF,
        help="29-DoF G1 scene XML. Override this if unitree_mujoco is elsewhere.",
    )
    parser.add_argument(
        "--input_fps",
        type=float,
        default=None,
        help="Input frame rate. Defaults to 120 for CSV and 20 for NPY.",
    )
    parser.add_argument(
        "--playback_fps",
        type=float,
        default=None,
        help="Viewer/video frame rate. Defaults to --input_fps.",
    )
    parser.add_argument(
        "--root_euler_order",
        type=str,
        default="xyz",
        help="CSV root_rotateX/Y/Z convention; lower-case is extrinsic, upper-case intrinsic.",
    )
    parser.add_argument(
        "--output_mp4",
        type=Path,
        default=None,
        help="Render an MP4 instead of opening the interactive viewer. Set MUJOCO_GL=egl on headless GPU nodes.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output MP4 width.")
    parser.add_argument("--height", type=int, default=720, help="Output MP4 height.")
    parser.add_argument(
        "--orbit_degrees",
        type=float,
        default=0.0,
        help="Total camera azimuth change in degrees over the trajectory; e.g. 180. Use a negative value for the opposite direction.",
    )
    parser.add_argument(
        "--camera_azimuth",
        type=float,
        default=None,
        help="Initial camera azimuth in degrees. Defaults to the MJCF visual global setting.",
    )
    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=None,
        help="Camera elevation in degrees. Defaults to the MJCF visual global setting.",
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=None,
        help="Camera distance in metres. Defaults to 1.5 times the MJCF scene extent.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the trajectory in the interactive viewer.")
    parser.add_argument(
        "--clip_joint_limits",
        action="store_true",
        help="Clamp commanded joint angles to the MJCF limits. Disabled by default for faithful reconstruction inspection.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate the XML, motion shape and joint-name mapping without opening a viewer or rendering a video.",
    )
    return parser.parse_args()


def load_motion(path: Path, root_euler_order: str = "xyz") -> tuple[np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return csv_values_to_model_features(read_g1_csv(path), root_euler_order=root_euler_order), 120.0
    if suffix == ".npy":
        features = np.load(path).astype(np.float32)
        if features.ndim != 2 or features.shape[1] != G1_FEATURE_DIM:
            raise ValueError(f"{path} must have shape (frames, {G1_FEATURE_DIM}), got {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError(f"{path} contains non-finite values")
        return features, 20.0
    raise ValueError(f"Unsupported input format {path.suffix!r}; use a .csv or .npy trajectory")


def qpos_addresses(model: mujoco.MjModel) -> tuple[int, np.ndarray]:
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT_NAME)
    if root_id < 0:
        raise ValueError(f"MJCF is missing required free joint {ROOT_JOINT_NAME!r}")
    if model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{ROOT_JOINT_NAME!r} must be a free joint")

    addresses = []
    missing = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            missing.append(name)
            continue
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name!r} must be a hinge joint in the supplied MJCF")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    if missing:
        raise ValueError(f"MJCF is missing G1 joints: {', '.join(missing)}")
    return int(model.jnt_qposadr[root_id]), np.asarray(addresses, dtype=np.int32)


def frame_to_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: np.ndarray,
    root_qposadr: int,
    joint_qposadrs: np.ndarray,
    clip_joint_limits: bool,
    frame_index: int,
    fps: float,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    data.ctrl[:] = 0
    data.time = frame_index / fps

    quaternion_xyzw = Rotation.from_matrix(rotation_6d_to_matrix(frame[G1_ROOT_ROTATION_SLICE])).as_quat()
    data.qpos[root_qposadr : root_qposadr + 3] = frame[:3]
    data.qpos[root_qposadr + 3 : root_qposadr + 7] = quaternion_xyzw[[3, 0, 1, 2]]

    joint_values = frame[G1_JOINT_SLICE].copy()
    if clip_joint_limits:
        for index, qposadr in enumerate(joint_qposadrs):
            joint_id = np.flatnonzero(model.jnt_qposadr == qposadr)[0]
            if model.jnt_limited[joint_id]:
                joint_values[index] = np.clip(joint_values[index], *model.jnt_range[joint_id])
    data.qpos[joint_qposadrs] = joint_values
    mujoco.mj_forward(model, data)


def make_camera(model: mujoco.MjModel, args: argparse.Namespace) -> tuple[mujoco.MjvCamera, int]:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = model.vis.global_.azimuth if args.camera_azimuth is None else args.camera_azimuth
    camera.elevation = (
        model.vis.global_.elevation if args.camera_elevation is None else args.camera_elevation
    )
    camera.distance = model.stat.extent * 1.5 if args.camera_distance is None else args.camera_distance

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if body_id < 0:
        raise ValueError("MJCF is missing the pelvis body required for camera tracking")
    return camera, body_id


def update_camera(
    camera: mujoco.MjvCamera,
    data: mujoco.MjData,
    pelvis_body_id: int,
    frame_index: int,
    frame_count: int,
    initial_azimuth: float,
    orbit_degrees: float,
) -> None:
    progress = frame_index / max(1, frame_count - 1)
    camera.azimuth = initial_azimuth + orbit_degrees * progress
    camera.lookat[:] = data.xpos[pelvis_body_id]


def replay_in_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    features: np.ndarray,
    input_fps: float,
    playback_fps: float,
    root_qposadr: int,
    joint_qposadrs: np.ndarray,
    args: argparse.Namespace,
) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False) as viewer:
        camera, pelvis_body_id = make_camera(model, args)
        initial_azimuth = camera.azimuth
        frame_index = 0
        while viewer.is_running():
            start_time = time.perf_counter()
            with viewer.lock():
                frame_to_qpos(
                    model,
                    data,
                    features[frame_index],
                    root_qposadr,
                    joint_qposadrs,
                    args.clip_joint_limits,
                    frame_index,
                    input_fps,
                )
                update_camera(
                    camera,
                    data,
                    pelvis_body_id,
                    frame_index,
                    len(features),
                    initial_azimuth,
                    args.orbit_degrees,
                )
                viewer.cam.type = camera.type
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
                viewer.cam.distance = camera.distance
                viewer.cam.lookat[:] = camera.lookat
            viewer.sync()

            frame_index += 1
            if frame_index == len(features):
                if not args.loop:
                    break
                frame_index = 0
            time.sleep(max(0.0, 1.0 / playback_fps - (time.perf_counter() - start_time)))


def render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    features: np.ndarray,
    input_fps: float,
    playback_fps: float,
    root_qposadr: int,
    joint_qposadrs: np.ndarray,
    args: argparse.Namespace,
) -> None:
    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    # scene_29dof.xml defaults to a 640x480 framebuffer.  Resize the compiled
    # model at runtime instead of modifying the shared Unitree XML asset.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera, pelvis_body_id = make_camera(model, args)
    initial_azimuth = camera.azimuth
    try:
        with imageio.get_writer(args.output_mp4, fps=playback_fps, codec="libx264") as writer:
            for frame_index, frame in enumerate(features):
                frame_to_qpos(
                    model,
                    data,
                    frame,
                    root_qposadr,
                    joint_qposadrs,
                    args.clip_joint_limits,
                    frame_index,
                    input_fps,
                )
                update_camera(
                    camera,
                    data,
                    pelvis_body_id,
                    frame_index,
                    len(features),
                    initial_azimuth,
                    args.orbit_degrees,
                )
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        renderer.close()
    print(f"Wrote {len(features)} frames at {playback_fps:g} FPS to {args.output_mp4}")


def main() -> None:
    args = parse_args()
    features, default_fps = load_motion(args.motion, args.root_euler_order)
    input_fps = args.input_fps or default_fps
    playback_fps = args.playback_fps or input_fps
    if input_fps <= 0 or playback_fps <= 0:
        raise ValueError("--input_fps and --playback_fps must be positive")
    if not args.mjcf.is_file():
        raise FileNotFoundError(f"MJCF file not found: {args.mjcf}")

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    root_qposadr, joint_qposadrs = qpos_addresses(model)
    print(
        f"Loaded {len(features)} frames from {args.motion} at {input_fps:g} FPS; "
        f"MJCF={args.mjcf}, nq={model.nq}, mapped joints={len(joint_qposadrs)}"
    )

    frame_to_qpos(
        model,
        data,
        features[0],
        root_qposadr,
        joint_qposadrs,
        args.clip_joint_limits,
        0,
        input_fps,
    )
    if args.dry_run:
        print("Kinematic mapping validation passed; no viewer or physics step was run.")
        return

    if args.output_mp4 is not None:
        render_video(
            model,
            data,
            features,
            input_fps,
            playback_fps,
            root_qposadr,
            joint_qposadrs,
            args,
        )
    else:
        replay_in_viewer(
            model,
            data,
            features,
            input_fps,
            playback_fps,
            root_qposadr,
            joint_qposadrs,
            args,
        )


if __name__ == "__main__":
    main()
