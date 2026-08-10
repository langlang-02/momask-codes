#!/usr/bin/env python3
"""Render side-by-side ground-truth and RVQ-reconstructed G1 trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from replay_g1 import (
    DEFAULT_MJCF,
    frame_to_qpos,
    load_motion,
    make_camera,
    qpos_addresses,
)


HEADER_HEIGHT = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a labelled side-by-side comparison of two 38-D G1 trajectories."
    )
    parser.add_argument("--reference_motion", type=Path, required=True, help="Ground-truth G1 .csv or .npy.")
    parser.add_argument("--prediction_motion", type=Path, required=True, help="RVQ reconstruction G1 .csv or .npy.")
    parser.add_argument("--output_mp4", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--input_fps", type=float, default=None, help="Shared input FPS; inferred by format when omitted.")
    parser.add_argument("--playback_fps", type=float, default=None, help="Output FPS; defaults to input FPS.")
    parser.add_argument("--root_euler_order", default="xyz")
    parser.add_argument("--clip_joint_limits", action="store_true")
    parser.add_argument("--camera_azimuth", type=float, default=50.0)
    parser.add_argument("--camera_elevation", type=float, default=-20.0)
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument(
        "--camera_lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed camera look-at point in metres. Defaults to the MJCF scene statistic center.",
    )
    parser.add_argument(
        "--follow_pelvis",
        action="store_true",
        help="Track the midpoint of the two pelvis bodies instead of keeping the camera fixed.",
    )
    parser.add_argument("--width", type=int, default=960, help="Width of each comparison panel.")
    parser.add_argument("--height", type=int, default=720, help="Height of each comparison panel.")
    parser.add_argument("--reference_label", default="Ground Truth")
    parser.add_argument("--prediction_label", default="RVQ Reconstruction")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def tint_robot(model: mujoco.MjModel, color: tuple[float, float, float]) -> None:
    """Tint robot geoms while leaving world geoms, including the floor, unchanged."""
    robot_geom_mask = model.geom_bodyid != 0
    model.geom_rgba[robot_geom_mask, :3] = color


def compose_panels(
    reference_pixels: np.ndarray,
    prediction_pixels: np.ndarray,
    frame_index: int,
    fps: float,
    reference_label: str,
    prediction_label: str,
) -> np.ndarray:
    panel_height, panel_width, _ = reference_pixels.shape
    image = Image.new("RGB", (panel_width * 2, panel_height + HEADER_HEIGHT), (16, 24, 36))
    image.paste(Image.fromarray(reference_pixels), (0, HEADER_HEIGHT))
    image.paste(Image.fromarray(prediction_pixels), (panel_width, HEADER_HEIGHT))

    draw = ImageDraw.Draw(image)
    draw.text((12, 12), reference_label, fill=(115, 185, 255))
    draw.text((panel_width + 12, 12), prediction_label, fill=(255, 166, 96))
    draw.text(
        (panel_width - 105, 12),
        f"frame {frame_index:04d}  {frame_index / fps:.2f}s",
        fill=(235, 239, 245),
    )
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    reference, reference_default_fps = load_motion(args.reference_motion, args.root_euler_order)
    prediction, prediction_default_fps = load_motion(args.prediction_motion, args.root_euler_order)
    if args.input_fps is None and reference_default_fps != prediction_default_fps:
        raise ValueError(
            "Input formats imply different frame rates; specify --input_fps after resampling the trajectories."
        )
    input_fps = args.input_fps or reference_default_fps
    playback_fps = args.playback_fps or input_fps
    if input_fps <= 0 or playback_fps <= 0:
        raise ValueError("--input_fps and --playback_fps must be positive")
    if not args.mjcf.is_file():
        raise FileNotFoundError(f"MJCF file not found: {args.mjcf}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")

    frame_count = min(len(reference), len(prediction))
    if frame_count == 0:
        raise ValueError("No frames to compare")
    if len(reference) != len(prediction):
        print(f"Warning: comparing the first {frame_count} frames of {len(reference)} and {len(prediction)} frames")

    reference_model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    prediction_model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    tint_robot(reference_model, (0.18, 0.45, 0.95))
    tint_robot(prediction_model, (0.95, 0.35, 0.08))
    reference_data = mujoco.MjData(reference_model)
    prediction_data = mujoco.MjData(prediction_model)
    reference_root_qposadr, reference_joint_qposadrs = qpos_addresses(reference_model)
    prediction_root_qposadr, prediction_joint_qposadrs = qpos_addresses(prediction_model)
    camera, reference_pelvis_id = make_camera(reference_model, args)
    prediction_pelvis_id = mujoco.mj_name2id(
        prediction_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    if prediction_pelvis_id < 0:
        raise ValueError("MJCF is missing the pelvis body required for camera tracking")
    fixed_lookat = (
        reference_model.stat.center.copy()
        if args.camera_lookat is None
        else np.asarray(args.camera_lookat, dtype=np.float64)
    )
    camera.lookat[:] = fixed_lookat

    print(
        f"Comparing {frame_count} frames at {input_fps:g} FPS: "
        f"{args.reference_motion} vs {args.prediction_motion}"
    )
    if args.dry_run:
        return

    reference_model.vis.global_.offwidth = max(reference_model.vis.global_.offwidth, args.width)
    reference_model.vis.global_.offheight = max(reference_model.vis.global_.offheight, args.height)
    prediction_model.vis.global_.offwidth = max(prediction_model.vis.global_.offwidth, args.width)
    prediction_model.vis.global_.offheight = max(prediction_model.vis.global_.offheight, args.height)
    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    reference_renderer = mujoco.Renderer(reference_model, height=args.height, width=args.width)
    prediction_renderer = mujoco.Renderer(prediction_model, height=args.height, width=args.width)
    try:
        with imageio.get_writer(args.output_mp4, fps=playback_fps, codec="libx264") as writer:
            for frame_index in range(frame_count):
                frame_to_qpos(
                    reference_model,
                    reference_data,
                    reference[frame_index],
                    reference_root_qposadr,
                    reference_joint_qposadrs,
                    args.clip_joint_limits,
                    frame_index,
                    input_fps,
                )
                frame_to_qpos(
                    prediction_model,
                    prediction_data,
                    prediction[frame_index],
                    prediction_root_qposadr,
                    prediction_joint_qposadrs,
                    args.clip_joint_limits,
                    frame_index,
                    input_fps,
                )
                if args.follow_pelvis:
                    camera.lookat[:] = (
                        reference_data.xpos[reference_pelvis_id]
                        + prediction_data.xpos[prediction_pelvis_id]
                    ) / 2.0
                else:
                    camera.lookat[:] = fixed_lookat
                reference_renderer.update_scene(reference_data, camera=camera)
                reference_pixels = reference_renderer.render()
                prediction_renderer.update_scene(prediction_data, camera=camera)
                prediction_pixels = prediction_renderer.render()
                writer.append_data(
                    compose_panels(
                        reference_pixels,
                        prediction_pixels,
                        frame_index,
                        input_fps,
                        args.reference_label,
                        args.prediction_label,
                    )
                )
    finally:
        reference_renderer.close()
        prediction_renderer.close()
    print(f"Wrote side-by-side comparison video to {args.output_mp4}")


if __name__ == "__main__":
    main()
