"""Small BVH reader for BONES-SEED SOMA files.

The existing repo BVH reader assumes uniform channel layouts. SOMA BVH files use
mixed 6-channel and 3-channel joints, so this parser keeps per-joint channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class BVHMotion:
    names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    channels: list[list[str]]
    frametime: float
    frames: np.ndarray


def _axis_rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unsupported rotation axis: {axis}")


def load_bvh(path: str | Path) -> BVHMotion:
    path = Path(path)
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    active = -1
    in_motion = False
    frame_count = None
    frametime = None
    motion_rows: list[list[float]] = []
    end_site_depth = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()

            if parts[0] == "MOTION":
                in_motion = True
                continue

            if not in_motion:
                if end_site_depth:
                    if parts[0] == "{":
                        end_site_depth = 1 if end_site_depth < 0 else end_site_depth + 1
                    elif parts[0] == "}":
                        end_site_depth -= 1
                    continue
                if parts[0] == "End" and len(parts) > 1 and parts[1] == "Site":
                    end_site_depth = -1
                    continue
                if parts[0] in {"ROOT", "JOINT"}:
                    parent = stack[-1] if stack else -1
                    names.append(parts[1])
                    parents.append(parent)
                    offsets.append([0.0, 0.0, 0.0])
                    channels.append([])
                    active = len(names) - 1
                    continue
                if parts[0] == "{":
                    if active >= 0:
                        stack.append(active)
                    continue
                if parts[0] == "}":
                    if stack:
                        stack.pop()
                    active = stack[-1] if stack else -1
                    continue
                if parts[0] == "OFFSET" and active >= 0:
                    offsets[active] = [float(v) for v in parts[1:4]]
                    continue
                if parts[0] == "CHANNELS" and active >= 0:
                    n_channels = int(parts[1])
                    channels[active] = parts[2:2 + n_channels]
                    continue
                continue

            if parts[0] == "Frames:":
                frame_count = int(parts[1])
                continue
            if parts[0] == "Frame" and parts[1] == "Time:":
                frametime = float(parts[2])
                continue
            motion_rows.append([float(v) for v in parts])

    if frame_count is not None and len(motion_rows) != frame_count:
        raise ValueError(f"{path}: expected {frame_count} frames, got {len(motion_rows)}")
    if frametime is None:
        raise ValueError(f"{path}: missing Frame Time")

    return BVHMotion(
        names=names,
        parents=np.asarray(parents, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.float64),
        channels=channels,
        frametime=frametime,
        frames=np.asarray(motion_rows, dtype=np.float64),
    )


def global_positions(motion: BVHMotion, scale: float = 0.01) -> np.ndarray:
    """Return global joint positions with shape (frames, joints, 3)."""
    n_frames = motion.frames.shape[0]
    n_joints = len(motion.names)
    out = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    local_rot = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)
    local_pos = np.zeros((n_frames, n_joints, 3), dtype=np.float64)

    cursor = 0
    for j, joint_channels in enumerate(motion.channels):
        values = motion.frames[:, cursor:cursor + len(joint_channels)]
        cursor += len(joint_channels)
        local_pos[:, j] = motion.offsets[j]
        rotations = np.repeat(np.eye(3, dtype=np.float64)[None], n_frames, axis=0)

        for c, channel in enumerate(joint_channels):
            axis = channel[0].upper()
            if channel.endswith("position"):
                local_pos[:, j, "XYZ".index(axis)] += values[:, c]
            elif channel.endswith("rotation"):
                for f in range(n_frames):
                    rotations[f] = rotations[f] @ _axis_rotation(axis, values[f, c])

        local_rot[:, j] = rotations

    global_rot = np.zeros_like(local_rot)
    for j in range(n_joints):
        parent = motion.parents[j]
        if parent < 0:
            global_rot[:, j] = local_rot[:, j]
            out[:, j] = local_pos[:, j]
        else:
            global_rot[:, j] = np.einsum("fij,fjk->fik", global_rot[:, parent], local_rot[:, j])
            out[:, j] = out[:, parent] + np.einsum("fij,fj->fi", global_rot[:, parent], local_pos[:, j])

    return out * scale


def downsample_indices(frametime: float, n_frames: int, target_fps: float) -> np.ndarray:
    source_fps = 1.0 / frametime
    if source_fps <= 0:
        raise ValueError(f"Invalid source FPS from frametime={frametime}")
    step = max(1, int(round(source_fps / target_fps)))
    return np.arange(0, n_frames, step, dtype=np.int64)


def select_joints(positions: np.ndarray, names: Iterable[str], mapping: dict[int, list[str]]) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(names)}
    selected = []
    missing = {}
    for target_idx in range(len(mapping)):
        candidates = mapping[target_idx]
        source_idx = next((name_to_idx[name] for name in candidates if name in name_to_idx), None)
        if source_idx is None:
            missing[target_idx] = candidates
            selected.append(np.zeros_like(positions[:, 0]))
        else:
            selected.append(positions[:, source_idx])
    if missing:
        details = "; ".join(f"{idx}: {cands}" for idx, cands in missing.items())
        raise KeyError(f"Missing required SOMA joints for target mapping: {details}")
    return np.stack(selected, axis=1)
