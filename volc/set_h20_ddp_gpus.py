#!/usr/bin/env python3
"""Update H20 V2 MoMask training YAMLs for a supported single-node GPU count."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


CONFIGS = (
    ("volc/H20/train_bones_g1_rvq_H20.yaml", "RVQ_BATCH_SIZE"),
    ("volc/H20/train_bones_g1_mask_H20.yaml", "MASK_BATCH_SIZE"),
    ("volc/H20/train_bones_g1_res_H20.yaml", "RES_BATCH_SIZE"),
)

RESOURCE_PRESETS = {
    1: {"Cpu": 96, "MemoryGiB": 1000, "GpuCount": 1},
    8: {"Cpu": 160, "MemoryGiB": 1600, "GpuCount": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
        choices=(1, 8),
        help="H20 queue GPU count. Verified with mlp job create: 1 and 8.",
    )
    parser.add_argument(
        "--global-rvq-batch",
        type=int,
        default=256,
        help="Global RVQ batch size to preserve when using DDP.",
    )
    parser.add_argument(
        "--global-transformer-batch",
        type=int,
        default=64,
        help="Global Mask/Residual Transformer batch size to preserve when using DDP.",
    )
    return parser.parse_args()


def set_env(data: dict, name: str, value: object) -> None:
    envs = data.setdefault("RuntimeConfig", {}).setdefault("Envs", [])
    for item in envs:
        if item.get("Name") == name:
            item["Value"] = str(value)
            item.setdefault("IsPrivate", False)
            return
    envs.append({"Name": name, "Value": str(value), "IsPrivate": False})


def per_gpu_batch(global_batch: int, gpus: int) -> int:
    if global_batch % gpus != 0:
        raise SystemExit(f"global batch {global_batch} is not divisible by {gpus} GPUs")
    return global_batch // gpus


def update_config(path: Path, gpus: int, batch_env: str, global_batch: int) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    role = data["ResourceConfig"]["Roles"][0]
    role["Replicas"] = 1
    resource = role.setdefault("Resource", {})
    resource["Type"] = "Flexible"
    resource["InstanceTypeId"] = ""
    resource["ZoneId"] = "cn-beijing-c"
    claim = resource.setdefault("FlexibleResourceClaim", {})
    claim.update(RESOURCE_PRESETS[gpus])
    claim["GpuType"] = "NVIDIA-H20-SXM5-96GB"
    claim["Family"] = "ml.pni3l"
    claim["RdmaEniCount"] = 0

    set_env(data, "DDP", "1" if gpus > 1 else "0")
    set_env(data, "NPROC_PER_NODE", gpus)
    set_env(data, "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(gpus)))
    set_env(data, "OMP_NUM_THREADS", max(4, 64 // gpus))
    set_env(data, "MKL_NUM_THREADS", max(4, 64 // gpus))
    set_env(data, batch_env, per_gpu_batch(global_batch, gpus))

    suffix = f"h20-{'ddp' + str(gpus) if gpus > 1 else '1gpu'}"
    task_name = data.get("Name", path.stem)
    for old in ("h20-ddp8", "h20-ddp1", "h20-1gpu"):
        task_name = task_name.replace(old, suffix)
    data["Name"] = task_name

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)
    print(f"{path}: gpus={gpus} {batch_env}={per_gpu_batch(global_batch, gpus)}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    for rel_path, batch_env in CONFIGS:
        global_batch = (
            args.global_rvq_batch
            if batch_env == "RVQ_BATCH_SIZE"
            else args.global_transformer_batch
        )
        update_config(root / rel_path, args.gpus, batch_env, global_batch)


if __name__ == "__main__":
    main()
