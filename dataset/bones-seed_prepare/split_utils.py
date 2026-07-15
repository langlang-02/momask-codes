"""Deterministic dataset split helpers."""

from __future__ import annotations

import random
from pathlib import Path


def split_ids(ids: list[str], seed: int, val_ratio: float, test_ratio: float) -> tuple[list[str], list[str], list[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    test_ids = sorted(ids[:n_test])
    val_ids = sorted(ids[n_test:n_test + n_val])
    train_ids = sorted(ids[n_test + n_val:])
    return train_ids, val_ids, test_ids


def write_split_files(output_root: Path, ids: list[str], seed: int, val_ratio: float, test_ratio: float) -> None:
    train_ids, val_ids, test_ids = split_ids(ids, seed, val_ratio, test_ratio)
    for name, split in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with (output_root / f"{name}.txt").open("w", encoding="utf-8") as f:
            for item in split:
                f.write(f"{item}\n")
