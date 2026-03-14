"""
Merge DroneAudioDataset binary manifests with DADS manifests for combined binary training.

Run after:
  1. prepare_real_data (produces data/processed/binary_no_aug/)
  2. prepare_dads (produces data/processed/dads/)

Output: data/processed/binary_combined/ with train.jsonl and val.jsonl containing rows from both.
Labels are aligned: 0 = noise/no_drone, 1 = drone.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from audron.data.manifests import write_split
from audron.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge DroneAudio binary + DADS manifests for combined binary training."
    )
    parser.add_argument(
        "--drone-audio-manifests",
        type=Path,
        default=Path("data/processed/binary_no_aug"),
        help="Dir containing train.jsonl and val.jsonl from prepare_real_data.",
    )
    parser.add_argument(
        "--dads-manifests",
        type=Path,
        default=Path("data/processed/dads/manifests"),
        help="Dir containing train.jsonl and val.jsonl from prepare_dads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/binary_combined"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    def load_manifest_dir(d: Path) -> tuple[list[dict], list[dict]]:
        train_path = d / "train.jsonl"
        val_path = d / "val.jsonl"
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(f"Expected train.jsonl and val.jsonl in {d}")
        return read_jsonl(train_path), read_jsonl(val_path)

    da_train, da_val = load_manifest_dir(args.drone_audio_manifests)
    dads_train_path = args.dads_manifests / "train.jsonl"
    dads_val_path = args.dads_manifests / "val.jsonl"
    if not dads_train_path.is_file() or not dads_val_path.is_file():
        raise FileNotFoundError(
            f"DADS manifests not found at {args.dads_manifests}. Run prepare_dads first."
        )
    dads_train, dads_val = read_jsonl(dads_train_path), read_jsonl(dads_val_path)

    # Normalize label_name so both use "noise" and "drone" (DADS may use "no_drone" -> "noise")
    def norm(row: dict) -> dict:
        r = dict(row)
        if r.get("label_name") == "no_drone":
            r["label_name"] = "noise"
        return r

    dads_train = [norm(r) for r in dads_train]
    dads_val = [norm(r) for r in dads_val]

    train_merged = da_train + dads_train
    val_merged = da_val + dads_val
    rng = random.Random(args.seed)
    rng.shuffle(train_merged)
    rng.shuffle(val_merged)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_split(train_merged, val_merged, out)
    print(f"Combined: train={len(train_merged)} (DroneAudio {len(da_train)} + DADS {len(dads_train)}), val={len(val_merged)}")
    print(f"Manifests: {out}/train.jsonl, {out}/val.jsonl")


if __name__ == "__main__":
    main()
