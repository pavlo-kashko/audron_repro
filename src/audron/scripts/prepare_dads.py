"""
Prepare the DADS (Drone Audio Detection Samples) dataset from Hugging Face for binary training.

Dataset: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
- 0 = no drone (noise), 1 = drone (same as our binary_no_aug)
- 16 kHz, 16-bit, mono, WAV; duration 0.5 s to several minutes
- ~180k samples (163k drone, 17k no-drone)

Requires: pip install datasets[audio] soundfile

Usage:
  python -m audron.scripts.prepare_dads --output-dir data/processed/dads
  python -m audron.scripts.prepare_dads --output-dir data/processed/dads --max-per-class 20000 --val-fraction 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from audron.data.manifests import train_val_split, write_split
from audron.utils.io import ensure_dir


DADS_HF_REPO = "geronimobasso/drone-audio-detection-samples"
TARGET_SR = 16000
LABEL_NAMES = ("noise", "drone")  # 0 = no drone, 1 = drone


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DADS from Hugging Face, save WAVs, and write train/val manifests."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap samples per class (e.g. 20000) to limit disk and balance with DroneAudio.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass to load_dataset if required by the dataset.",
    )
    args = parser.parse_args()

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        raise SystemExit("Install Hugging Face datasets: pip install 'datasets[audio]'")

    out_dir = ensure_dir(args.output_dir)
    audio_dir = ensure_dir(out_dir / "audio" / "no_drone")
    ensure_dir(out_dir / "audio" / "drone")

    print("Loading DADS from Hugging Face (this may download ~6.8 GB on first run)...")
    ds = load_dataset(
        DADS_HF_REPO,
        split="train",
        trust_remote_code=args.trust_remote_code,
    )
    # Cast audio column so each row["audio"] decodes to {"array": ndarray, "sampling_rate": int}
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    # Collect indices per class for stratified subsampling (DADS: 0 = no drone, 1 = drone)
    label_col = "label" if "label" in ds.column_names else "class"
    if label_col not in ds.column_names:
        raise RuntimeError(f"Expected column 'label' or 'class' in DADS; got {ds.column_names}")

    def to_label_id(x) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        return int(str(x).strip())

    indices_by_class = {0: [], 1: []}
    for i in range(len(ds)):
        lab = to_label_id(ds[i][label_col])
        if lab not in indices_by_class:
            indices_by_class[lab] = []
        indices_by_class[lab].append(i)

    # Subsample if requested (recommended for first run: e.g. --max-per-class 20000)
    rng = np.random.default_rng(args.seed)
    for k in indices_by_class:
        idx = indices_by_class[k]
        if args.max_per_class is not None and len(idx) > args.max_per_class:
            idx = rng.choice(idx, size=args.max_per_class, replace=False).tolist()
        indices_by_class[k] = idx

    # Flatten to a single list of (idx, label_id) to preserve stratification when writing
    flat = []
    for label_id, indices in indices_by_class.items():
        for idx in indices:
            flat.append((int(idx), label_id))
    rng.shuffle(flat)

    records = []
    total = len(flat)
    written = 0

    for idx, label_id in flat:
        name = LABEL_NAMES[label_id]
        subdir = out_dir / "audio" / name
        subdir.mkdir(parents=True, exist_ok=True)
        row = ds[idx]
        audio = row["audio"]
        # HF Audio column can be dict or AudioDecoder-like; extract array and sampling_rate
        if isinstance(audio, dict):
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", TARGET_SR))
        elif hasattr(audio, "get") and callable(audio.get):
            arr = np.asarray(audio.get("array"), dtype=np.float32)
            sr = int(audio.get("sampling_rate", TARGET_SR))
        elif hasattr(audio, "array"):
            arr = np.asarray(audio.array, dtype=np.float32)
            sr = int(getattr(audio, "sampling_rate", TARGET_SR))
        elif hasattr(audio, "__getitem__"):
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", TARGET_SR) if hasattr(audio, "get") else TARGET_SR)
        else:
            raise TypeError(f"Unexpected audio type: {type(audio)}; use datasets with decode=True for audio column")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr.astype(np.float64), orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        path = subdir / f"dads_{label_id}_{idx:06d}.wav"
        sf.write(path, arr, TARGET_SR)
        records.append({"path": str(path.resolve()), "label_name": name, "label_id": label_id})
        written += 1
        if written % 5000 == 0:
            print(f"  Written {written}/{total}...")

    print(f"Wrote {len(records)} WAVs. Splitting train/val...")
    train_rows, val_rows = train_val_split(records, val_fraction=args.val_fraction, seed=args.seed)
    write_split(train_rows, val_rows, out_dir / "manifests")
    print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}")
    print(f"Manifests: {out_dir / 'manifests'}/train.jsonl, val.jsonl")


if __name__ == "__main__":
    main()
