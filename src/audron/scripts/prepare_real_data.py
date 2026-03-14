from __future__ import annotations

import argparse
from pathlib import Path

from audron.data.manifests import make_records, train_val_split, write_split


def build_binary_records(root: Path) -> list[dict]:
    yes_paths = list((root / 'Binary_Drone_Audio' / 'yes_drone').rglob('*.wav'))
    unknown_paths = list((root / 'Binary_Drone_Audio' / 'unknown').rglob('*.wav'))
    rows = []
    rows.extend(make_records(yes_paths, 'drone', 1))
    rows.extend(make_records(unknown_paths, 'noise', 0))
    return rows


def build_multiclass_records(root: Path) -> list[dict]:
    rows = []
    rows.extend(make_records((root / 'Multiclass_Drone_Audio' / 'unknown').rglob('*.wav'), 'noise', 0))
    rows.extend(make_records((root / 'Multiclass_Drone_Audio' / 'bebop_1').rglob('*.wav'), 'bebop', 1))
    rows.extend(make_records((root / 'Multiclass_Drone_Audio' / 'membo_1').rglob('*.wav'), 'membo', 2))
    return rows


def add_drone_noise_augmentation(records: list[dict], drone_noise_root: Path) -> list[dict]:
    extra = list(drone_noise_root.rglob('*.wav'))
    augmented = records[:]
    augmented.extend(make_records(extra, 'drone', 1))
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare manifests for AUDRON real-data experiments.')
    parser.add_argument('--drone-audio-root', type=Path, required=True, help='Root of saraalemadi/DroneAudioDataset repository.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--drone-noise-root', type=Path, default=None, help='Optional DroneNoise folder for binary augmentation experiment.')
    args = parser.parse_args()

    out = args.output_dir
    binary = build_binary_records(args.drone_audio_root)
    train_rows, val_rows = train_val_split(binary, val_fraction=args.val_fraction, seed=args.seed)
    write_split(train_rows, val_rows, out / 'binary_no_aug')

    if args.drone_noise_root is not None:
        binary_aug = add_drone_noise_augmentation(binary, args.drone_noise_root)
        train_rows, val_rows = train_val_split(binary_aug, val_fraction=args.val_fraction, seed=args.seed)
        write_split(train_rows, val_rows, out / 'binary_with_aug')

    multiclass = build_multiclass_records(args.drone_audio_root)
    train_rows, val_rows = train_val_split(multiclass, val_fraction=args.val_fraction, seed=args.seed)
    write_split(train_rows, val_rows, out / 'multiclass_real')


if __name__ == '__main__':
    main()
