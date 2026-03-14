from __future__ import annotations

import argparse
from pathlib import Path

from audron.data.manifests import write_split
from audron.data.synthetic import synth_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate AUDRON synthetic drone dataset.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--duration-sec', type=float, default=3.0)
    parser.add_argument('--train-per-class', type=int, default=1000)
    parser.add_argument('--val-per-class', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    audio_dir = args.output_dir / 'audio'
    manifest = synth_dataset(
        out_dir=audio_dir,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        seed=args.seed,
    )
    write_split(manifest['train'], manifest['val'], args.output_dir / 'manifests')


if __name__ == '__main__':
    main()
