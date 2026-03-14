#!/usr/bin/env bash
set -euo pipefail

python -m audron.scripts.prepare_synthetic_data \
  --output-dir data/smoke_synth \
  --sample-rate 8000 \
  --duration-sec 0.5 \
  --train-per-class 4 \
  --val-per-class 2

python -m audron.scripts.train \
  --config configs/smoke/tiny_synthetic.yaml \
  --train-manifest data/smoke_synth/manifests/train.jsonl \
  --val-manifest data/smoke_synth/manifests/val.jsonl \
  --output-dir runs/smoke
