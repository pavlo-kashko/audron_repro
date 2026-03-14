#!/usr/bin/env bash
set -euo pipefail

python -m audron.scripts.prepare_synthetic_data --output-dir data/synthetic
python -m audron.scripts.train \
  --config configs/paper/synthetic.yaml \
  --train-manifest data/synthetic/manifests/train.jsonl \
  --val-manifest data/synthetic/manifests/val.jsonl \
  --output-dir runs/synthetic
