#!/usr/bin/env bash
# Train on real DroneAudioDataset. Run scripts/run_real_data.sh first.
# Usage: ./scripts/train_real.sh <binary_no_aug|binary_augmented|multiclass_real>
set -euo pipefail

TASK="${1:-}"
PROCESSED_DIR="data/processed"

case "$TASK" in
  binary_no_aug)
    python -m audron.scripts.train \
      --config configs/paper/binary_no_aug.yaml \
      --train-manifest "${PROCESSED_DIR}/binary_no_aug/train.jsonl" \
      --val-manifest "${PROCESSED_DIR}/binary_no_aug/val.jsonl" \
      --output-dir runs/binary_no_aug
    ;;
  binary_augmented)
    python -m audron.scripts.train \
      --config configs/paper/binary_augmented.yaml \
      --train-manifest "${PROCESSED_DIR}/binary_no_aug/train.jsonl" \
      --val-manifest "${PROCESSED_DIR}/binary_no_aug/val.jsonl" \
      --output-dir runs/binary_augmented
    ;;
  multiclass_real)
    python -m audron.scripts.train \
      --config configs/paper/multiclass_real.yaml \
      --train-manifest "${PROCESSED_DIR}/multiclass_real/train.jsonl" \
      --val-manifest "${PROCESSED_DIR}/multiclass_real/val.jsonl" \
      --output-dir runs/multiclass_real
    ;;
    *)
    echo "Usage: $0 binary_no_aug|binary_augmented|multiclass_real"
    echo "  binary_no_aug     - drone vs noise (2 classes)"
    echo "  binary_augmented  - same + on-the-fly SOTA augmentation"
    echo "  multiclass_real  - noise, bebop, membo (3 classes)"
    exit 1
    ;;
esac
