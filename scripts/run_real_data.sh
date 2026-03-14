#!/usr/bin/env bash
# Clone DroneAudioDataset (https://github.com/saraalemadi/DroneAudioDataset) and prepare
# train/val manifests for binary and multiclass experiments.
set -euo pipefail

REPO_URL="https://github.com/saraalemadi/DroneAudioDataset.git"
RAW_DIR="data/raw"
DATASET_DIR="${RAW_DIR}/DroneAudioDataset"
PROCESSED_DIR="data/processed"

mkdir -p "$RAW_DIR"
if [[ ! -d "${DATASET_DIR}/.git" ]]; then
  echo "Cloning DroneAudioDataset into ${DATASET_DIR} ..."
  git clone --depth 1 "$REPO_URL" "$DATASET_DIR"
else
  echo "DroneAudioDataset already present at ${DATASET_DIR}"
fi

echo "Building manifests (binary_no_aug, multiclass_real) ..."
python -m audron.scripts.prepare_real_data \
  --drone-audio-root "$DATASET_DIR" \
  --output-dir "$PROCESSED_DIR"

echo "Done. Manifests:"
echo "  binary_no_aug:  ${PROCESSED_DIR}/binary_no_aug/{train,val}.jsonl"
echo "  multiclass_real: ${PROCESSED_DIR}/multiclass_real/{train,val}.jsonl"
echo ""
echo "Train binary (drone vs noise):"
echo "  python -m audron.scripts.train --config configs/paper/binary_no_aug.yaml \\"
echo "    --train-manifest ${PROCESSED_DIR}/binary_no_aug/train.jsonl \\"
echo "    --val-manifest ${PROCESSED_DIR}/binary_no_aug/val.jsonl \\"
echo "    --output-dir runs/binary_no_aug"
echo ""
echo "Train multiclass (noise, bebop, membo):"
echo "  python -m audron.scripts.train --config configs/paper/multiclass_real.yaml \\"
echo "    --train-manifest ${PROCESSED_DIR}/multiclass_real/train.jsonl \\"
echo "    --val-manifest ${PROCESSED_DIR}/multiclass_real/val.jsonl \\"
echo "    --output-dir runs/multiclass_real"
