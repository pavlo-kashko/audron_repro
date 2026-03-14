# Verification log

This file records what was actually executed in the build environment for this reproduction package.

## Executed successfully

1. Installed the repo in editable mode:
   - `pip install -e .`

2. Smoke forward test on the tiny config:
   - `python -m audron.scripts.inspect_model --config configs/smoke/tiny_synthetic.yaml`

3. Generated synthetic smoke data:
   - `python -m audron.scripts.prepare_synthetic_data --output-dir data/smoke_synth --sample-rate 8000 --duration-sec 0.5 --train-per-class 4 --val-per-class 2`

4. Ran smoke training for 1 epoch:
   - `python -m audron.scripts.train --config configs/smoke/tiny_synthetic.yaml --train-manifest data/smoke_synth/manifests/train.jsonl --val-manifest data/smoke_synth/manifests/val.jsonl --output-dir runs/smoke4 --override train.epochs=1`

5. Ran smoke evaluation:
   - `python -m audron.scripts.evaluate --config configs/smoke/tiny_synthetic.yaml --manifest data/smoke_synth/manifests/val.jsonl --checkpoint runs/smoke4/best.pt --output-dir runs/smoke4/eval`

6. Ran pytest smoke test:
   - `pytest -q tests/smoke_test.py`

7. Instantiated the paper-sized model and verified a forward pass with batch size 1:
   - `python -m audron.scripts.inspect_model --config configs/paper/synthetic.yaml --batch-size 1`
   - Observed shapes: logits `(1, 4)`, MFCC `(1, 128)`, STFT `(1, 256)`, RNN `(1, 192)`, autoencoder `(1, 160)`, reconstruction `(1, 48000)`.
   - Parameter count: `205,589,413`.

## Not executed here

1. External dataset downloads for DroneAudioDataset, ESC-50, Speech Commands background noise, and DroneNoise Database.
2. End-to-end paper-scale training on the full real datasets.
3. Reproduction of the exact paper accuracy numbers.
4. Baseline CNN / RNN / CRNN reimplementation from reference [5].

## Bugs fixed during build

1. Fixed malformed JSONL writer line endings in `audron.utils.io`.
2. Fixed malformed newline handling in `audron.scripts.download_helpers`.
3. Fixed deadlock / severe slowdown in repeated STFT branch usage by explicitly setting `torch_num_threads: 1` in configs and scripts.
4. Fixed relative-path manifest loading by resolving dataset paths more robustly and writing absolute paths in generated manifests.
5. Removed progress-bar behavior that interfered with non-interactive validation logs.
