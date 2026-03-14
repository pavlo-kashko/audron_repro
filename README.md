# AUDRON reproduction (PyTorch)

This repository is a clean-room PyTorch reproduction of **AUDRON: A Deep Learning Framework with Fused Acoustic Signatures for Drone Type Recognition**.

The code is organized like a small research repo and aims to stay close to the paper while making the places where the paper is underspecified explicit.

## What this repo contains

- A four-branch AUDRON implementation:
  - **MFCC + 1D CNN** -> 128-d features
  - **STFT / spectrogram CNN** -> 256-d features
  - **BiLSTM + attention** -> 192-d features
  - **Autoencoder** -> 160-d latent embedding
- Feature fusion into a **736-d** concatenated vector and a dense fusion / classification head.
- Synthetic drone audio generation for the four drone classes described in the paper.
- Real-data manifest preparation for:
  - binary classification without augmentation
  - binary classification with extra drone clips from DroneNoise Database
  - multiclass classification on Bebop / Membo / Noise
- Training and evaluation scripts with:
  - AdamW
  - ReduceLROnPlateau
  - early stopping
  - combined CE + weighted reconstruction MSE loss
- Ablation-ready branch toggles via `model.enabled_branches`
- A smoke config for constrained environments and paper-sized configs for actual reproduction runs.

## Repo layout

```text
configs/
  paper/
  smoke/
scripts/
src/audron/
  data/
  models/
  training/
  utils/
  scripts/
tests/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Synthetic-data path

Generate the synthetic dataset described in the paper:

```bash
python -m audron.scripts.prepare_synthetic_data   --output-dir data/synthetic
```

Train on it:

```bash
python -m audron.scripts.train   --config configs/paper/synthetic.yaml   --train-manifest data/synthetic/manifests/train.jsonl   --val-manifest data/synthetic/manifests/val.jsonl   --output-dir runs/synthetic
```

## Real-data path

First write the download instructions:

```bash
python -m audron.scripts.download_helpers --output data/DATA_SOURCES.txt
```

Then, after downloading the external corpora locally, create manifests:

```bash
python -m audron.scripts.prepare_real_data   --drone-audio-root data/raw/DroneAudioDataset   --drone-noise-root data/raw/DroneNoise_Database   --output-dir data/processed
```

Train binary no augmentation:

```bash
python -m audron.scripts.train   --config configs/paper/binary_no_aug.yaml   --train-manifest data/processed/binary_no_aug/train.jsonl   --val-manifest data/processed/binary_no_aug/val.jsonl   --output-dir runs/binary_no_aug
```

Train binary with augmentation:

```bash
python -m audron.scripts.train   --config configs/paper/binary_with_aug.yaml   --train-manifest data/processed/binary_with_aug/train.jsonl   --val-manifest data/processed/binary_with_aug/val.jsonl   --output-dir runs/binary_with_aug
```

Train multiclass:

```bash
python -m audron.scripts.train   --config configs/paper/multiclass_real.yaml   --train-manifest data/processed/multiclass_real/train.jsonl   --val-manifest data/processed/multiclass_real/val.jsonl   --output-dir runs/multiclass_real
```

Evaluate:

```bash
python -m audron.scripts.evaluate   --config configs/paper/multiclass_real.yaml   --manifest data/processed/multiclass_real/val.jsonl   --checkpoint runs/multiclass_real/best.pt   --output-dir runs/multiclass_real/eval
```

## Live drone detection (listen + alert)

After training a **binary** model (drone vs noise) and exporting to ONNX, you can run a continuous listener that uses the default microphone and prints (and optionally beeps) when a drone is detected:

```bash
pip install onnxruntime sounddevice
python -m audron.scripts.export_onnx --checkpoint runs/binary_no_aug/best.pt --output runs/binary_no_aug/best.onnx --dynamic
python -m audron.scripts.listen_drone --onnx runs/binary_no_aug/best.onnx --threshold 0.6 --alert-cooldown 10
```

- **--threshold**: minimum P(drone) to trigger (default 0.6).
- **--alert-cooldown**: seconds between alerts to avoid spam (default 10).
- **--quiet**: only print when drone detected. Press Ctrl+C to stop.

## Paper-aligned implementation notes

The paper gives the high-level architecture and several dimensionalities, but not every low-level hyperparameter. This repo follows the paper exactly where the paper is explicit, and uses transparent defaults where it is not.

### Explicitly taken from the paper

- four parallel branches: MFCC, STFT-CNN, RNN, autoencoder
- branch output dimensions: 128, 256, 192, 160
- total fused dimension: 736
- synthetic classes: quadcopter, hexacopter, octocopter, racing drone
- synthetic rotor base frequencies: ~75 Hz, ~65 Hz, ~55 Hz, ~120 Hz
- low-frequency modulation parameters: `alpha ~= 0.15`, `fm ~= 1.5 Hz`
- training: batch size 16, AdamW, initial LR 0.001, ReduceLROnPlateau, early stopping, up to 50 epochs
- loss = cross-entropy + weighted MSE reconstruction term
- real dataset ingredients: DroneAudioDataset + ESC-50 + Speech Commands white noise + silence, with DroneNoise Database added for the binary augmentation experiment

### Inferred because the paper does not specify them exactly

- The paper’s autoencoder input is **48,000 samples**, while the referenced DroneAudioDataset is built from **1 s clips at 16 kHz**. To reconcile this, the repo defaults to **3 s fixed-length inputs at 16 kHz** using repeat padding for shorter clips. This matches the 48,000-sample autoencoder input and the 3-second synthetic plots, but the exact padding strategy is **not stated in the paper**.
- The paper shows STFT-CNN conv blocks and linear layers visually, but does not list every activation/pooling detail. This repo uses four Conv2d-BN-ReLU-MaxPool blocks and an adaptive pooling step chosen to match the figure’s 4096 -> 512 -> 256 head.
- The RNN branch is implemented as a 1-layer BiLSTM with attention and a final linear projection to 192 dims because the paper does not state LSTM hidden size or number of layers.
- The classification/reconstruction weighting coefficient is configurable; default `reconstruction_weight=0.1` is an implementation choice because the paper does not provide the coefficient.
- Synthetic harmonic counts and amplitude decay are configurable because `K_c` is mentioned conceptually but not numerically fixed in the text.

## Smoke validation

For low-resource validation:

```bash
python -m audron.scripts.inspect_model --config configs/smoke/tiny_synthetic.yaml
pytest -q tests/smoke_test.py
```

The paper-sized configs are included for real reproduction runs. The smoke config exists only to verify the code path quickly on CPU-constrained machines.
