"""
Run the trained model on a single WAV file. Supports PyTorch checkpoint or ONNX.

Usage:
  # With ONNX (no config needed; input length from model)
  python -m audron.scripts.predict_audio --onnx runs/binary_no_aug/best.onnx --audio experiment_audio.wav

  # With PyTorch checkpoint (config optional; uses config saved in checkpoint if omitted)
  python -m audron.scripts.predict_audio --checkpoint runs/binary_no_aug/best.pt --audio experiment_audio.wav
  python -m audron.scripts.predict_audio --checkpoint runs/binary_no_aug/best.pt --config configs/paper/binary_no_aug.yaml --audio experiment_audio.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from audron.data.audio import fit_audio_length, load_audio, peak_normalize
from audron.utils.config import load_yaml


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_onnx(audio_path: Path, onnx_path: Path) -> tuple[np.ndarray, list[str]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_spec = session.get_inputs()[0]
    # shape may be ['batch', 48000] or [1, 48000] or ['batch', 'samples']
    shape = input_spec.shape
    if len(shape) == 2:
        target_length = int(shape[1]) if isinstance(shape[1], int) else 48000
    else:
        target_length = 48000
    sample_rate = 16000

    waveform = load_audio(audio_path, sample_rate=sample_rate)
    waveform = fit_audio_length(waveform, target_length, pad_mode="repeat")
    waveform = peak_normalize(waveform)
    waveform = waveform.astype(np.float32).reshape(1, -1)

    logits = session.run(None, {input_spec.name: waveform})[0][0]
    probs = softmax(logits)
    # Default binary class names for ONNX (no config)
    class_names = ["noise", "drone"] if len(probs) == 2 else [f"class_{i}" for i in range(len(probs))]
    return probs, class_names


def run_pytorch(audio_path: Path, checkpoint_path: Path, config_path: Path | None) -> tuple[np.ndarray, list[str]]:
    import torch

    from audron.models.audron import Audron

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"] if config_path is None else load_yaml(config_path)
    sample_rate = int(cfg["data"]["sample_rate"])
    clip_duration_sec = float(cfg["data"]["clip_duration_sec"])
    target_length = int(round(sample_rate * clip_duration_sec))
    class_names = list(cfg["task"]["class_names"])

    model = Audron(checkpoint["cfg"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    waveform = load_audio(audio_path, sample_rate=sample_rate)
    waveform = fit_audio_length(waveform, target_length, pad_mode="repeat")
    waveform = peak_normalize(waveform)
    x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = model(x).logits[0].numpy()
    probs = softmax(logits)
    return probs, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model on a single WAV file.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to WAV file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", type=Path, default=None, help="Path to ONNX model (e.g. best.onnx).")
    group.add_argument("--checkpoint", type=Path, default=None, help="Path to PyTorch checkpoint (optional --config, else uses checkpoint's config).")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (optional with --checkpoint; overrides config saved in checkpoint).")
    args = parser.parse_args()

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if args.onnx is not None:
        onnx_path = args.onnx.resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        probs, class_names = run_onnx(audio_path, onnx_path)
    else:
        ckpt_path = args.checkpoint.resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        cfg_path = args.config.resolve() if args.config is not None else None
        if cfg_path is not None and not cfg_path.is_file():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        probs, class_names = run_pytorch(audio_path, ckpt_path, cfg_path)

    pred_id = int(np.argmax(probs))
    print(f"File: {audio_path}")
    print(f"Prediction: {class_names[pred_id]} (class {pred_id})")
    print("Probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
