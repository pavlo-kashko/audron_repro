"""
Run the model with a sliding 3 s window over a long WAV file and plot P(drone) vs time.

Usage:
  python -m audron.scripts.plot_sliding_predictions --onnx runs/binary_no_aug/best.onnx --audio experiment_audio.wav --output experiment_audio_prediction.png
  python -m audron.scripts.plot_sliding_predictions --checkpoint runs/binary_no_aug/best.pt --audio experiment_audio.wav --hop 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audron.data.audio import fit_audio_length, load_audio, peak_normalize
from audron.utils.config import load_yaml


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _get_onnx_runner(onnx_path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_spec = session.get_inputs()[0]
    shape = input_spec.shape
    window_samples = int(shape[1]) if len(shape) == 2 and isinstance(shape[1], int) else 48000
    sample_rate = 16000
    class_names = ["noise", "drone"]

    def run(window: np.ndarray) -> np.ndarray:
        window = window.astype(np.float32)
        window = fit_audio_length(window, window_samples, pad_mode="repeat")
        window = peak_normalize(window)
        window = window.reshape(1, -1)
        logits = session.run(None, {input_spec.name: window})[0][0]
        return softmax(logits)

    return run, window_samples, sample_rate, class_names


def _get_pytorch_runner(checkpoint_path: Path, config_path: Path | None):
    import torch

    from audron.models.audron import Audron

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["cfg"] if config_path is None else load_yaml(config_path)
    sample_rate = int(cfg["data"]["sample_rate"])
    clip_duration_sec = float(cfg["data"]["clip_duration_sec"])
    window_samples = int(round(sample_rate * clip_duration_sec))
    class_names = list(cfg["task"]["class_names"])

    model = Audron(checkpoint["cfg"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    def run(window: np.ndarray) -> np.ndarray:
        window = window.astype(np.float32)
        window = fit_audio_length(window, window_samples, pad_mode="repeat")
        window = peak_normalize(window)
        x = torch.from_numpy(window).unsqueeze(0)
        with torch.no_grad():
            logits = model(x).logits[0].numpy()
        return softmax(logits)

    return run, window_samples, sample_rate, class_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sliding 3 s window over WAV file and plot model output (e.g. P(drone)) vs time."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to WAV file.")
    parser.add_argument("--output", type=Path, default=None, help="Output plot path (default: <audio_stem>_prediction.png).")
    parser.add_argument("--window", type=float, default=3.0, help="Window length in seconds.")
    parser.add_argument("--hop", type=float, default=0.1, help="Hop between windows in seconds.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", type=Path, default=None, help="Path to ONNX model.")
    group.add_argument("--checkpoint", type=Path, default=None, help="Path to PyTorch checkpoint.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (optional with --checkpoint).")
    args = parser.parse_args()

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if args.onnx is not None:
        onnx_path = args.onnx.resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        run_fn, window_samples, sample_rate, class_names = _get_onnx_runner(onnx_path)
    else:
        ckpt_path = args.checkpoint.resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        cfg_path = args.config.resolve() if args.config else None
        run_fn, window_samples, sample_rate, class_names = _get_pytorch_runner(ckpt_path, cfg_path)

    window_sec = window_samples / sample_rate
    hop_samples = int(round(sample_rate * args.hop))

    waveform = load_audio(audio_path, sample_rate=sample_rate)
    duration_sec = len(waveform) / sample_rate

    times = []
    probs_list = []

    start = 0
    while start + window_samples <= len(waveform):
        window = waveform[start : start + window_samples]
        probs = run_fn(window)
        probs_list.append(probs)
        time_center_sec = (start + window_samples / 2) / sample_rate
        times.append(time_center_sec)
        start += hop_samples

    if not times:
        # File shorter than window: run once on padded window
        window = waveform
        probs = run_fn(window)
        probs_list = [probs]
        times = [duration_sec / 2]

    times = np.array(times)
    probs_arr = np.array(probs_list)

    out_path = args.output
    if out_path is None:
        out_path = audio_path.with_name(f"{audio_path.stem}_prediction.png")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(class_names):
        ax.plot(times, probs_arr[:, i], label=name, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Sliding window prediction (window={window_sec}s, hop={args.hop}s)\n{audio_path.name}")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Plot saved to {out_path}")
    print(f"  Duration: {duration_sec:.1f} s, windows: {len(times)}, hop: {args.hop} s")


if __name__ == "__main__":
    main()
