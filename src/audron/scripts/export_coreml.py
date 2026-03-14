"""
Export the binary ONNX model to Core ML for use in iOS (e.g. iPhone demo).

Requires: pip install coremltools

Usage:
  python -m audron.scripts.export_coreml --onnx runs/binary_no_aug/best.onnx --output DroneClassifier.mlpackage

Then add the .mlpackage to your Xcode project. Input: waveform [1, 48000] float32. Output: logits [1, 2].
Apply softmax in Swift to get P(noise), P(drone). Class 1 = drone.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX model to Core ML for iOS.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("DroneClassifier.mlpackage"))
    parser.add_argument("--minimum-deployment-target", type=int, default=16, help="e.g. 16 for iOS 16")
    args = parser.parse_args()

    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("Install coremltools: pip install coremltools")

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    # Convert; use Neural Network backend for best compatibility
    model = ct.converters.onnx.convert(
        model_path=str(onnx_path),
        minimum_deployment_target=ct.target.iOS(args.minimum_deployment_target),
    )
    out = args.output.resolve()
    model.save(str(out))
    print(f"Saved Core ML model to {out}")
    print("  Add to Xcode: drag .mlpackage into the project.")
    print("  Input: waveform [1, 48000] float32, 16 kHz mono, peak-normalized.")
    print("  Output: logits [1, 2]; index 1 = drone. Apply softmax in Swift.")


if __name__ == "__main__":
    main()
