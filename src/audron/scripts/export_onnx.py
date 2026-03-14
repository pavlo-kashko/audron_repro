"""Export a trained AUDRON checkpoint to ONNX (classification logits only)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from audron.models.audron import Audron


def main() -> None:
    parser = argparse.ArgumentParser(description='Export AUDRON checkpoint to ONNX.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to best.pt (or other checkpoint).')
    parser.add_argument('--output', type=Path, default=None, help='Output .onnx path (default: same dir as checkpoint, stem.onnx).')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version.')
    parser.add_argument('--dynamic', action='store_true', help='Export with dynamic batch size.')
    args = parser.parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    out_path = args.output
    if out_path is None:
        out_path = ckpt_path.with_suffix('.onnx')
    else:
        out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = checkpoint['cfg']
    model = Audron(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    input_samples = int(cfg['data']['sample_rate'] * cfg['data']['clip_duration_sec'])
    dummy = torch.randn(1, input_samples)

    input_names = ['waveform']
    output_names = ['logits']
    dynamic_axes = {'waveform': {0: 'batch'}, 'logits': {0: 'batch'}} if args.dynamic else None

    # Export with a wrapper that returns only logits (ONNX doesn't need the full AudronOutputs)
    class LogitsOnly(torch.nn.Module):
        def __init__(self, audron: Audron):
            super().__init__()
            self.audron = audron

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return self.audron(waveform).logits

    wrapped = LogitsOnly(model)
    wrapped.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f'Exported to {out_path}')
    print(f'  Input: waveform [batch, {input_samples}] (float32)')
    print(f'  Output: logits [batch, {cfg["task"]["num_classes"]}] (float32)')
    if args.dynamic:
        print('  Dynamic batch size: yes')


if __name__ == '__main__':
    main()
