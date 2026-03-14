from __future__ import annotations

import argparse
from pathlib import Path

import torch

from audron.models.audron import Audron
from audron.utils.config import apply_overrides, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect AUDRON tensor shapes.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--override', action='append', default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)
    torch.set_num_threads(int(cfg['train'].get('torch_num_threads', 1)))
    model = Audron(cfg)
    x = torch.randn(args.batch_size, int(cfg['data']['sample_rate'] * cfg['data']['clip_duration_sec']))
    model.eval()
    with torch.no_grad():
        out = model(x)
    print('logits', tuple(out.logits.shape))
    for k, v in out.features.items():
        print(k, tuple(v.shape))
    print('reconstruction', tuple(out.reconstruction.shape))
    print('parameters', sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    main()
