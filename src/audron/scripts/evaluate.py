from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audron.data.dataset import AudioManifestDataset
from audron.models.audron import Audron
from audron.training.engine import evaluate
from audron.utils.config import apply_overrides, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate AUDRON.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    torch.set_num_threads(int(cfg['train'].get('torch_num_threads', 1)))
    dataset = AudioManifestDataset(
        args.manifest,
        sample_rate=cfg['data']['sample_rate'],
        clip_duration_sec=cfg['data']['clip_duration_sec'],
        pad_mode=cfg['data'].get('pad_mode', 'repeat'),
        normalize_audio=cfg['data'].get('normalize_audio', True),
    )
    loader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'])
    device = torch.device(args.device)
    model = Audron(cfg).to(device)
    stats = evaluate(model, loader, cfg, args.checkpoint, args.output_dir, device=device)
    print(stats)


if __name__ == '__main__':
    main()
