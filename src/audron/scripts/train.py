from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audron.data.dataset import AudioManifestDataset
from audron.models.audron import Audron
from audron.training.engine import fit
from audron.utils.config import apply_overrides, load_yaml, save_yaml
from audron.utils.io import ensure_dir
from audron.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description='Train AUDRON.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--train-manifest', type=Path, required=True)
    parser.add_argument('--val-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    torch.set_num_threads(int(cfg['train'].get('torch_num_threads', 1)))
    set_seed(int(cfg['seed']))
    output_dir = ensure_dir(args.output_dir)
    save_yaml(cfg, output_dir / 'resolved_config.yaml')

    train_ds = AudioManifestDataset(
        args.train_manifest,
        sample_rate=cfg['data']['sample_rate'],
        clip_duration_sec=cfg['data']['clip_duration_sec'],
        pad_mode=cfg['data'].get('pad_mode', 'repeat'),
        normalize_audio=cfg['data'].get('normalize_audio', True),
    )
    val_ds = AudioManifestDataset(
        args.val_manifest,
        sample_rate=cfg['data']['sample_rate'],
        clip_duration_sec=cfg['data']['clip_duration_sec'],
        pad_mode=cfg['data'].get('pad_mode', 'repeat'),
        normalize_audio=cfg['data'].get('normalize_audio', True),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'])

    device = torch.device(args.device)
    model = Audron(cfg).to(device)
    summary = fit(model, train_loader, val_loader, cfg, output_dir=output_dir, device=device)
    print(summary)


if __name__ == '__main__':
    main()
