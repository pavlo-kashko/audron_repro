from __future__ import annotations

from pathlib import Path

import torch

from audron.models.audron import Audron
from audron.utils.config import load_yaml


def test_smoke_forward() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / 'configs' / 'smoke' / 'tiny_synthetic.yaml')
    torch.set_num_threads(1)
    model = Audron(cfg)
    x = torch.randn(2, int(cfg['data']['sample_rate'] * cfg['data']['clip_duration_sec']))
    out = model(x)
    assert out.logits.shape == (2, cfg['task']['num_classes'])
    assert out.reconstruction.shape == x.shape
