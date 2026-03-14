from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from audron.utils.io import write_jsonl


def make_records(paths: Iterable[Path], label_name: str, label_id: int) -> list[dict]:
    rows = []
    for path in sorted(paths):
        rows.append({'path': str(Path(path).resolve()), 'label_name': label_name, 'label_id': label_id})
    return rows


def train_val_split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in records:
        grouped.setdefault(int(row['label_id']), []).append(row)

    train_rows, val_rows = [], []
    rng = random.Random(seed)
    for _, group in grouped.items():
        group = group[:]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_fraction))
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_split(train_rows: list[dict], val_rows: list[dict], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, out_dir / 'train.jsonl')
    write_jsonl(val_rows, out_dir / 'val.jsonl')
