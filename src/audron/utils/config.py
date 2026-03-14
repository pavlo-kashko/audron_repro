from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _assign_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split('.')
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None = None) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    if not overrides:
        return out
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Override must have KEY=VALUE form, got: {item}")
        key, raw = item.split('=', 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            lowered = raw.lower()
            if lowered in {'true', 'false'}:
                value = lowered == 'true'
            else:
                try:
                    value = int(raw)
                except ValueError:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
        _assign_nested(out, key, value)
    return out
