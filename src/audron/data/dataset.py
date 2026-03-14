from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from audron.data.audio import fit_audio_length, load_audio, peak_normalize
from audron.utils.io import read_jsonl


class AudioManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int,
        clip_duration_sec: float,
        pad_mode: str = 'repeat',
        normalize_audio: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.rows = read_jsonl(manifest_path)
        self.sample_rate = sample_rate
        self.clip_duration_sec = clip_duration_sec
        self.target_length = int(round(sample_rate * clip_duration_sec))
        self.pad_mode = pad_mode
        self.normalize_audio = normalize_audio

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        row = self.rows[idx]
        audio_path = Path(row['path'])
        if not audio_path.is_absolute():
            candidate = self.base_dir / audio_path
            if candidate.exists():
                audio_path = candidate
            else:
                # common case for generated data: manifest lives in a sibling folder to audio/
                candidate = self.base_dir.parent / audio_path
                if candidate.exists():
                    audio_path = candidate
        waveform = load_audio(audio_path, sample_rate=self.sample_rate)
        waveform = fit_audio_length(waveform, self.target_length, pad_mode=self.pad_mode)
        if self.normalize_audio:
            waveform = peak_normalize(waveform)
        return {
            'waveform': torch.from_numpy(np.asarray(waveform, dtype=np.float32)),
            'label': int(row['label_id']),
            'label_name': row['label_name'],
            'path': str(audio_path),
        }
