from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audron.training.losses import AudronLoss
from audron.utils.metrics import compute_metrics


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict[str, Any]:
    model.train(train)
    y_true, y_pred = [], []
    total_loss = []
    cls_loss = []
    recon_loss = []

    iterator = loader
    for batch in iterator:
        waveform = batch['waveform'].to(device)
        labels = batch['label'].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(waveform)
            loss, parts = criterion(outputs.logits, labels, outputs.reconstruction, waveform)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = torch.argmax(outputs.logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        total_loss.append(parts['loss'])
        cls_loss.append(parts['classification_loss'])
        recon_loss.append(parts['reconstruction_loss'])

    metrics = compute_metrics(y_true, y_pred, average='weighted')
    return {
        'loss': float(np.mean(total_loss)),
        'classification_loss': float(np.mean(cls_loss)),
        'reconstruction_loss': float(np.mean(recon_loss)),
        'accuracy': metrics.accuracy,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1': metrics.f1,
        'confusion': metrics.confusion.tolist(),
    }


def fit(model, train_loader: DataLoader, val_loader: DataLoader, cfg: dict, output_dir: str | Path, device: torch.device) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg['train']
    criterion = AudronLoss(reconstruction_weight=float(train_cfg['reconstruction_weight']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg['lr']), weight_decay=float(train_cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=float(train_cfg['lr_factor']), patience=int(train_cfg['scheduler_patience']))

    history = History()
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    best_path = output_dir / 'best.pt'
    summary_path = output_dir / 'summary.json'

    for epoch in range(1, int(train_cfg['epochs']) + 1):
        epoch_start = time.perf_counter()
        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_stats = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        epoch_elapsed = time.perf_counter() - epoch_start
        scheduler.step(val_stats['accuracy'])

        print(
            f'epoch {epoch}/{train_cfg["epochs"]}  '
            f'train_loss={train_stats["loss"]:.4f} train_acc={train_stats["accuracy"]:.4f}  '
            f'val_loss={val_stats["loss"]:.4f} val_acc={val_stats["accuracy"]:.4f}  '
            f'time={epoch_elapsed:.1f}s'
        )

        history.train_loss.append(train_stats['loss'])
        history.val_loss.append(val_stats['loss'])
        history.train_acc.append(train_stats['accuracy'])
        history.val_acc.append(val_stats['accuracy'])

        if val_stats['accuracy'] > best_val_acc:
            best_val_acc = val_stats['accuracy']
            best_epoch = epoch
            patience_counter = 0
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= int(train_cfg['early_stopping_patience']):
                print(f'Early stopping at epoch {epoch}')
                break

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'epoch': epoch,
                'train': train_stats,
                'val': val_stats,
                'best_val_accuracy': best_val_acc,
                'best_epoch': best_epoch,
            }, f, indent=2)

    _plot_history(history, output_dir)
    return {
        'best_checkpoint': str(best_path),
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'history': history.__dict__,
    }


def evaluate(model, loader: DataLoader, cfg: dict, checkpoint_path: str | Path, output_dir: str | Path, device: torch.device) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    criterion = AudronLoss(reconstruction_weight=float(cfg['train']['reconstruction_weight']))
    stats = run_epoch(model, loader, criterion, optimizer=None, device=device, train=False)
    with open(output_dir / 'evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    return stats


def _plot_history(history: History, output_dir: Path) -> None:
    epochs = np.arange(1, len(history.train_loss) + 1)
    if len(epochs) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_loss, label='train_loss')
    plt.plot(epochs, history.val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_acc, label='train_accuracy')
    plt.plot(epochs, history.val_acc, label='val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curve.png', dpi=150)
    plt.close()
