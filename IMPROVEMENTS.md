# Possible improvements

Ideas for future work, ordered by impact vs effort. Some are already done (see below).

## Already done in this repo

- **Class imbalance (binary)**: `class_weights: balanced` in config and `AudronLoss(class_weights=...)` for inverse-frequency weighting.
- **Per-class metrics**: `precision_per_class` and `recall_per_class` in training summary and epoch log (binary: noise/drone recall).
- **Checkpoint loading**: `torch.load(..., weights_only=True)` in evaluate for safer loading.
- **Data loading**: Clear `FileNotFoundError` / `RuntimeError` when a manifest path is missing or audio load fails.
- **binary_with_aug.yaml**: Uses `class_weights: balanced` like binary_no_aug.

---

## Training and data

- **WeightedRandomSampler**: In addition to class weights, sample so each batch is more balanced (e.g. oversample drone). Config: `train.balanced_sampler: true`.
- **Data augmentation**: Time stretch, pitch shift, or additive noise on waveform (e.g. via torchaudio or spec_augment) for real data; config toggles.
- **Gradient clipping**: `train.grad_clip: 1.0` and `torch.nn.utils.clip_grad_norm_(model.parameters(), ...)` in the training loop for stability.
- **Multiclass imbalance**: Check class counts for `multiclass_real`; add `class_weights: balanced` if needed.
- **Stratified batch sampling**: Ensure each batch has at least one sample per class when possible (custom sampler or dataloader collate).

---

## Evaluation and logging

- **Best checkpoint by F1**: Option to save best by weighted F1 (or drone recall) instead of accuracy for imbalanced binary.
- **TensorBoard / W&B**: Log curves and metrics per epoch for comparison across runs.
- **Export per-class metrics to CSV**: From `summary.json` or evaluation output for easy plotting.

---

## Code and robustness

- **Pin memory and non_blocking**: `DataLoader(..., pin_memory=True)` and `.to(device, non_blocking=True)` when using GPU for a bit of speed.
- **Reproducibility**: `torch.backends.cudnn.deterministic` and `benchmark=False` when seed is set (optional config).
- **Validation only when needed**: Option to validate every N epochs to save time on large datasets.
- **Resume training**: Load checkpoint and continue from `epoch+1` (optimizer/scheduler state in checkpoint).

---

## Model and config

- **Mixed precision (AMP)**: `torch.autocast` and `GradScaler` for faster training on GPU (optional).
- **Config schema**: Validate YAML (e.g. with pydantic or a small schema) so typos fail fast.
- **Ablation configs**: Predefined configs that disable one branch (e.g. `enabled_branches: [mfcc, stft, rnn]`) for ablations.

---

## Docs and ops

- **Requirements for ONNX**: Document `pip install onnx onnxscript` (or add to optional-deps in pyproject.toml).
- **Dockerfile**: Image with deps and entrypoint for training so Kaggle/local runs match.
- **Pre-commit**: Format (ruff/black) and lint (ruff) on commit.

If you implement one of these, consider adding it to the “Already done” section at the top of this file.
