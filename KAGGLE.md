# Running AUDRON training on Kaggle (GPU)

## 1. Push your code to GitHub

From your machine (after committing):

```bash
# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/audron_repro.git
git branch -M main
git push -u origin main
```

## 2. Create a Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com) → **Code** → **New Notebook**.
2. In the notebook **Settings** (right sidebar):
   - **Accelerator**: **GPU T4 x2** or **P100** (free tier).
   - **Internet**: **On** (so you can clone from GitHub and pip install).

## 3. Clone repo and install

In the first notebook cell:

```python
!git clone https://github.com/YOUR_USERNAME/audron_repro.git
%cd audron_repro
!pip install -e . -q
```

Replace `YOUR_USERNAME` with your GitHub username (or use the repo URL you created).

## 4. Generate synthetic data and train

Next cells:

```python
# Generate synthetic dataset (fast)
!python -m audron.scripts.prepare_synthetic_data --output-dir data/synthetic
```

```python
# Train with GPU (paper config)
!python -m audron.scripts.train \
  --config configs/paper/synthetic.yaml \
  --train-manifest data/synthetic/manifests/train.jsonl \
  --val-manifest data/synthetic/manifests/val.jsonl \
  --output-dir runs/synthetic \
  --device cuda
```

## 5. Save outputs before session ends

Kaggle sessions are temporary. To keep your run:

- **Option A**: Download from the notebook UI: **File** → **Download** and grab `runs/synthetic/` (or zip it first).
- **Option B**: Use Kaggle Datasets: in a cell run `!zip -r runs.zip runs/`, then create a new Dataset and upload `runs.zip`, or use the **Output** tab to save the notebook’s output files.

## Quick one-shot (copy-paste in one cell)

Assumes you’ve already run the clone + `pip install -e .` cell:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "audron.scripts.prepare_synthetic_data", "--output-dir", "data/synthetic"], check=True)
subprocess.run([
    sys.executable, "-m", "audron.scripts.train",
    "--config", "configs/paper/synthetic.yaml",
    "--train-manifest", "data/synthetic/manifests/train.jsonl",
    "--val-manifest", "data/synthetic/manifests/val.jsonl",
    "--output-dir", "runs/synthetic",
    "--device", "cuda"
], check=True)
```

## Notes

- **GPU**: With **Accelerator: GPU** and `--device cuda`, training uses the GPU automatically.
- **Time limit**: Free GPU sessions have a time limit (~9–12 h per week); the paper config often finishes in under an hour on T4/P100.
- **Data**: Synthetic data is generated on the fly; no external datasets required for the synthetic experiment.

---

# Running on real data (DroneAudioDataset) on Kaggle

Uses the public [DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset) (binary: drone vs noise; multiclass: noise, bebop, membo). No GitHub token needed.

## 1. Clone your repo and install (same as above)

```python
!git clone https://github.com/YOUR_USERNAME/audron_repro.git
%cd audron_repro
!pip install -e . -q
```

## 2. Clone DroneAudioDataset and prepare manifests

```python
# DroneAudioDataset is public – no auth needed
!mkdir -p data/raw
!git clone --depth 1 https://github.com/saraalemadi/DroneAudioDataset.git data/raw/DroneAudioDataset

!python -m audron.scripts.prepare_real_data \
  --drone-audio-root data/raw/DroneAudioDataset \
  --output-dir data/processed
```

## 3. Train (choose one)

**Multiclass (noise / bebop / membo):**

```python
!python -m audron.scripts.train \
  --config configs/paper/multiclass_real.yaml \
  --train-manifest data/processed/multiclass_real/train.jsonl \
  --val-manifest data/processed/multiclass_real/val.jsonl \
  --output-dir runs/multiclass_real
```

**Binary (drone vs noise):**

```python
!python -m audron.scripts.train \
  --config configs/paper/binary_no_aug.yaml \
  --train-manifest data/processed/binary_no_aug/train.jsonl \
  --val-manifest data/processed/binary_no_aug/val.jsonl \
  --output-dir runs/binary_no_aug
```

## 4. Save results

Before the session ends: **File** → **Download** (or zip `runs/` and add to a Kaggle Dataset).

## One-shot real-data cells (after clone + pip install)

Run **one** of the following in a cell after the first cell (clone audron_repro + `pip install -e .`). Both clone DroneAudioDataset and prepare manifests; the last part is either **binary** or **multiclass** training.

**Binary (drone vs noise, 2 classes):**

```python
import os, subprocess, sys
os.makedirs("data/raw", exist_ok=True)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/saraalemadi/DroneAudioDataset.git", "data/raw/DroneAudioDataset"], check=True)
subprocess.run([
    sys.executable, "-m", "audron.scripts.prepare_real_data",
    "--drone-audio-root", "data/raw/DroneAudioDataset",
    "--output-dir", "data/processed"
], check=True)
subprocess.run([
    sys.executable, "-m", "audron.scripts.train",
    "--config", "configs/paper/binary_no_aug.yaml",
    "--train-manifest", "data/processed/binary_no_aug/train.jsonl",
    "--val-manifest", "data/processed/binary_no_aug/val.jsonl",
    "--output-dir", "runs/binary_no_aug",
], check=True)
```

**Multiclass (noise / bebop / membo, 3 classes):**

```python
import os, subprocess, sys
os.makedirs("data/raw", exist_ok=True)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/saraalemadi/DroneAudioDataset.git", "data/raw/DroneAudioDataset"], check=True)
subprocess.run([
    sys.executable, "-m", "audron.scripts.prepare_real_data",
    "--drone-audio-root", "data/raw/DroneAudioDataset",
    "--output-dir", "data/processed"
], check=True)
subprocess.run([
    sys.executable, "-m", "audron.scripts.train",
    "--config", "configs/paper/multiclass_real.yaml",
    "--train-manifest", "data/processed/multiclass_real/train.jsonl",
    "--val-manifest", "data/processed/multiclass_real/val.jsonl",
    "--output-dir", "runs/multiclass_real",
], check=True)
```
