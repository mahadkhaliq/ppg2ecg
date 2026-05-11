# PPG → ECG Reconstruction

Comparative study of 1D U-Net, BiLSTM, and Transformer architectures for reconstructing ECG signals from PPG. Course project for CMP_SCI 8770 (Intro to Neural Networks), Spring 2026.

## Quickstart

```bash
# Local setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download BIDMC dataset
bash scripts/download_bidmc.sh

# Preprocess
python -m src.data.preprocess

# Train one model locally (smoke test)
python -m src.train --config configs/unet.yaml --epochs 5

# Submit full training on Hellbender
sbatch scripts/slurm_train.sh
```

## Project structure

See `CLAUDE.md` for the full layout and conventions. Key entry points:

- `src/data/preprocess.py` — generates `data/{train,val,test}.npz`
- `src/train.py` — trains a single model from a YAML config
- `src/evaluate.py` — three-tier evaluation on test set
- `report/report.md` — the deliverable

## Hellbender workflow

```bash
# On local: push changes
git add -A && git commit -m "..." && git push

# On Hellbender:
ssh mkfqm@hellbender-login.rnet.missouri.edu
cd ~/ppg2ecg && git pull
sbatch scripts/slurm_train.sh

# Pull results back:
scp -r mkfqm@hellbender-login:~/ppg2ecg/results .
```

## Deadline

May 13, 2026.
