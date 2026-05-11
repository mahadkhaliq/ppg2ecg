# CLAUDE.md — PPG-to-ECG Reconstruction Project

> **Project memory for Claude Code.** Read this before doing anything in this repo.

---

## Mission

Comparative study of deep learning architectures for reconstructing 12-lead ECG (lead II) from PPG signals. Final deliverable: a CMP_SCI 8770 course report (≤2,500 words) plus reproducible training/evaluation code.

**Hard deadline: May 13, 2026.** Today is May 10. Two days of effective work + buffer day. **Scope discipline is more important than ambition.**

---

## Project context

- **Course:** CMP_SCI 8770 — Intro to Neural Networks (Dr. Jordan Malof)
- **Author:** Mahad
- **Project type:** Type 5 from the assignment — reproduce-and-extend (CardioGAN, Sarkar & Etemad 2021)
- **Rubric levers:** several ML models compared, appropriate literature, appropriate metrics, clean experimental design
- **Word limit:** 2,500 (excluding refs). Required: at least one figure, one flowchart, three references.

---

## Scope (locked)

**In scope:**
- Three model families: 1D U-Net, BiLSTM seq2seq, Transformer
- BIDMC PPG and Respiration Dataset (PhysioNet) — primary
- Three-tier evaluation: waveform fidelity, morphology, downstream classification
- Subject-level train/val/test split (38/7/8)
- A small hyperparameter sweep on the best-performing model (3 conditions)
- Final report in markdown, exported to PDF

**Explicitly out of scope (do NOT implement):**
- Diffusion models — too risky for the timeline
- Mamba / state-space models — too risky
- MIMIC-III cross-dataset evaluation — drop unless time remains on day 2
- Streamlit dashboard — drop. Report and code only.
- Echocardiogram synthesis — never in scope
- 12-lead reconstruction — only lead II (BIDMC limitation anyway)

If a request would expand scope beyond the locked list, **push back and remind the user of the deadline.**

---

## Repository layout

```
ppg2ecg/
├── CLAUDE.md                   # This file
├── README.md                   # Human-readable overview
├── requirements.txt
├── .gitignore
├── src/
│   ├── data/
│   │   ├── bidmc.py            # Dataset class, train/val/test split
│   │   └── preprocess.py       # Filtering, segmentation, SQI
│   ├── models/
│   │   ├── unet.py             # 1D U-Net
│   │   ├── bilstm.py           # BiLSTM seq2seq
│   │   └── transformer.py      # Transformer encoder-decoder
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Three-tier evaluation
│   └── utils.py                # Seeding, logging, IO helpers
├── scripts/
│   ├── download_bidmc.sh       # Pulls dataset from PhysioNet
│   └── slurm_train.sh          # Hellbender SLURM job
├── configs/
│   ├── unet.yaml
│   ├── bilstm.yaml
│   └── transformer.yaml
├── checkpoints/                # gitignored
├── results/                    # gitignored
└── report/
    └── report.md               # Final report draft
```

---

## Coding conventions

- **Python 3.10+, PyTorch 2.x.** Type hints everywhere — `from __future__ import annotations`.
- **No fancy frameworks.** Plain PyTorch + a small `train.py`. No PyTorch Lightning, no Hydra (use simple YAML configs read with `yaml.safe_load`). Time pressure favors transparent code.
- **Reproducibility:** every entry point sets seeds via `utils.set_seed(seed)`. Default seed = 42.
- **Logging:** print to stdout with timestamps. Save per-epoch metrics to `results/<run_id>/metrics.json`. Save the config alongside. **No Weights & Biases setup** — no time.
- **Checkpoints:** `checkpoints/<model_name>/best.pt` (best val loss) and `last.pt`. Save `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_loss`, and the config dict.
- **All shapes documented** in docstrings: `(B, 1, T)` for raw signal, `(B, T)` after squeeze, etc.
- **Tensor dtype:** float32 throughout. PPG and ECG both normalized per-segment to zero mean, unit variance.
- **Deterministic where possible** — but don't fight cuDNN, partial determinism is fine for this project.

---

## Environment

### Local (Alienware Ubuntu, code-only)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Hellbender (training)
- SSH: `ssh mkfqm@hellbender-login.rnet.missouri.edu`
- Conda env: create a fresh `ppg2ecg` env, do not reuse `synthgrad` (different deps, avoid cross-contamination)
- Partition: `engineering` (same as SynthGrad)
- GPU request: 1 GPU, 16 GB memory is plenty for these models
- Estimated wall time per training run: 1–3 hours

### Sync workflow
- Develop locally with Claude Code
- Push to GitHub (private repo)
- On Hellbender: `git pull` then submit SLURM job
- Pull results back via `scp -r mkfqm@hellbender-login:/path/to/results .`

---

## Data — BIDMC

- **Source:** https://physionet.org/content/bidmc/1.0.0/
- **License:** Open Data Commons Attribution License (ODC-By 1.0)
- **Format:** WaveForm DataBase (WFDB) `.dat` + `.hea` files, plus a `bidmc_csv/` directory with per-subject CSVs
- **Use the CSV directory** — simpler to parse than WFDB binary format
- **53 subjects, ~8 minutes each at 125 Hz**
- **Channels we use:** PPG (column `PLETH`) and ECG lead II (column `II`)
- **Drop subjects with corrupted segments** — flag during preprocessing, do not silently include
- **Segment length:** 4 seconds (500 samples at 125 Hz), 50% overlap stride
- **Splits (subject-level, no patient leak):**
  - Train: subjects 1–38
  - Val: subjects 39–45
  - Test: subjects 46–53
  - Save the exact subject IDs to `results/splits.json` for reproducibility

### Preprocessing pipeline (in this exact order)
1. Load PPG and ECG as float32 numpy arrays
2. Bandpass filter PPG: 0.5–8 Hz, 4th-order Butterworth, zero-phase (`scipy.signal.filtfilt`)
3. Bandpass filter ECG: 0.5–40 Hz, 4th-order Butterworth, zero-phase
4. Segment into 4-second windows with 50% overlap
5. Compute Signal Quality Index (SQI) per window — **template matching against beat templates**, drop windows with SQI < 0.5
6. Z-score normalize per-window (mean 0, std 1) — both PPG and ECG independently
7. Save to `.npz` files: `train.npz`, `val.npz`, `test.npz`, each with `ppg` and `ecg` arrays of shape `(N, 500)`

---

## Models

All three follow this interface:

```python
class Model(nn.Module):
    def __init__(self, config: dict): ...
    def forward(self, ppg: Tensor) -> Tensor:
        # ppg: (B, 1, T) where T=500
        # returns: (B, 1, T) reconstructed ECG
        ...
```

### 1D U-Net (`src/models/unet.py`)
- Encoder: 4 down-blocks, each `Conv1d → BN → ReLU → Conv1d → BN → ReLU → MaxPool1d(2)`
- Bottleneck: 2 conv blocks
- Decoder: 4 up-blocks with skip connections, `ConvTranspose1d → concat → Conv1d → BN → ReLU → Conv1d → BN → ReLU`
- Final: `Conv1d(channels, 1, kernel_size=1)`
- Channels: [32, 64, 128, 256, 512] (encoder), reverse for decoder
- **~2M parameters target**

### BiLSTM seq2seq (`src/models/bilstm.py`)
- Encoder: 2-layer BiLSTM, hidden dim 128
- Decoder: 2-layer LSTM with attention over encoder outputs (Bahdanau-style)
- Final projection to 1 channel
- **~1.5M parameters target**
- Trained with teacher forcing during training, autoregressive at inference (or use parallel decoder for speed — document the choice)

### Transformer (`src/models/transformer.py`)
- Patch the input: split 500-sample window into patches of 25 → 20 patches per sequence
- Linear projection to `d_model=128`
- Positional encoding (sinusoidal, fixed)
- 4 encoder layers, 4 decoder layers, 4 heads, FFN dim 512
- Linear head back to patches, reshape to `(B, 1, T)`
- **~5M parameters target**

---

## Training

### Loss
Combined: `L = L1(y_pred, y_true) + λ * L_freq(y_pred, y_true)` where `L_freq` is the L1 distance between magnitude STFTs.
- Default `λ = 0.5`
- STFT settings: `n_fft=128`, `hop_length=32`, `win_length=128`, Hann window

### Optimizer
- AdamW, `lr=1e-4`, `weight_decay=1e-5`
- Cosine annealing schedule, T_max = total epochs
- Gradient clip at 1.0

### Schedule
- 100 epochs default, but **early stopping with patience=10** on validation loss
- Batch size 64
- All three models should fit on a single GPU with 16 GB

### Per-epoch logging (printed and JSON)
- Train loss
- Val loss (the best metric for checkpoint selection)
- Time per epoch
- Learning rate

---

## Evaluation — Three Tiers

Implemented in `src/evaluate.py`. Run after training each model.

### Tier 1: Waveform fidelity
- **RMSE** between predicted and true ECG (lower is better)
- **Pearson correlation coefficient** (higher is better)
- **DTW distance** using `dtaidistance` library (lower is better) — compute on a 200-segment subsample of test set, full DTW is too slow

### Tier 2: Morphology preservation
- Detect R-peaks on both real and reconstructed ECG using `neurokit2.ecg_findpeaks`
- **R-peak F1**: match detected peaks within ±50 ms tolerance
- **PR interval error** (median absolute error in ms)
- **QT interval error** (median absolute error in ms)

### Tier 3: Downstream task
- Train a small 1D CNN arrhythmia classifier on real ECG segments (the BIDMC labels are limited; for a quick proxy, use heart-rate-bucket classification: bradycardia <60, normal 60–100, tachycardia >100 BPM. Three-class, balanced sampling.)
- Apply trained classifier to reconstructed ECG segments
- Report **accuracy degradation vs. real ECG baseline**
- This is the most important metric for the report — it answers "does the reconstruction preserve clinically meaningful information?"

### Output
- Save all metrics per model to `results/<model_name>/metrics.json`
- Save 5 random test-set reconstructions as numpy arrays (for the qualitative figure) to `results/<model_name>/qualitative.npz`

---

## Report deliverable

Located in `report/report.md`. Map exactly to the template sections:

1. **Introduction** (~400 words) — motivation, problem statement, what we deliver
2. **Related Work** (~500 words) — CardioGAN, P2E-WGAN, diffusion biosignal work, evaluation methodology. Minimum 3 citations, target 12–15.
3. **Data** (~200 words + Table 1) — BIDMC description, splits, preprocessing
4. **Experiments** (~500 words + flowchart Figure 1) — three architectures, training protocol, evaluation tiers
5. **Results** (~600 words + Tables 2,3 + Figures 2,3) — main results table, qualitative reconstructions, metric bar chart
6. **Conclusions** (~200 words) — key finding, limitations, future work
7. **References** (IEEE format, 12+ entries)

**Length target: 2,400 words to leave a safety margin under the 2,500 cap.**

Required figures (per template):
- **Figure 1:** pipeline flowchart (data → preprocessing → 3 models → 3-tier eval) — generate with matplotlib or Mermaid
- **Figure 2:** qualitative reconstruction examples (4 panels: input PPG, ground truth ECG, and one model's output, for two test segments)
- **Figure 3:** per-metric comparison bar chart with error bars across the three models

---

## Key references (build the Zotero library from these)

These are the must-cites. Add more during the lit review pass.

1. Sarkar & Etemad, *CardioGAN*, AAAI 2021 — **primary reproduction target**
2. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
3. Ronneberger, Fischer, Brox, *U-Net*, MICCAI 2015 — adapt to 1D
4. Sutskever, Vinyals, Le, *Sequence to Sequence Learning*, NeurIPS 2014
5. Hannun et al., *Cardiologist-level arrhythmia detection*, Nature Medicine 2019
6. Goldberger et al., *PhysioBank/PhysioNet/PhysioToolkit*, Circulation 2000 — the BIDMC citation
7. Pimentel et al., *Toward a robust estimation of respiratory rate from pulse oximeters*, IEEE TBME 2017 — BIDMC paper
8. Vo, Nguyen, Le, *P2E-WGAN: ECG waveform synthesis from PPG with conditional Wasserstein GAN*, 2021
9. Tang et al., *PPG2ECGps: An end-to-end subject-aware deep neural network for PPG to ECG translation*, 2022
10. Zhu et al., *ECG synthesis from PPG using region-of-interest based GAN*, 2021
11. Loshchilov & Hutter, *Decoupled Weight Decay Regularization (AdamW)*, ICLR 2019
12. Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017

---

## Workflow with Claude Code

Recommended sequence on day 1:

1. `bash scripts/download_bidmc.sh` — get the data (run on Hellbender, BIDMC is small ~200 MB)
2. Implement `src/data/preprocess.py` and run it to generate `train/val/test.npz`. **Verify shapes and visually plot a few segments** before going further.
3. Implement and unit-test U-Net first (simplest). Train for 5 epochs as a smoke test.
4. Once U-Net training works end-to-end, implement BiLSTM and Transformer.
5. Submit all three full SLURM jobs in parallel — overnight runs.

Day 2:
1. Run evaluation pipeline on all three trained checkpoints.
2. Generate figures.
3. Write report from the skeleton.

**If anything blocks you for >30 minutes, simplify or cut.** This deadline cannot slip.

---

## Things to actively avoid

- Don't add datasets beyond BIDMC mid-project
- Don't refactor working code into "cleaner" abstractions
- Don't tune hyperparameters extensively — defaults from this file are fine
- Don't add visualization libraries beyond matplotlib
- Don't write tests unless something is actually breaking
- Don't try to make the dashboard work — it's cut from scope
- Don't introduce new dependencies without a strong reason

---

## When in doubt

Ask: *does this make the report more rigorous, or is it just polish?* If polish, skip it. The grading rubric rewards rigor, not features.
