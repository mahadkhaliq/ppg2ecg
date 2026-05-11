# PPG-to-ECG Reconstruction

<img width="800" height="336" alt="ezgif-30604901485443b5" src="https://github.com/user-attachments/assets/9864aeb4-b133-4e99-b6b0-67e8438e6f78" />


Comparative study of deep learning architectures for reconstructing single-lead ECG (Lead II) from photoplethysmography (PPG) signals.  
Course Project: CMPSCI 8770 Introduction to Neural Networks, University of Missouri.

---

<img width="800" height="336" alt="ezgif-39e84b7ef2fda190" src="https://github.com/user-attachments/assets/77b16c34-c7b7-4814-80fe-aea3241b013b" />

---

## Dataset

![Dataset Overview](report/dataset_overview.png)

---

## Results

| Model        | Params   | RMSE  | Pearson r | DTW   | R-peak F1 | RR err (ms) | HR-bucket acc |
|--------------|----------|-------|-----------|-------|-----------|-------------|---------------|
| **BiLSTM**   | 892 K    | **1.050** | **0.334** | **7.37** | 0.806 | **8.0** | **98.0%** |
| U-Net        | 2,711 K  | 1.134 | 0.266     | 9.81  | **0.811** | 16.0        | 97.5%         |
| BiLSTM+GAN   | 892+42 K | 1.200 | 0.161     | 7.97  | 0.732     | **8.0**     | 97.2%         |
| Transformer  | ~800 K   | 1.270 | 0.067     | 8.64  | 0.457     | 72.0        | 96.0%         |

**BiLSTM wins overall.** Low Pearson r is expected for L1-trained regression models (regression-to-mean); the high downstream accuracy (96–98%) confirms that rhythm information is preserved even when morphology is averaged.

---

## Training Curves

![Training and Validation Loss](report/training_curves.png)

| Model       | Best Val Loss | Stopped Epoch | Early Stop Patience |
|-------------|---------------|---------------|---------------------|
| U-Net       | 1.5156        | 17 / 27       | 10                  |
| BiLSTM      | 1.5306        | 18 / 28       | 10                  |
| BiLSTM+GAN  | 1.5808        | 5 / 15        | 10                  |
| Transformer | 1.6141        | 16 / 26       | 10                  |

GAN discriminator collapsed early (d_loss curve visible in bottom-right panel) — dataset too small (3,184 segments) for stable adversarial training.

---

<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/177bdbd8-f2d8-4358-8b54-a4a1bf9ed91e" />

## Model Architectures

### 1D U-Net — 2,710,753 parameters

```
Input (B, 1, 500)
  Encoder × 4  [Conv1d→BN→ReLU→Conv1d→BN→ReLU→MaxPool(2)]
    Channels: 1→32→64→128→256
  Bottleneck   [2× Conv1d blocks, 512 channels]
  Decoder × 4  [ConvTranspose1d + skip concat → Conv1d→BN→ReLU×2]
    Channels: 512→256→128→64→32
  Head         [Conv1d(32,1,k=1)]
Output (B, 1, 500)
```

Encoder captures multi-scale PPG features; skip connections preserve temporal alignment. Best R-peak F1 (0.811).

---

### BiLSTM Seq2Seq — 892,161 parameters

```
Input (B, 1, 500) → squeeze → (B, 500)
  Feature projection  Linear(1→32)
  Encoder BiLSTM      2 layers, hidden=128, bidirectional
    encoder_out: (B, 500, 256)
  Decoder LSTM        2 layers, hidden=256
    + Scaled dot-product attention over encoder_out
  Output projection   Linear(256→1)
Output (B, 1, 500)
```

Note: Bahdanau attention replaced with scaled dot-product (OOM on B=64, T=500 with 8 GB VRAM). Best overall model — lowest RMSE and highest Pearson r.

---

### Transformer Encoder-Decoder — ~800,000 parameters

```
Input (B, 1, 500)
  Patch embed    500→20 patches of 25 samples, Linear(25→128)
  + Sinusoidal PE  (B, 20, 128)
  Encoder        4 layers, 4 heads, FFN dim=512
  Decoder        4 layers, 4 heads, FFN dim=512, causal mask
  Head           Linear(128→25), reshape (B,1,500)
Output (B, 1, 500)
```

Weakest on morphology (R-peak F1=0.457) and RR error (72 ms). Data-efficiency problem — Transformers need larger datasets than BIDMC's 3,184 training segments.

---

### BiLSTM + GAN — 892 K (G) + 42 K (D) parameters

Generator identical to BiLSTM above. Discriminator: 4× `Conv1d+LeakyReLU` with spectral normalisation → `AdaptiveAvgPool → Linear(128,1)`. Discriminator collapsed despite spectral normalisation — honest negative result.

---

## Beat Classifier (MIT-BIH ResNet1D) — 530,273 parameters

Trained on MIT-BIH Arrhythmia Database (48 records, 109,375 beats).  
AAMI 5-class: **N** Normal · **S** Supraventricular ectopic · **V** Ventricular ectopic · **F** Fusion · **Q** Unknown/paced

```
Input (1, 250)  — ±1 s window around R-peak at 125 Hz
  Stem   Conv1d(1,32,k=15,s=2) → BN → ReLU
  Block1 ResBlock(32,32)
  Block2 ResBlock(32,64,  stride=2)
  Block3 ResBlock(64,64)
  Block4 ResBlock(64,128, stride=2)
  Block5 ResBlock(128,128)
  Pool   AdaptiveAvgPool1d(1)
  Head   Linear(128,5)
```

**Val accuracy: 99.4%** — 30 epochs, AdamW lr=1e-3, cosine annealing, balanced class weights.  
Checkpoint: `app/checkpoints/mitbih_resnet1d.pt` (2.1 MB). Retrain in ~1 min: `python app/train_mitbih.py`.


---

<img width="2234" height="819" alt="image" src="https://github.com/user-attachments/assets/f7259a9b-a8f5-4c68-afab-5162a4fe9bac" />

---

<img width="2085" height="1242" alt="image" src="https://github.com/user-attachments/assets/31a3ec26-fd98-41bb-8438-3743531902bb" />

---

## Dataset

**BIDMC PPG and Respiration Dataset** (PhysioNet, ODC-By 1.0)  
53 ICU subjects · ~8 min each · 125 Hz · signals: `PLETH` (PPG) and `II` (ECG Lead II)

| Split | Subjects | Segments |
|-------|----------|----------|
| Train | 28       | 3,184    |
| Val   | 5        | 452      |
| Test  | 7        | 354      |

~25% of subjects dropped after SQI filtering (noisy ICU PPG). Preprocessing: 4th-order Butterworth bandpass (PPG: 0.5–8 Hz, ECG: 0.5–40 Hz), 4-second windows with 50% overlap, per-window z-score normalisation.

---

## Quickstart

```bash
# 1. Create environment
conda create -n ppg2ecg python=3.11 -y
conda activate ppg2ecg
pip install -r requirements.txt

# 2. Download data
bash scripts/download_bidmc.sh

# 3. Preprocess
python src/data/preprocess.py

# 4. Train  (example: BiLSTM)
python src/train.py --config configs/bilstm.yaml

# 5. Evaluate
python src/evaluate.py --model bilstm --checkpoint checkpoints/bilstm/best.pt

# 6. Launch demo app
conda activate ppg2ecg && streamlit run app/app.py
```

### Train beat classifier (one-time)
```bash
python app/train_mitbih.py
```

---

## Loss Function

```
L = L1(ŷ, y) + 0.5 × L1(|STFT(ŷ)|, |STFT(y)|)
```

STFT: n_fft=128, hop=32, Hann window. Encourages both time-domain accuracy and spectral fidelity.

---

## Training Protocol

- Optimiser: AdamW (lr=1e-4, weight_decay=1e-5)
- Schedule: cosine annealing, T_max=100
- Gradient clipping: 1.0
- Early stopping: patience=10 on val loss
- Batch size: 64 · Seed: 42
- Hardware: Hellbender HPC (RTX GPU, 8 GB VRAM) · float32 throughout

---

## Repository Layout

```
ppg2ecg/
├── src/
│   ├── data/
│   │   ├── bidmc.py            # Dataset class, subject-level splits
│   │   └── preprocess.py       # Filtering, segmentation, SQI
│   ├── models/
│   │   ├── unet.py             # 1D U-Net
│   │   ├── bilstm.py           # BiLSTM seq2seq + attention
│   │   └── transformer.py      # Patch-based Transformer
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Three-tier evaluation
│   └── utils.py                # Seeding, logging, IO helpers
├── app/
│   ├── app.py                  # Streamlit demo (hospital monitor UI)
│   ├── inference.py            # Sliding-window BiLSTM PPG→ECG
│   ├── classifier.py           # Rhythm classification (ResNet1D + rules)
│   ├── train_mitbih.py         # One-time MIT-BIH beat classifier training
│   └── checkpoints/
│       └── mitbih_resnet1d.pt  # Beat classifier (530K params, 99.4% acc)
├── scripts/
│   ├── download_bidmc.sh
│   ├── slurm_train.sh
│   └── gen_training_curves.py
├── configs/
│   ├── unet.yaml · bilstm.yaml · transformer.yaml
├── report/
│   ├── report.md
│   └── training_curves.png
└── checkpoints/
    └── bilstm/best.pt          # Best PPG→ECG model (892K params)
```
