"""Train a ResNet1D beat classifier on MIT-BIH Arrhythmia Database.

AAMI EC57 five-class mapping:
    N  – Normal + bundle-branch block + escape beats
    S  – Supraventricular ectopic (APB, JPB, etc.)
    V  – Ventricular ectopic (PVC, VE)
    F  – Fusion beat
    Q  – Unknown / paced

Run once from the ppg2ecg directory:
    conda run -n ppg2ecg python app/train_mitbih.py

Writes: app/checkpoints/mitbih_resnet1d.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import wfdb
except ImportError:
    raise SystemExit("wfdb not found: conda run -n ppg2ecg pip install wfdb")

MITBIH_DIR = ROOT / "data" / "mitbih"
CKPT_OUT   = ROOT / "app" / "checkpoints" / "mitbih_resnet1d.pt"
FS_MIT     = 360
FS_TARGET  = 125
WIN_SEC    = 2.0    # ±1 s around each R-peak
CLASSES    = ["N", "S", "V", "F", "Q"]

AAMI_MAP: dict[str, int] = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,   # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,             # Supraventricular
    "V": 2, "E": 2,                               # Ventricular
    "F": 3,                                       # Fusion
    "/": 4, "f": 4, "Q": 4, "U": 4, "?": 4,     # Unknown/paced
}

MIT_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234",
]


def _bandpass(sig: np.ndarray, fs: int) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def _resample(sig: np.ndarray) -> np.ndarray:
    """Resample from FS_MIT to FS_TARGET using rational ratio."""
    from math import gcd
    g = gcd(FS_TARGET, FS_MIT)
    return resample_poly(sig, FS_TARGET // g, FS_MIT // g).astype(np.float32)


def load_segments() -> tuple[np.ndarray, np.ndarray]:
    half_win_orig = int(WIN_SEC / 2 * FS_MIT)
    half_win_tgt  = int(WIN_SEC / 2 * FS_TARGET)
    seg_len = 2 * half_win_tgt

    segs: list[np.ndarray] = []
    labels: list[int]      = []

    for rec_id in MIT_RECORDS:
        rec_path = MITBIH_DIR / rec_id
        if not rec_path.with_suffix(".dat").exists():
            continue
        try:
            rec = wfdb.rdrecord(str(rec_path))
            ann = wfdb.rdann(str(rec_path), "atr")
        except Exception:
            continue

        # Use MLII lead (channel 0) for single-lead classification
        raw = rec.p_signal[:, 0].astype(np.float32)
        filt = _bandpass(raw, FS_MIT)
        resampled = _resample(filt)
        n = len(resampled)

        # Scale annotation sample indices to target fs
        scale = FS_TARGET / FS_MIT
        for sym, samp in zip(ann.symbol, ann.sample):
            label = AAMI_MAP.get(sym, -1)
            if label < 0:
                continue
            center = int(samp * scale)
            start  = center - half_win_tgt
            end    = center + half_win_tgt
            if start < 0 or end > n:
                continue
            win = resampled[start:end]
            if len(win) != seg_len:
                continue
            win = (win - win.mean()) / (win.std() + 1e-8)
            segs.append(win)
            labels.append(label)

    X = np.stack(segs).reshape(-1, 1, seg_len)   # (N, 1, T)
    y = np.array(labels, dtype=np.int64)
    print(f"Loaded {len(y)} beats  dist: {np.bincount(y)}")
    return X, y


# ── Lightweight ResNet1D (no fastai dependency) ──────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 7, stride=1, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_c))
            if in_c != out_c or stride != 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class ECGResNet(nn.Module):
    """~200 K-parameter ResNet1D for single-lead ECG beat classification."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.blocks(self.stem(x))).squeeze(-1))


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    X, y = load_segments()

    # Shuffle and split 90/10
    idx = np.random.default_rng(42).permutation(len(y))
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    weights = compute_class_weight("balanced", classes=np.arange(5), y=y_tr)
    w_tensor = torch.tensor(weights, dtype=torch.float32)

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    va_ds = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True,  num_workers=4)
    va_ld = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=4)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    model = ECGResNet(n_classes=5).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=w_tensor.to(dev))
    opt       = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    best_acc = 0.0
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 31):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(xb), yb,
                                        weight=w_tensor.to(dev)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(dev), yb.to(dev)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total   += len(yb)
        acc = correct / total
        print(f"Epoch {epoch:2d}/30  val_acc={acc:.3f}  lr={opt.param_groups[0]['lr']:.2e}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(),
                        "seg_len": X.shape[-1],
                        "fs": FS_TARGET,
                        "classes": CLASSES}, CKPT_OUT)
            print(f"  -> saved (acc={best_acc:.3f})")

    print(f"\nDone. Best val accuracy: {best_acc:.3f}")
    print(f"Checkpoint: {CKPT_OUT}")


if __name__ == "__main__":
    train()
