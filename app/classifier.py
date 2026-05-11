"""ECG rhythm classifier.

Primary path: deep learning beat classifier (ResNet1D trained on MIT-BIH,
AAMI 5-class N/S/V/F/Q) combined with NeuroKit2 R-peak detection.

Fallback: pure rule-based approach if the DL checkpoint is absent.

Classification logic (on aggregate beat predictions):
    1. Signal too short / no R-peaks detected    → Unanalysable
    2. HR < 60 bpm                               → Bradycardia
    3. HR > 100 bpm                              → Tachycardia
    4. RR CV > 0.20 AND pNN50 > 0.50            → Probable Atrial Fibrillation
    5. V or S beat fraction > 0.10              → Frequent Ectopy
    6. Otherwise                                 → Normal Sinus Rhythm
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import neurokit2 as nk
import numpy as np
import torch
import torch.nn as nn

FS          = 125
MIN_BEATS   = 3
WIN_SEC     = 2.0    # must match train_mitbih.py
CLASSES     = ["N", "S", "V", "F", "Q"]
CKPT_PATH   = Path(__file__).resolve().parent / "checkpoints" / "mitbih_resnet1d.pt"


# ── ResNet1D architecture (must match train_mitbih.py) ───────────────────────

class _ResBlock(nn.Module):
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


class _ECGResNet(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _ResBlock(32, 32),
            _ResBlock(32, 64, stride=2),
            _ResBlock(64, 64),
            _ResBlock(64, 128, stride=2),
            _ResBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.blocks(self.stem(x))).squeeze(-1))


# ── Load DL model (cached) ───────────────────────────────────────────────────

_dl_model: _ECGResNet | None = None
_dl_dev:   torch.device | None = None
_seg_len:  int = int(WIN_SEC * FS)


def _load_dl_model() -> tuple[_ECGResNet | None, torch.device]:
    global _dl_model, _dl_dev
    if _dl_model is not None:
        return _dl_model, _dl_dev
    if not CKPT_PATH.exists():
        return None, None
    try:
        dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(CKPT_PATH, map_location=dev)
        m    = _ECGResNet(n_classes=5).to(dev)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        _dl_model = m
        _dl_dev   = dev
        return m, dev
    except Exception:
        return None, None


def _classify_beats_dl(ecg: np.ndarray, r_peaks: list[int],
                       model: _ECGResNet, dev: torch.device) -> np.ndarray:
    """Return AAMI class index per beat using the DL model."""
    half = _seg_len // 2
    n    = len(ecg)
    preds: list[int] = []
    for p in r_peaks:
        start = p - half
        end   = p + half
        if start < 0 or end > n:
            preds.append(0)   # treat edge beats as Normal
            continue
        win = ecg[start:end].copy()
        win = (win - win.mean()) / (win.std() + 1e-8)
        x   = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            idx = model(x).argmax(1).item()
        preds.append(idx)
    return np.array(preds, dtype=np.int64)


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RhythmResult:
    label: str
    confidence: str               # "High" / "Medium" / "Low"
    heart_rate: float
    rr_intervals_ms: list[float]
    r_peaks: list[int]
    hrv: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    beat_labels: list[str] = field(default_factory=list)   # per-beat AAMI class


# ── Main classifier ──────────────────────────────────────────────────────────

def classify(ecg: np.ndarray, fs: int = FS, reconstructed: bool = False) -> RhythmResult:
    """Classify cardiac rhythm from a single-lead ECG array."""
    notes: list[str] = []
    if reconstructed:
        notes.append("ECG reconstructed from PPG — confidence reduced.")

    # ── R-peak detection ──────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_process(ecg.astype(float), sampling_rate=fs)
        r_peaks: list[int] = info["ECG_R_Peaks"].tolist()
    except Exception:
        return RhythmResult(
            label="Unanalysable", confidence="Low",
            heart_rate=float("nan"), rr_intervals_ms=[], r_peaks=[],
            notes=["R-peak detection failed."],
        )

    if len(r_peaks) < MIN_BEATS:
        return RhythmResult(
            label="Unanalysable", confidence="Low",
            heart_rate=float("nan"), rr_intervals_ms=[], r_peaks=r_peaks,
            notes=[f"Too few beats detected ({len(r_peaks)})."],
        )

    # ── RR intervals + HRV ───────────────────────────────────────────────
    rr_samples = np.diff(r_peaks)
    rr_ms      = (rr_samples / fs * 1000).tolist()
    hr         = 60.0 / (np.mean(rr_samples) / fs)

    hrv: dict[str, float] = {}
    rr_arr = np.array(rr_ms)
    hrv["SDNN_ms"]  = float(np.std(rr_arr))
    hrv["RMSSD_ms"] = float(np.sqrt(np.mean(np.diff(rr_arr) ** 2)))
    hrv["CV"]       = float(np.std(rr_arr) / np.mean(rr_arr))
    diffs           = np.abs(np.diff(rr_arr))
    hrv["pNN50"]    = float(np.sum(diffs > 50) / len(diffs)) if len(diffs) > 0 else 0.0

    cv    = hrv["CV"]
    pnn50 = hrv["pNN50"]

    # ── DL beat classification ────────────────────────────────────────────
    beat_labels: list[str] = []
    ectopy_frac = 0.0
    dl_active   = False

    model, dev = _load_dl_model()
    if model is not None:
        beat_idx  = _classify_beats_dl(ecg, r_peaks, model, dev)
        beat_labels = [CLASSES[i] for i in beat_idx]
        n_beats   = len(beat_idx)
        n_ectopic = int(np.sum((beat_idx == 1) | (beat_idx == 2)))   # S or V
        ectopy_frac = n_ectopic / n_beats if n_beats > 0 else 0.0
        dl_active = True
        notes.append(f"DL beat classifier (ResNet1D/MIT-BIH): "
                      f"{n_ectopic}/{n_beats} ectopic beats detected.")

    # ── Classification rules ──────────────────────────────────────────────
    if hr < 60:
        label = "Bradycardia"
        conf  = "High" if hr < 50 else "Medium"
    elif hr > 100:
        label = "Tachycardia"
        conf  = "High" if hr > 120 else "Medium"
    elif cv > 0.20 and pnn50 > 0.50:
        label = "Probable Atrial Fibrillation"
        conf  = "Medium"
        notes.append("Irregular RR intervals detected. Confirm with 12-lead ECG.")
    elif ectopy_frac > 0.10:
        label = "Frequent Ectopy"
        conf  = "Medium" if dl_active else "Low"
    else:
        label = "Normal Sinus Rhythm"
        conf  = "High" if cv < 0.08 else "Medium"

    if reconstructed and conf == "High":
        conf = "Medium"

    if not dl_active:
        notes.append("DL classifier not loaded — rule-based fallback active. "
                      "Run app/train_mitbih.py to enable.")

    return RhythmResult(
        label=label,
        confidence=conf,
        heart_rate=round(hr, 1),
        rr_intervals_ms=[round(r, 1) for r in rr_ms],
        r_peaks=r_peaks,
        hrv=hrv,
        notes=notes,
        beat_labels=beat_labels,
    )
