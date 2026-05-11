"""Model inference helpers — load checkpoint and run PPG→ECG reconstruction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.bilstm import BiLSTMSeq2Seq  # noqa: E402

FS = 125
WINDOW_LEN = 500
STRIDE_LEN = 250
PPG_BAND = (0.5, 8.0)
ECG_BAND = (0.5, 40.0)


def _bandpass(signal: np.ndarray, low: float, high: float) -> np.ndarray:
    nyq = FS / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def load_model(ckpt_path: str | Path) -> tuple[BiLSTMSeq2Seq, torch.device]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=dev)
    cfg = ckpt["config"]
    cfg["model"]["name"] = "bilstm"
    model = BiLSTMSeq2Seq(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, dev


def reconstruct_ecg(
    ppg_raw: np.ndarray,
    model: BiLSTMSeq2Seq,
    dev: torch.device,
) -> np.ndarray:
    """Reconstruct a full-length ECG from raw PPG using sliding windows.

    Returns an ECG array of the same length as ppg_raw, assembled by
    averaging overlapping window predictions.
    """
    ppg_f = _bandpass(ppg_raw, *PPG_BAND)
    n = len(ppg_f)
    ecg_sum = np.zeros(n, dtype=np.float64)
    ecg_cnt = np.zeros(n, dtype=np.float64)

    starts = list(range(0, n - WINDOW_LEN + 1, STRIDE_LEN))
    if not starts:
        return np.zeros(n, dtype=np.float32)

    for start in starts:
        end = start + WINDOW_LEN
        win = ppg_f[start:end]
        win_norm = (win - win.mean()) / (win.std() + 1e-8)
        x = torch.tensor(win_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()
        ecg_sum[start:end] += pred
        ecg_cnt[start:end] += 1.0

    mask = ecg_cnt > 0
    ecg_out = np.zeros(n, dtype=np.float32)
    ecg_out[mask] = (ecg_sum[mask] / ecg_cnt[mask]).astype(np.float32)
    return ecg_out


def filter_ecg(ecg_raw: np.ndarray) -> np.ndarray:
    return _bandpass(ecg_raw, *ECG_BAND)
