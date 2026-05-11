"""Three-tier evaluation on the test set.

Tier 1 — Waveform fidelity:  RMSE, Pearson r, DTW (subsampled)
Tier 2 — Morphology:         R-peak F1, RR-interval error
Tier 3 — Downstream task:    Heart-rate-bucket classification accuracy on
                              real vs reconstructed ECG.

Usage:
    python -m src.evaluate --model unet
    python -m src.evaluate --model bilstm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bidmc import BIDMCDataset, make_loader  # noqa: E402
from src.models.bilstm import BiLSTMSeq2Seq  # noqa: E402
from src.models.transformer import TransformerSeq2Seq  # noqa: E402
from src.models.unet import UNet1D  # noqa: E402
from src.utils import device, get_logger, save_json, set_seed  # noqa: E402

LOG = get_logger("eval")

MODEL_REGISTRY = {
    "unet": UNet1D,
    "bilstm": BiLSTMSeq2Seq,
    "transformer": TransformerSeq2Seq,
}

FS = 125  # Sampling rate


# ---------- Tier 1: Waveform ----------
def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred - pred.mean()
    t = target - target.mean()
    denom = np.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-12
    return float((p * t).sum() / denom)


def dtw_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute DTW distance. Uses dtaidistance if installed, else numpy fallback."""
    try:
        from dtaidistance import dtw
        return float(dtw.distance_fast(pred.astype(np.float64), target.astype(np.float64), use_pruning=True))
    except ImportError:
        # O(N^2) fallback — only for small N
        n, m = len(pred), len(target)
        if n > 600 or m > 600:
            return float("nan")
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(pred[i - 1] - target[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return float(D[n, m])


# ---------- Tier 2: Morphology ----------
def detect_r_peaks(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Detect R-peaks. Falls back to a simple threshold detector if neurokit2 unavailable."""
    try:
        import neurokit2 as nk
        # ecg_findpeaks expects a 1D signal
        clean = nk.ecg_clean(signal, sampling_rate=fs)
        peaks_dict = nk.ecg_findpeaks(clean, sampling_rate=fs)
        return np.asarray(peaks_dict.get("ECG_R_Peaks", []), dtype=np.int64)
    except Exception:
        # Simple amplitude-threshold fallback
        sig = (signal - signal.mean()) / (signal.std() + 1e-8)
        thr = 1.5
        # Refractory period of 0.3s
        refractory = int(0.3 * fs)
        peaks = []
        last = -refractory
        for i in range(1, len(sig) - 1):
            if sig[i] > thr and sig[i] > sig[i - 1] and sig[i] > sig[i + 1] and (i - last) > refractory:
                peaks.append(i)
                last = i
        return np.array(peaks, dtype=np.int64)


def r_peak_f1(true_peaks: np.ndarray, pred_peaks: np.ndarray, tolerance_samples: int = 6) -> float:
    """F1 of R-peak detection. tolerance_samples=6 ≈ 48ms at 125 Hz."""
    if len(true_peaks) == 0 and len(pred_peaks) == 0:
        return 1.0
    if len(true_peaks) == 0 or len(pred_peaks) == 0:
        return 0.0

    matched_true = np.zeros(len(true_peaks), dtype=bool)
    matched_pred = np.zeros(len(pred_peaks), dtype=bool)
    # Greedy nearest match
    for i, p in enumerate(pred_peaks):
        diffs = np.abs(true_peaks - p)
        valid = (~matched_true) & (diffs <= tolerance_samples)
        if valid.any():
            j = np.argmin(np.where(valid, diffs, np.inf))
            matched_true[j] = True
            matched_pred[i] = True

    tp = int(matched_pred.sum())
    fp = len(pred_peaks) - tp
    fn = len(true_peaks) - tp
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return float(2 * prec * rec / (prec + rec + 1e-12))


def rr_interval_error_ms(true_peaks: np.ndarray, pred_peaks: np.ndarray, fs: int = FS) -> float:
    """Median absolute error in RR intervals (ms). Returns NaN if too few peaks."""
    if len(true_peaks) < 2 or len(pred_peaks) < 2:
        return float("nan")
    rr_true = np.diff(true_peaks) / fs * 1000.0
    rr_pred = np.diff(pred_peaks) / fs * 1000.0
    n = min(len(rr_true), len(rr_pred))
    return float(np.median(np.abs(rr_true[:n] - rr_pred[:n])))


# ---------- Tier 3: Downstream classification ----------
def hr_from_peaks(peaks: np.ndarray, signal_len: int, fs: int = FS) -> float:
    """Heart rate (bpm) from R-peaks."""
    if len(peaks) < 2:
        return float("nan")
    duration_min = signal_len / fs / 60.0
    return (len(peaks) - 1) / duration_min


def hr_bucket(hr: float) -> int:
    """Bucket: 0=brady (<60), 1=normal (60-100), 2=tachy (>100). NaN -> 1."""
    if not np.isfinite(hr):
        return 1
    if hr < 60:
        return 0
    if hr > 100:
        return 2
    return 1


def downstream_accuracy(real_signals: np.ndarray, recon_signals: np.ndarray, fs: int = FS) -> dict:
    """Compute heart-rate-bucket classification agreement.

    The "task" is bucket prediction. Real ECG defines the ground-truth bucket;
    we then derive the same bucket from the reconstructed ECG and report
    accuracy. Also reports the same-class confusion.
    """
    n = len(real_signals)
    real_buckets = np.zeros(n, dtype=np.int64)
    recon_buckets = np.zeros(n, dtype=np.int64)
    for i in range(n):
        rp_real = detect_r_peaks(real_signals[i], fs)
        rp_recon = detect_r_peaks(recon_signals[i], fs)
        real_buckets[i] = hr_bucket(hr_from_peaks(rp_real, len(real_signals[i]), fs))
        recon_buckets[i] = hr_bucket(hr_from_peaks(rp_recon, len(recon_signals[i]), fs))

    acc = float(np.mean(real_buckets == recon_buckets))
    # Per-class breakdown
    classes = [0, 1, 2]
    per_class = {}
    for c in classes:
        mask = real_buckets == c
        if mask.sum() == 0:
            per_class[f"class_{c}_acc"] = float("nan")
            per_class[f"class_{c}_n"] = 0
        else:
            per_class[f"class_{c}_acc"] = float(np.mean(recon_buckets[mask] == c))
            per_class[f"class_{c}_n"] = int(mask.sum())
    return {"accuracy": acc, **per_class}


# ---------- Inference ----------
def run_inference(model: torch.nn.Module, loader: DataLoader, device_) -> tuple[np.ndarray, np.ndarray]:
    """Returns real_ecg and pred_ecg arrays of shape (N, T)."""
    model.eval()
    real_all, pred_all = [], []
    with torch.no_grad():
        for ppg, ecg in loader:
            ppg = ppg.to(device_, non_blocking=True)
            pred = model(ppg).cpu().numpy().squeeze(1)  # (B, T)
            real_all.append(ecg.numpy().squeeze(1))
            pred_all.append(pred)
    return np.concatenate(real_all), np.concatenate(pred_all)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--ckpt", default=None, help="path to checkpoint (default: checkpoints/<model>/best.pt)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default=None, help="default: results/<model>")
    parser.add_argument("--dtw_subsample", type=int, default=200, help="number of segments for DTW")
    args = parser.parse_args()

    set_seed(42)
    dev = device()

    ckpt_path = Path(args.ckpt or f"checkpoints/{args.model}/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    LOG.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt["config"]

    Model = MODEL_REGISTRY[args.model]
    model = Model(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loader = make_loader(Path(args.data_dir) / "test.npz", batch_size=64, shuffle=False, num_workers=2)
    LOG.info(f"Test batches: {len(test_loader)}")

    real, pred = run_inference(model, test_loader, dev)
    LOG.info(f"Inference complete. Shape: {real.shape}")

    # Tier 1
    LOG.info("Tier 1: Waveform fidelity...")
    rmse_per = [rmse(pred[i], real[i]) for i in range(len(real))]
    pearson_per = [pearson_r(pred[i], real[i]) for i in range(len(real))]
    # DTW: subsample
    rng = np.random.default_rng(42)
    dtw_idx = rng.choice(len(real), size=min(args.dtw_subsample, len(real)), replace=False)
    dtw_per = [dtw_distance(pred[i], real[i]) for i in dtw_idx]

    tier1 = {
        "rmse_mean": float(np.mean(rmse_per)),
        "rmse_std": float(np.std(rmse_per)),
        "pearson_mean": float(np.nanmean(pearson_per)),
        "pearson_std": float(np.nanstd(pearson_per)),
        "dtw_mean": float(np.nanmean(dtw_per)),
        "dtw_std": float(np.nanstd(dtw_per)),
        "n_segments": int(len(real)),
        "dtw_n_subsample": int(len(dtw_idx)),
    }
    LOG.info(f"  RMSE: {tier1['rmse_mean']:.4f} ± {tier1['rmse_std']:.4f}")
    LOG.info(f"  Pearson r: {tier1['pearson_mean']:.4f} ± {tier1['pearson_std']:.4f}")
    LOG.info(f"  DTW: {tier1['dtw_mean']:.2f} ± {tier1['dtw_std']:.2f}")

    # Tier 2
    LOG.info("Tier 2: Morphology...")
    f1_per, rr_err_per = [], []
    for i in range(len(real)):
        rp_real = detect_r_peaks(real[i])
        rp_pred = detect_r_peaks(pred[i])
        f1_per.append(r_peak_f1(rp_real, rp_pred))
        rr_err_per.append(rr_interval_error_ms(rp_real, rp_pred))
    tier2 = {
        "rpeak_f1_mean": float(np.mean(f1_per)),
        "rpeak_f1_std": float(np.std(f1_per)),
        "rr_err_ms_median": float(np.nanmedian(rr_err_per)),
        "rr_err_ms_iqr": float(np.nanpercentile(rr_err_per, 75) - np.nanpercentile(rr_err_per, 25)),
    }
    LOG.info(f"  R-peak F1: {tier2['rpeak_f1_mean']:.4f} ± {tier2['rpeak_f1_std']:.4f}")
    LOG.info(f"  RR-interval err (ms): median={tier2['rr_err_ms_median']:.1f}, IQR={tier2['rr_err_ms_iqr']:.1f}")

    # Tier 3
    LOG.info("Tier 3: Downstream HR-bucket classification...")
    tier3 = downstream_accuracy(real, pred)
    LOG.info(f"  HR-bucket accuracy: {tier3['accuracy']:.4f}")

    # Save
    res_dir = Path(args.results_dir or f"results/{args.model}")
    res_dir.mkdir(parents=True, exist_ok=True)
    save_json({"tier1": tier1, "tier2": tier2, "tier3": tier3}, res_dir / "eval_metrics.json")

    # Save 5 random qualitative examples
    qual_idx = rng.choice(len(real), size=min(5, len(real)), replace=False)
    np.savez_compressed(
        res_dir / "qualitative.npz",
        real=real[qual_idx],
        pred=pred[qual_idx],
        idx=qual_idx,
    )
    LOG.info(f"Wrote evaluation results to {res_dir}")


if __name__ == "__main__":
    main()
