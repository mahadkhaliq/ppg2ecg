"""Training entry point.

Usage:
    python -m src.train --config configs/unet.yaml
    python -m src.train --config configs/bilstm.yaml --epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bidmc import make_loader  # noqa: E402
from src.models.bilstm import BiLSTMSeq2Seq  # noqa: E402
from src.models.transformer import TransformerSeq2Seq  # noqa: E402
from src.models.unet import UNet1D  # noqa: E402
from src.utils import (  # noqa: E402
    Timer, count_parameters, device, get_logger,
    load_config, save_json, set_seed,
)

LOG = get_logger("train")

MODEL_REGISTRY = {
    "unet": UNet1D,
    "bilstm": BiLSTMSeq2Seq,
    "transformer": TransformerSeq2Seq,
}


# ----- Loss -----
def stft_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 128,
    hop: int = 32,
    win_length: int = 128,
) -> torch.Tensor:
    """L1 distance between magnitude STFTs of pred and target.

    Both tensors: (B, 1, T). We squeeze the channel dim.
    """
    p = pred.squeeze(1)
    t = target.squeeze(1)
    window = torch.hann_window(win_length, device=p.device)
    P = torch.stft(p, n_fft=n_fft, hop_length=hop, win_length=win_length,
                   window=window, return_complex=True)
    T = torch.stft(t, n_fft=n_fft, hop_length=hop, win_length=win_length,
                   window=window, return_complex=True)
    return torch.mean(torch.abs(P.abs() - T.abs()))


class CombinedLoss(nn.Module):
    def __init__(self, l1_w: float = 1.0, freq_w: float = 0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_w = l1_w
        self.freq_w = freq_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_w * self.l1(pred, target) + self.freq_w * stft_l1_loss(pred, target)


# ----- Train / val passes -----
def run_epoch(model, loader, criterion, device_, optimizer=None, grad_clip: float = 1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for ppg, ecg in loader:
            ppg = ppg.to(device_, non_blocking=True)
            ecg = ecg.to(device_, non_blocking=True)
            pred = model(ppg)
            loss = criterion(pred, ecg)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * ppg.size(0)
            n += ppg.size(0)
    return total / max(n, 1)


# ----- Main -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override epochs (e.g. for smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    set_seed(cfg["training"]["seed"])
    dev = device()
    LOG.info(f"Device: {dev}")

    model_name = cfg["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(MODEL_REGISTRY)}")
    Model = MODEL_REGISTRY[model_name]
    model = Model(cfg).to(dev)
    LOG.info(f"Model: {model_name} ({count_parameters(model):,} params)")

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    nw = cfg["data"]["num_workers"]
    bs = cfg["training"]["batch_size"]
    train_loader = make_loader(data_dir / "train.npz", bs, shuffle=True, num_workers=nw)
    val_loader = make_loader(data_dir / "val.npz", bs, shuffle=False, num_workers=nw)
    LOG.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # Optimizer
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])
    criterion = CombinedLoss(
        l1_w=cfg["training"]["loss_l1_weight"],
        freq_w=cfg["training"]["loss_freq_weight"],
    )

    # Paths
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    res_dir = Path(cfg["paths"]["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, res_dir / "config.json")

    # Loop
    best_val = float("inf")
    patience = cfg["training"]["patience"]
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        with Timer() as t:
            train_loss = run_epoch(
                model, train_loader, criterion, dev,
                optimizer=opt, grad_clip=cfg["training"]["grad_clip"],
            )
            val_loss = run_epoch(model, val_loader, criterion, dev)
        sched.step()
        lr_now = opt.param_groups[0]["lr"]

        LOG.info(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"lr={lr_now:.2e}  time={t.elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "lr": lr_now, "time_s": t.elapsed,
        })

        # Checkpoint
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            LOG.info(f"  -> new best (val={val_loss:.5f}), saved {ckpt_dir/'best.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOG.info(f"Early stopping after {patience} epochs without improvement.")
                break

        save_json({"history": history, "best_val": best_val}, res_dir / "metrics.json")

    save_json({"history": history, "best_val": best_val}, res_dir / "metrics.json")
    LOG.info(f"Training complete. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
