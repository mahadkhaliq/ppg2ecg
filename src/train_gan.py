"""GAN training entry point (generator = BiLSTM, discriminator = PatchGAN1D).

Reproduces the CardioGAN setup (Sarkar & Etemad, AAAI 2021) with a single
patch discriminator instead of their dual-discriminator design, and LSGAN loss
instead of vanilla BCE for training stability on small datasets.

Usage:
    python -m src.train_gan --config configs/bilstm_gan.yaml

Checkpoint format is identical to train.py (model_state_dict + config), so
evaluate.py works on the saved generator checkpoint without modification.

Loss breakdown:
    L_G = L1(pred, real) + lambda_freq * STFT_L1(pred, real)
          + lambda_adv * MSE(D(pred), 1)          # fool discriminator
    L_D = MSE(D(real), 1) + MSE(D(pred.detach()), 0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bidmc import make_loader                     # noqa: E402
from src.models.bilstm import BiLSTMSeq2Seq                # noqa: E402
from src.models.discriminator import PatchDiscriminator1D  # noqa: E402
from src.utils import (                                    # noqa: E402
    Timer, count_parameters, device, get_logger,
    load_config, save_json, set_seed,
)

LOG = get_logger("train_gan")


# ── Losses ──────────────────────────────────────────────────────────────────

def stft_l1(pred: torch.Tensor, target: torch.Tensor,
            n_fft: int = 128, hop: int = 32) -> torch.Tensor:
    p, t = pred.squeeze(1), target.squeeze(1)
    win = torch.hann_window(n_fft, device=p.device)
    P = torch.stft(p, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, return_complex=True)
    T = torch.stft(t, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, return_complex=True)
    return torch.mean(torch.abs(P.abs() - T.abs()))


def recon_loss(pred, real, l1_w, freq_w):
    l1 = nn.functional.l1_loss(pred, real)
    return l1_w * l1 + freq_w * stft_l1(pred, real)


def lsgan_loss(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """LSGAN: MSE to 1 (real) or 0 (fake). Spectral norm handles stability."""
    target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return nn.functional.mse_loss(logits, target)


# ── Train / val passes ───────────────────────────────────────────────────────

def train_epoch(G, D, loader, opt_G, opt_D, cfg_t, dev):
    G.train(); D.train()
    l1_w  = cfg_t["loss_l1_weight"]
    freq_w = cfg_t["loss_freq_weight"]
    adv_w  = cfg_t["lambda_adv"]
    clip   = cfg_t["grad_clip"]

    sum_G = sum_D = n = 0
    d_interval = cfg_t.get("d_update_interval", 3)
    for step, (ppg, ecg) in enumerate(loader):
        ppg = ppg.to(dev, non_blocking=True)
        ecg = ecg.to(dev, non_blocking=True)
        B = ppg.size(0)

        fake = G(ppg)

        # ── Discriminator step (every d_interval generator steps) ──
        if step % d_interval == 0:
            opt_D.zero_grad(set_to_none=True)
            loss_D = 0.5 * (lsgan_loss(D(ecg), True) + lsgan_loss(D(fake.detach()), False))
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), clip)
            opt_D.step()

        # ── Generator step ──
        opt_G.zero_grad(set_to_none=True)
        loss_G = recon_loss(fake, ecg, l1_w, freq_w) + adv_w * lsgan_loss(D(fake), True)
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), clip)
        opt_G.step()

        sum_G += loss_G.item() * B
        sum_D += loss_D.item() * B
        n += B

    return sum_G / n, sum_D / n


def val_epoch(G, loader, cfg_t, dev):
    G.eval()
    l1_w  = cfg_t["loss_l1_weight"]
    freq_w = cfg_t["loss_freq_weight"]
    total = n = 0
    with torch.no_grad():
        for ppg, ecg in loader:
            ppg = ppg.to(dev, non_blocking=True)
            ecg = ecg.to(dev, non_blocking=True)
            fake = G(ppg)
            loss = recon_loss(fake, ecg, l1_w, freq_w)
            total += loss.item() * ppg.size(0)
            n += ppg.size(0)
    return total / max(n, 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    set_seed(cfg["training"]["seed"])
    dev = device()
    LOG.info(f"Device: {dev}")

    cfg_t = cfg["training"]

    # ── Models ──
    G = BiLSTMSeq2Seq(cfg).to(dev)
    D = PatchDiscriminator1D(n_filters=cfg_t.get("n_filters", 64)).to(dev)
    LOG.info(f"Generator:     {count_parameters(G):,} params")
    LOG.info(f"Discriminator: {count_parameters(D):,} params")

    # ── Data ──
    data_dir = Path(cfg["data"]["data_dir"])
    nw = cfg["data"]["num_workers"]
    bs = cfg_t["batch_size"]
    train_loader = make_loader(data_dir / "train.npz", bs, shuffle=True,  num_workers=nw)
    val_loader   = make_loader(data_dir / "val.npz",   bs, shuffle=False, num_workers=nw)
    LOG.info(f"Train: {len(train_loader)} batches  Val: {len(val_loader)} batches")

    # ── Optimizers ──
    opt_G = optim.AdamW(G.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])
    opt_D = optim.AdamW(D.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg_t["epochs"])
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg_t["epochs"])

    # ── Paths ──
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    res_dir  = Path(cfg["paths"]["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, res_dir / "config.json")

    best_val = float("inf")
    patience = cfg_t["patience"]
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg_t["epochs"] + 1):
        with Timer() as t:
            g_loss, d_loss = train_epoch(G, D, train_loader, opt_G, opt_D, cfg_t, dev)
            val_loss = val_epoch(G, val_loader, cfg_t, dev)
        sched_G.step(); sched_D.step()
        lr_now = opt_G.param_groups[0]["lr"]

        LOG.info(
            f"Epoch {epoch:3d}/{cfg_t['epochs']}  "
            f"G={g_loss:.4f}  D={d_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr_now:.2e}  {t.elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch, "g_loss": g_loss, "d_loss": d_loss,
            "val_loss": val_loss, "lr": lr_now, "time_s": t.elapsed,
        })

        # Save generator checkpoint (same format as train.py → evaluate.py works as-is)
        ckpt = {
            "model_state_dict": G.state_dict(),
            "optimizer_state_dict": opt_G.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            LOG.info(f"  -> new best (val={val_loss:.5f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOG.info(f"Early stopping at epoch {epoch}.")
                break

        save_json({"history": history, "best_val": best_val}, res_dir / "metrics.json")

    save_json({"history": history, "best_val": best_val}, res_dir / "metrics.json")
    LOG.info(f"Done. Best val reconstruction loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
