"""Generate training curves figure for README."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "training_curves.png"

MODELS = {
    "unet":        ("U-Net",        "#4fc3f7"),
    "bilstm":      ("BiLSTM",       "#a5d6a7"),
    "transformer": ("Transformer",  "#ce93d8"),
    "bilstm_gan":  ("BiLSTM+GAN",   "#ffb74d"),
}

BG = "#0d0d0d"
GRID = "#2a2a2a"

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, (key, (name, color)) in zip(axes, MODELS.items()):
    metrics_path = RESULTS / key / "metrics.json"
    if not metrics_path.exists():
        ax.set_visible(False)
        continue

    with open(metrics_path) as f:
        data = json.load(f)

    history    = data["history"]
    epochs     = [h["epoch"] for h in history]
    val_loss   = [h["val_loss"] for h in history]

    # GAN uses g_loss as train loss
    train_loss = [h.get("train_loss") or h.get("g_loss") for h in history]

    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color("#eeeeee")

    ax.plot(epochs, train_loss, color=color,     linewidth=1.8, label="Train loss")
    ax.plot(epochs, val_loss,   color="white",   linewidth=1.4, linestyle="--", alpha=0.75, label="Val loss")

    # Mark best val epoch
    best_idx = int(np.argmin(val_loss))
    ax.scatter(epochs[best_idx], val_loss[best_idx], color=color, s=60, zorder=5)
    ax.annotate(f"  best={val_loss[best_idx]:.4f}",
                xy=(epochs[best_idx], val_loss[best_idx]),
                color=color, fontsize=8)

    # GAN: discriminator loss on secondary y-axis
    d_loss_vals = [h.get("d_loss") for h in history]
    if any(v is not None for v in d_loss_vals):
        history_has_d = True
        d_loss_vals = [v for v in d_loss_vals if v is not None]
    else:
        history_has_d = False

    if history_has_d:
        ax2 = ax.twinx()
        ax2.set_facecolor(BG)
        ax2.tick_params(colors="#ff7070", labelsize=8)
        ax2.spines["right"].set_edgecolor("#ff7070")
        ax2.plot(epochs[:len(d_loss_vals)], d_loss_vals, color="#ff5252", linewidth=1.0,
                 linestyle=":", alpha=0.7, label="Discriminator loss")
        ax2.set_ylabel("D loss", color="#ff7070", fontsize=8)
        ax2.yaxis.label.set_color("#ff7070")
        ax2.tick_params(axis="y", colors="#ff5252")

    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GRID, labelcolor="#cccccc")

fig.suptitle("Training & Validation Loss", color="#eeeeee", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT}")
