"""Generate training curves figure for README."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT     = ROOT / "report" / "training_curves.png"

MODELS = {
    "unet":        ("U-Net",        "#4fc3f7"),
    "bilstm":      ("BiLSTM",       "#69f0ae"),
    "transformer": ("Transformer",  "#ea80fc"),
    "bilstm_gan":  ("BiLSTM+GAN",   "#ffab40"),
}

BG      = "#0a0a0a"
PANEL   = "#111111"
GRID    = "#1e1e1e"
TICK    = "#666666"
LABEL   = "#aaaaaa"
WHITE   = "#e8e8e8"

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
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
    train_loss = [h.get("train_loss") or h.get("g_loss") for h in history]

    # Panel styling
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TICK, labelsize=8.5, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)

    # Loss curves
    ax.plot(epochs, train_loss, color=color,  linewidth=2.0, label="Train", alpha=0.9)
    ax.plot(epochs, val_loss,   color=WHITE,  linewidth=1.5, linestyle="--", alpha=0.6, label="Val")

    # Best val marker
    best_idx = int(np.argmin(val_loss))
    bx, by   = epochs[best_idx], val_loss[best_idx]
    ax.scatter(bx, by, color=color, s=70, zorder=6, edgecolors=WHITE, linewidths=0.7)
    # annotation offset: go left if close to right edge
    xoff = 0.5 if bx < epochs[-1] * 0.75 else -0.5
    ha   = "left" if xoff > 0 else "right"
    ax.annotate(f"best {by:.4f}",
                xy=(bx, by), xytext=(bx + xoff, by),
                color=color, fontsize=7.5, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6))

    # GAN discriminator secondary axis
    d_vals = [h.get("d_loss") for h in history]
    if any(v is not None for v in d_vals):
        d_vals = [v for v in d_vals if v is not None]
        ax2 = ax.twinx()
        ax2.set_facecolor(PANEL)
        ax2.spines["right"].set_edgecolor("#ff5252")
        ax2.spines["right"].set_linewidth(0.8)
        for s in ["top","left","bottom"]:
            ax2.spines[s].set_visible(False)
        ax2.tick_params(axis="y", colors="#ff7070", labelsize=7.5, length=3, width=0.6)
        ax2.plot(epochs[:len(d_vals)], d_vals, color="#ff5252",
                 linewidth=1.0, linestyle=":", alpha=0.65, label="Discriminator")
        ax2.set_ylabel("Discriminator loss", color="#ff7070", fontsize=7.5, labelpad=6)

    # Axis labels & title
    ax.set_title(name, color=WHITE, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Epoch", color=LABEL, fontsize=8.5, labelpad=4)
    ax.set_ylabel("Loss", color=LABEL, fontsize=8.5, labelpad=4)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Legend
    leg = ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#2a2a2a",
                    labelcolor=LABEL, loc="upper right",
                    framealpha=0.85, borderpad=0.6, handlelength=1.6)

fig.suptitle("Training & Validation Loss — PPG-to-ECG Models",
             color=WHITE, fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout(pad=1.5, h_pad=2.2, w_pad=1.8)
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT}")
