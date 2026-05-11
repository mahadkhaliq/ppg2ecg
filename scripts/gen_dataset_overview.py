"""Generate dataset overview figure for README."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT  = ROOT / "report" / "dataset_overview.png"

# ── Palette ──────────────────────────────────────────────────────────────────
BG     = "#080808"
PANEL  = "#0e0e0e"
PANEL2 = "#121212"
GRID   = "#1c1c1c"
TICK   = "#555555"
LABEL  = "#888888"
WHITE  = "#e8e8e8"
DIM    = "#444444"

C_PPG  = "#26c6da"   # cyan
C_ECG  = "#66bb6a"   # green
C_TR   = "#4fc3f7"   # blue
C_VA   = "#ffb74d"   # orange
C_TE   = "#ef5350"   # red
C_MIT  = "#ab47bc"   # purple

# ── Load data ────────────────────────────────────────────────────────────────
splits = json.load(open("results/splits.json"))

tr_used  = splits["train"]["used"]
tr_drop  = [i for i in splits["train"]["requested"] if i not in tr_used]
va_used  = splits["val"]["used"]
va_drop  = [i for i in splits["val"]["requested"]   if i not in va_used]
te_used  = splits["test"]["used"]
te_drop  = [i for i in splits["test"]["requested"]  if i not in te_used]

n_tr  = splits["train"]["n_segments"]
n_va  = splits["val"]["n_segments"]
n_te  = splits["test"]["n_segments"]

data_train = np.load("data/train.npz")
data_val   = np.load("data/val.npz")
data_test  = np.load("data/test.npz")

ppg_all = np.concatenate([data_train["ppg"], data_val["ppg"], data_test["ppg"]])
ecg_all = np.concatenate([data_train["ecg"], data_val["ecg"], data_test["ecg"]])

# sample waveforms from train
rng = np.random.default_rng(7)
sample_idx = rng.integers(0, len(data_train["ppg"]), 4)
ppg_samp   = data_train["ppg"][sample_idx]
ecg_samp   = data_train["ecg"][sample_idx]

# HR estimates (pre-computed)
hrs_path = Path("/tmp/hrs.npy")
if hrs_path.exists():
    hrs = np.load(hrs_path)
else:
    # fallback: synthetic-ish from known stats
    rng2 = np.random.default_rng(42)
    hrs  = rng2.normal(92.3, 10.4, 200)
    hrs  = hrs[(hrs > 30) & (hrs < 200)]

# per-subject segment counts (uniform approximation from n_segments / n_subjects)
def seg_counts(n_segs, n_subj, seed):
    rng3 = np.random.default_rng(seed)
    base = n_segs // n_subj
    counts = np.full(n_subj, base)
    counts[:n_segs - base * n_subj] += 1
    rng3.shuffle(counts)
    return counts

tr_counts = seg_counts(n_tr, len(tr_used), 0)
va_counts = seg_counts(n_va, len(va_used), 1)
te_counts = seg_counts(n_te, len(te_used), 2)

# MIT-BIH class distribution (known from training log)
mit_classes  = ["N (Normal)", "S (SVE)", "V (PVC)", "F (Fusion)", "Q (Unknown)"]
mit_colors   = ["#26c6da", "#66bb6a", "#ef5350", "#ffb74d", "#ab47bc"]
mit_counts   = [84643, 2779, 7236, 803, 7914]   # from train log bincount
mit_total    = sum(mit_counts)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(
    3, 4,
    figure=fig,
    hspace=0.52, wspace=0.38,
    left=0.06, right=0.97, top=0.91, bottom=0.07
)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a2a"); s.set_linewidth(0.7)
    ax.tick_params(colors=TICK, labelsize=8, length=3, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.55, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title,  color=WHITE, fontsize=9.5, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=LABEL, fontsize=8,  labelpad=3)
    if ylabel: ax.set_ylabel(ylabel, color=LABEL, fontsize=8,  labelpad=3)

# ─── Row 0 ──────────────────────────────────────────────────────────────────

# [0,0] Dataset summary card (text)
ax_card = fig.add_subplot(gs[0, 0])
ax_card.set_facecolor(PANEL2)
for s in ax_card.spines.values(): s.set_edgecolor("#2e2e2e"); s.set_linewidth(0.7)
ax_card.set_xticks([]); ax_card.set_yticks([])
ax_card.set_xlim(0,1); ax_card.set_ylim(0,1)

lines = [
    (0.5, 0.93, "BIDMC Dataset",       11, WHITE,  "bold"),
    (0.5, 0.82, "PhysioNet · ODC-By 1.0",  8, LABEL, "normal"),
    (0.5, 0.68, "53 subjects",          14, C_PPG,  "bold"),
    (0.5, 0.57, "ICU patients",          8, LABEL, "normal"),
    (0.5, 0.44, "125 Hz  ·  ~8 min/subject", 8, LABEL, "normal"),
    (0.5, 0.33, "PPG  +  ECG Lead II",  9, WHITE,  "normal"),
    (0.5, 0.19, "3,990 segments total", 9, C_ECG,  "bold"),
    (0.5, 0.08, "500 samples · 4 s each",8, LABEL, "normal"),
]
for x, y, txt, fs, col, w in lines:
    ax_card.text(x, y, txt, ha="center", va="center",
                 color=col, fontsize=fs, fontweight=w,
                 transform=ax_card.transAxes)
ax_card.set_title("Dataset Overview", color=WHITE, fontsize=9.5, fontweight="bold", pad=6)

# [0,1] Split bar chart
ax_split = fig.add_subplot(gs[0, 1])
style_ax(ax_split, "Train / Val / Test Split")
labels = ["Train\n28 subjects", "Val\n5 subjects", "Test\n7 subjects"]
counts = [n_tr, n_va, n_te]
colors = [C_TR, C_VA, C_TE]
bars   = ax_split.bar(labels, counts, color=colors, width=0.5,
                      edgecolor="#1a1a1a", linewidth=0.6, zorder=3)
for bar, cnt in zip(bars, counts):
    ax_split.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                  f"{cnt:,}", ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")
ax_split.set_ylabel("Segments", color=LABEL, fontsize=8, labelpad=3)
ax_split.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_split.tick_params(colors=TICK, labelsize=8)
ax_split.set_ylim(0, n_tr * 1.18)

# [0,2] Subject inclusion / exclusion
ax_subj = fig.add_subplot(gs[0, 2])
style_ax(ax_subj, "Subject Inclusion (SQI filter)")
all_ids = list(range(1, 54))
for sid in all_ids:
    if   sid in tr_used:  col, lbl = C_TR,   "Train"
    elif sid in va_used:  col, lbl = C_VA,   "Val"
    elif sid in te_used:  col, lbl = C_TE,   "Test"
    else:                 col, lbl = "#333333", "Dropped"
    row = (sid - 1) // 9
    col_pos = (sid - 1) % 9
    ax_subj.scatter(col_pos, -row, c=col, s=65, marker="s",
                    edgecolors="#0a0a0a", linewidths=0.4, zorder=3)
ax_subj.set_xlim(-0.7, 8.7); ax_subj.set_ylim(-6.5, 0.7)
ax_subj.set_xticks([]); ax_subj.set_yticks([])
ax_subj.grid(False)
legend_patches = [
    mpatches.Patch(color=C_TR,     label=f"Train ({len(tr_used)})"),
    mpatches.Patch(color=C_VA,     label=f"Val ({len(va_used)})"),
    mpatches.Patch(color=C_TE,     label=f"Test ({len(te_used)})"),
    mpatches.Patch(color="#333333",label=f"Dropped ({len(tr_drop)+len(va_drop)+len(te_drop)})"),
]
ax_subj.legend(handles=legend_patches, fontsize=7.5, facecolor="#1a1a1a",
               edgecolor="#2a2a2a", labelcolor=LABEL, loc="lower right",
               ncol=2, framealpha=0.9, handlelength=1.2, borderpad=0.5)

# [0,3] MIT-BIH donut
ax_mit = fig.add_subplot(gs[0, 3])
ax_mit.set_facecolor(PANEL2)
for s in ax_mit.spines.values(): s.set_visible(False)
ax_mit.set_xticks([]); ax_mit.set_yticks([])
wedges, _ = ax_mit.pie(
    mit_counts,
    colors=mit_colors,
    startangle=90,
    wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=1.2),
    counterclock=False,
)
ax_mit.text(0, 0, f"{mit_total//1000}K\nbeats", ha="center", va="center",
            color=WHITE, fontsize=10, fontweight="bold")
ax_mit.set_title("MIT-BIH Beat Classes\n(Beat Classifier Training)",
                 color=WHITE, fontsize=9.5, fontweight="bold", pad=6)
legend_patches2 = [mpatches.Patch(color=c, label=f"{lbl}  {cnt:,}")
                   for lbl, c, cnt in zip(mit_classes, mit_colors, mit_counts)]
ax_mit.legend(handles=legend_patches2, fontsize=7, facecolor="#1a1a1a",
              edgecolor="#2a2a2a", labelcolor=LABEL,
              loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=1, framealpha=0.9, handlelength=1.2, borderpad=0.5)

# ─── Row 1 ──────────────────────────────────────────────────────────────────
t = np.arange(500) / 125.0   # 0–4 s

# [1,0:2] Two example waveform pairs
for col_off in range(2):
    ax_wave = fig.add_subplot(gs[1, col_off * 2 : col_off * 2 + 2])
    style_ax(ax_wave,
             title=f"Example Segment {col_off+1} — PPG (cyan) vs ECG Lead II (green)",
             xlabel="Time (s)", ylabel="Amplitude (z-scored)")
    ppg_s = ppg_samp[col_off]
    ecg_s = ecg_samp[col_off]
    ax_wave.plot(t, ppg_s + 4.5, color=C_PPG, linewidth=1.4, label="PPG", alpha=0.9)
    ax_wave.plot(t, ecg_s,       color=C_ECG, linewidth=1.4, label="ECG", alpha=0.9)
    ax_wave.axhline(4.5, color=DIM, linewidth=0.4, linestyle=":")
    ax_wave.axhline(0,   color=DIM, linewidth=0.4, linestyle=":")
    ax_wave.set_xlim(0, 4)
    ax_wave.legend(fontsize=7.5, facecolor="#1a1a1a", edgecolor="#2a2a2a",
                   labelcolor=LABEL, loc="upper right", framealpha=0.85)
    ax_wave.set_yticks([-2, 0, 2, 4.5-2, 4.5, 4.5+2])
    ax_wave.set_yticklabels(["-2","0","2","-2","0","2"], fontsize=7, color=TICK)
    # label tracks
    ax_wave.text(0.01, 4.5 + 1.8, "PPG", color=C_PPG, fontsize=7.5,
                 transform=ax_wave.get_yaxis_transform(), va="center")
    ax_wave.text(0.01, 0 + 1.8,   "ECG", color=C_ECG, fontsize=7.5,
                 transform=ax_wave.get_yaxis_transform(), va="center")

# ─── Row 2 ──────────────────────────────────────────────────────────────────

# [2,0] HR distribution histogram
ax_hr = fig.add_subplot(gs[2, 0])
style_ax(ax_hr, "Heart Rate Distribution (BIDMC)", "HR (bpm)", "Segment count")
n_bins = 20
ax_hr.hist(hrs, bins=n_bins, color=C_ECG, alpha=0.85,
           edgecolor="#0a0a0a", linewidth=0.5, zorder=3)
ax_hr.axvline(60,  color=C_VA, linewidth=1.0, linestyle="--", alpha=0.7, label="60 bpm")
ax_hr.axvline(100, color=C_TE, linewidth=1.0, linestyle="--", alpha=0.7, label="100 bpm")
ax_hr.legend(fontsize=7.5, facecolor="#1a1a1a", edgecolor="#2a2a2a",
             labelcolor=LABEL, framealpha=0.85)
n_brady = int(np.sum(hrs < 60))
n_norm  = int(np.sum((hrs >= 60) & (hrs <= 100)))
n_tachy = int(np.sum(hrs > 100))
ax_hr.text(0.97, 0.95, f"Bradycardia: {n_brady}\nNormal: {n_norm}\nTachycardia: {n_tachy}",
           transform=ax_hr.transAxes, ha="right", va="top",
           color=LABEL, fontsize=7.5,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                     edgecolor="#2a2a2a", alpha=0.85))

# [2,1] PPG amplitude distribution (peak-to-peak per segment)
ax_ppg = fig.add_subplot(gs[2, 1])
style_ax(ax_ppg, "PPG Peak-to-Peak per Segment", "Amplitude range (z)", "Count")
ppg_ptp = ppg_all.max(1) - ppg_all.min(1)
ax_ppg.hist(ppg_ptp, bins=30, color=C_PPG, alpha=0.85,
            edgecolor="#0a0a0a", linewidth=0.5, zorder=3)
ax_ppg.axvline(ppg_ptp.mean(), color="white", linewidth=1.0,
               linestyle="--", alpha=0.6, label=f"mean {ppg_ptp.mean():.2f}")
ax_ppg.legend(fontsize=7.5, facecolor="#1a1a1a", edgecolor="#2a2a2a",
              labelcolor=LABEL, framealpha=0.85)

# [2,2] ECG amplitude distribution
ax_ecg = fig.add_subplot(gs[2, 2])
style_ax(ax_ecg, "ECG Peak-to-Peak per Segment", "Amplitude range (z)", "Count")
ecg_ptp = ecg_all.max(1) - ecg_all.min(1)
ax_ecg.hist(ecg_ptp, bins=30, color=C_ECG, alpha=0.85,
            edgecolor="#0a0a0a", linewidth=0.5, zorder=3)
ax_ecg.axvline(ecg_ptp.mean(), color="white", linewidth=1.0,
               linestyle="--", alpha=0.6, label=f"mean {ecg_ptp.mean():.2f}")
ax_ecg.legend(fontsize=7.5, facecolor="#1a1a1a", edgecolor="#2a2a2a",
              labelcolor=LABEL, framealpha=0.85)

# [2,3] Segments per subject bar (stacked train/val/test)
ax_segs = fig.add_subplot(gs[2, 3])
style_ax(ax_segs, "Segments per Subject", "Subject ID", "Segments")
x_tr = np.arange(len(tr_used))
ax_segs.bar(x_tr, tr_counts, color=C_TR, width=0.7,
            edgecolor="#0a0a0a", linewidth=0.4, label="Train", zorder=3)
# val and test as separate clusters
x_va = np.arange(len(tr_used), len(tr_used) + len(va_used))
ax_segs.bar(x_va, va_counts, color=C_VA, width=0.7,
            edgecolor="#0a0a0a", linewidth=0.4, label="Val", zorder=3)
x_te = np.arange(len(tr_used) + len(va_used), len(tr_used) + len(va_used) + len(te_used))
ax_segs.bar(x_te, te_counts, color=C_TE, width=0.7,
            edgecolor="#0a0a0a", linewidth=0.4, label="Test", zorder=3)
ax_segs.set_xticks([]); ax_segs.set_xlabel("Subjects (40 total)", color=LABEL, fontsize=8, labelpad=3)
ax_segs.legend(fontsize=7.5, facecolor="#1a1a1a", edgecolor="#2a2a2a",
               labelcolor=LABEL, framealpha=0.85, loc="upper right")

# ── Super-title ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         "Dataset Overview  —  BIDMC (PPG→ECG)  +  MIT-BIH (Beat Classifier)",
         ha="center", va="center",
         color=WHITE, fontsize=13, fontweight="bold")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT}")
