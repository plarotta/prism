"""Generate plots for Phase 6 hybrid experiment results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results") / "hybrid"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================================
# Data from experiment logs
# ==========================================================================

# Experiment 0: Improved baselines (7000 steps)
exp0 = {
    "PRISM-MeanPool 2K": {
        "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ndcg":  [0.2184, 0.3898, 0.5125, 0.6014, 0.7043, 0.7524, 0.7700],
        "loss":  [0.6892, 0.3288, 0.1532, 0.1108, 0.0626, 0.0538, 0.0315],
    },
    "Transformer 2K": {
        "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ndcg":  [0.2870, 0.3659, 0.4575, 0.5417, 0.6061, 0.6524, 0.6716],
        "loss":  [1.0555, 0.6946, 0.4524, 0.2740, 0.1917, 0.1473, 0.1089],
    },
    "PRISM-MeanPool 8K": {
        "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ndcg":  [0.1677, 0.2783, 0.3930, 0.4835, 0.5597, 0.6419, 0.6747],
        "loss":  [0.7540, 0.4065, 0.1962, 0.1313, 0.0788, 0.0715, 0.0465],
    },
}

# Experiment 1: Attentive pooling (7000 steps)
exp1 = {
    "Attentive 2K": {
        "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ndcg":  [0.2531, 0.3942, 0.4564, 0.5842, 0.6775, 0.7306, 0.7526],
    },
    "Attentive 8K": {
        "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ndcg":  [0.1640, 0.2751, 0.4082, 0.5183, 0.6110, 0.6729, 0.6922],
    },
    "Mean 2K": {
        "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ndcg":  [0.2184, 0.3898, 0.5125, 0.6014, 0.7043, 0.7524, 0.7700],
    },
    "Mean 8K": {
        "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "ndcg":  [0.1677, 0.2783, 0.3930, 0.4835, 0.5597, 0.6419, 0.6747],
    },
}

# Experiment 4: Decay spacing ablation (5000 steps)
exp4 = {
    "All-slow (λ=0.99)": {
        "steps": [1000, 2000, 3000, 4000, 5000],
        "ndcg":  [0.3603, 0.4865, 0.6190, 0.7286, 0.7637],
        "loss":  [0.5581, 0.2314, 0.0941, 0.0582, 0.0455],
    },
    "Geometric": {
        "steps": [1000, 2000, 3000, 4000, 5000],
        "ndcg":  [0.2506, 0.3954, 0.5556, 0.6497, 0.6918],
        "loss":  [0.6576, 0.2949, 0.1264, 0.0744, 0.0521],
    },
    "Linear": {
        "steps": [1000, 2000, 3000, 4000, 5000],
        "ndcg":  [0.2552, 0.3923, 0.5365, 0.6190, 0.6539],
        "loss":  [0.6698, 0.3036, 0.1500, 0.0913, 0.0751],
    },
    "Random": {
        "steps": [1000, 2000, 3000, 4000, 5000],
        "ndcg":  [0.2513, 0.3893, 0.5319, 0.6113, 0.6406],
        "loss":  [0.6589, 0.2850, 0.1364, 0.0977, 0.0756],
    },
}


# ==========================================================================
# Plot 1: Experiment 0 — Baseline convergence trajectories
# ==========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

colors_exp0 = {
    "PRISM-MeanPool 2K": "#2563EB",
    "Transformer 2K": "#DC2626",
    "PRISM-MeanPool 8K": "#2563EB",
}
styles_exp0 = {
    "PRISM-MeanPool 2K": "-",
    "Transformer 2K": "-",
    "PRISM-MeanPool 8K": "--",
}

for name, data in exp0.items():
    ax1.plot(data["steps"], data["ndcg"], marker="o", markersize=5,
             color=colors_exp0[name], linestyle=styles_exp0[name], label=name, linewidth=2)
    ax2.plot(data["steps"], data["loss"], marker="o", markersize=5,
             color=colors_exp0[name], linestyle=styles_exp0[name], label=name, linewidth=2)

ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Avg nDCG@10", fontsize=11)
ax1.set_title("Experiment 0: Quality During Training", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.85)

ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Loss", fontsize=11)
ax2.set_title("Experiment 0: Training Loss", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "exp0_baselines.png", bbox_inches="tight", dpi=150)
print(f"Saved: {RESULTS_DIR / 'exp0_baselines.png'}")
plt.close("all")


# ==========================================================================
# Plot 2: Experiment 1 — Attentive vs Mean pooling
# ==========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: convergence trajectories
colors_exp1 = {
    "Attentive 2K": "#2563EB",
    "Attentive 8K": "#2563EB",
    "Mean 2K": "#9CA3AF",
    "Mean 8K": "#9CA3AF",
}
styles_exp1 = {
    "Attentive 2K": "-",
    "Attentive 8K": "--",
    "Mean 2K": "-",
    "Mean 8K": "--",
}

for name, data in exp1.items():
    ax1.plot(data["steps"], data["ndcg"], marker="o", markersize=5,
             color=colors_exp1[name], linestyle=styles_exp1[name], label=name, linewidth=2)

ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Avg nDCG@10", fontsize=11)
ax1.set_title("Experiment 1: Pooling Convergence", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.85)

# Right: final scores bar chart
configs = ["Mean\n2048", "Attentive\n2048", "Mean\n8192", "Attentive\n8192"]
scores = [0.770, 0.753, 0.675, 0.692]
bar_colors = ["#9CA3AF", "#2563EB", "#9CA3AF", "#2563EB"]
hatches = ["", "", "//", "//"]

bars = ax2.bar(range(len(configs)), scores, color=bar_colors, edgecolor="black", linewidth=0.5)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)
for bar, score in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
             f"{score:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(configs, fontsize=10)
ax2.set_ylabel("Avg nDCG@10", fontsize=11)
ax2.set_title("Experiment 1: Final Scores", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 0.9)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "exp1_attentive_pooling.png", bbox_inches="tight", dpi=150)
print(f"Saved: {RESULTS_DIR / 'exp1_attentive_pooling.png'}")
plt.close("all")


# ==========================================================================
# Plot 3: Experiment 4 — Decay spacing ablation
# ==========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

colors_exp4 = {
    "All-slow (λ=0.99)": "#10B981",
    "Geometric": "#2563EB",
    "Linear": "#F59E0B",
    "Random": "#8B5CF6",
}

# Left: convergence trajectories
for name, data in exp4.items():
    ax1.plot(data["steps"], data["ndcg"], marker="o", markersize=6,
             color=colors_exp4[name], label=name, linewidth=2.5)

ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Avg nDCG@10", fontsize=11)
ax1.set_title("Experiment 4: Decay Spacing Convergence", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.85)

# Right: final scores bar chart
modes = list(exp4.keys())
final_scores = [exp4[m]["ndcg"][-1] for m in modes]
bar_colors_4 = [colors_exp4[m] for m in modes]

bars = ax2.bar(range(len(modes)), final_scores, color=bar_colors_4, edgecolor="black", linewidth=0.5)
for bar, score in zip(bars, final_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
             f"{score:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax2.set_xticks(range(len(modes)))
ax2.set_xticklabels([m.replace(" (λ=0.99)", "\n(λ=0.99)") for m in modes], fontsize=10)
ax2.set_ylabel("Avg nDCG@10", fontsize=11)
ax2.set_title("Experiment 4: Final Scores (5000 steps)", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 0.9)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "exp4_decay_spacing.png", bbox_inches="tight", dpi=150)
print(f"Saved: {RESULTS_DIR / 'exp4_decay_spacing.png'}")
plt.close("all")


# ==========================================================================
# Plot 4: Summary — All results comparison
# ==========================================================================

fig, ax = plt.subplots(figsize=(12, 7))

entries = [
    ("All-slow (λ=0.99) @ 2K", 0.764, "#10B981"),
    ("PRISM-MeanPool @ 2K (7K steps)", 0.770, "#2563EB"),
    ("PRISM-Attentive @ 2K", 0.753, "#60A5FA"),
    ("PRISM-Geometric @ 2K", 0.692, "#93C5FD"),
    ("PRISM-Attentive @ 8K", 0.692, "#60A5FA"),
    ("PRISM-MeanPool @ 8K", 0.675, "#2563EB"),
    ("Transformer (tuned) @ 2K", 0.672, "#DC2626"),
    ("PRISM-Linear @ 2K", 0.654, "#93C5FD"),
    ("PRISM-Random @ 2K", 0.641, "#93C5FD"),
    ("M2-BERT-80M (8K)*", 0.506, "#6B7280"),
    ("BM25*", 0.486, "#6B7280"),
]

names, scores, colors = zip(*entries)
y_pos = range(len(names))

bars = ax.barh(y_pos, scores, color=colors, edgecolor="black", linewidth=0.5, height=0.7)
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=10, fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("Avg nDCG@10 (LoCoV1, 12 tasks)", fontsize=11)
ax.set_title("PRISM Hybrid Experiments: All Results", fontsize=14, fontweight="bold")
ax.set_xlim(0, 0.9)
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()

# Reference lines
ax.axvline(0.770, color="#2563EB", linestyle=":", alpha=0.4, linewidth=1)
ax.axvline(0.672, color="#DC2626", linestyle=":", alpha=0.4, linewidth=1)

ax.text(0.02, 0.02, "* Published baselines (approximate)", transform=ax.transAxes,
        fontsize=8, color="#6B7280", style="italic")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "hybrid_all_results.png", bbox_inches="tight", dpi=150)
print(f"Saved: {RESULTS_DIR / 'hybrid_all_results.png'}")
plt.close("all")

print("\nAll plots generated.")
