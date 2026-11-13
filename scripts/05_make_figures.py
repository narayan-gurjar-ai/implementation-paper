"""
Generate all figures from the saved result CSVs (never from re-derived or
hand-typed numbers). Saves to both results/figures/ (working copy) and
paper/figures/ (camera-ready copies used by the LaTeX source).
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIRS = ["results/figures", "paper/figures"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {"OLD_frozen": "#888888", "NAIVE": "#d62728", "PCT": "#ff7f0e", "BCWI": "#1f77b4"}
MARKERS = {"OLD_frozen": "s", "NAIVE": "o", "PCT": "^", "BCWI": "D"}


def savefig(fig, name):
    for d in OUT_DIRS:
        fig.savefig(f"{d}/{name}", bbox_inches="tight")
    print(f"saved {name}")


# ---------------------------------------------------------------
# Figure 1: ablation trade-off curves (accuracy vs NFR), sigma=8
# ---------------------------------------------------------------
abl = pd.read_csv("results/csv/ablation_results.csv")

pct_rows = abl[abl.label.str.startswith("PCT_lambda=")].copy()
pct_rows["lambda"] = pct_rows.label.str.replace("PCT_lambda=", "").astype(float)
pct_summary = pct_rows.groupby("lambda").agg(
    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    nfr_mean=("nfr", "mean"), nfr_std=("nfr", "std"),
).reset_index().sort_values("lambda")

bcwi_rows = abl[abl.label.str.startswith("BCWI_alpha=")].copy()
bcwi_rows["alpha"] = bcwi_rows.label.str.replace("BCWI_alpha=", "").astype(float)
bcwi_summary = bcwi_rows.groupby("alpha").agg(
    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
    nfr_mean=("nfr", "mean"), nfr_std=("nfr", "std"),
).reset_index().sort_values("alpha")

naive_rows = abl[abl.label == "NAIVE"]
naive_acc_mean, naive_acc_std = naive_rows.accuracy.mean(), naive_rows.accuracy.std()
naive_nfr_mean, naive_nfr_std = naive_rows.nfr.mean(), naive_rows.nfr.std()

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.errorbar(pct_summary.nfr_mean, pct_summary.acc_mean,
            xerr=pct_summary.nfr_std, yerr=pct_summary.acc_std,
            marker="^", color="#ff7f0e", label="PCT (sweep over $\\lambda$)",
            capsize=2, linewidth=1.3, markersize=5)
ax.errorbar(bcwi_summary.nfr_mean, bcwi_summary.acc_mean,
            xerr=bcwi_summary.nfr_std, yerr=bcwi_summary.acc_std,
            marker="D", color="#1f77b4", label="BCWI (sweep over $\\alpha$)",
            capsize=2, linewidth=1.3, markersize=5)
ax.errorbar([naive_nfr_mean], [naive_acc_mean], xerr=[naive_nfr_std], yerr=[naive_acc_std],
            marker="o", color="#d62728", label="NAIVE (reference point)",
            capsize=2, markersize=8, linestyle="none")
# annotate selected alpha*
import csv as _csv
_astar = [float(r["value"]) for r in _csv.DictReader(open("results/csv/selected_hyperparams.csv")) if r["param"]=="alpha_star"][0]
sel = bcwi_summary[np.isclose(bcwi_summary.alpha, _astar)]
if len(sel):
    ax.annotate(rf"$\alpha^*={_astar}$", xy=(sel.nfr_mean.values[0], sel.acc_mean.values[0]),
                xytext=(sel.nfr_mean.values[0] + 0.012, sel.acc_mean.values[0] - 0.012),
                fontsize=9, color="#1f77b4")
ax.set_xlabel("Negative Flip Rate (lower is better)")
ax.set_ylabel("Test accuracy (higher is better)")
ax.set_title(r"Accuracy--NFR trade-off at $\sigma=8$, validation split (5 seeds)")
ax.legend(loc="lower right", fontsize=8)
savefig(fig, "fig1_tradeoff_ablation.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 2: magnitude sensitivity, NFR vs sigma
# ---------------------------------------------------------------
mag = pd.read_csv("results/csv/magnitude_sweep_summary.csv")

fig, ax = plt.subplots(figsize=(5.4, 4.0))
for label in ["OLD_frozen", "NAIVE", "PCT", "BCWI"]:
    sub = mag[mag.label == label].sort_values("sigma")
    ax.plot(sub.sigma, sub.nfr_mean, marker=MARKERS[label], color=COLORS[label], label=label, linewidth=1.5)
    ax.fill_between(sub.sigma, sub.nfr_mean - sub.nfr_std, sub.nfr_mean + sub.nfr_std,
                     color=COLORS[label], alpha=0.15)
ax.set_xlabel(r"Update magnitude $\sigma$ (corruption level of new training data)")
ax.set_ylabel("Negative Flip Rate")
ax.set_title("Negative Flip Rate vs. update magnitude (mean $\\pm$ std, 5 seeds)")
ax.legend(fontsize=8)
savefig(fig, "fig2_nfr_vs_sigma.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 3: magnitude sensitivity, accuracy vs sigma
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 4.0))
for label in ["OLD_frozen", "NAIVE", "PCT", "BCWI"]:
    sub = mag[mag.label == label].sort_values("sigma")
    ax.plot(sub.sigma, sub.acc_mean, marker=MARKERS[label], color=COLORS[label], label=label, linewidth=1.5)
    ax.fill_between(sub.sigma, sub.acc_mean - sub.acc_std, sub.acc_mean + sub.acc_std,
                     color=COLORS[label], alpha=0.15)
ax.set_xlabel(r"Update magnitude $\sigma$ (corruption level of new training data)")
ax.set_ylabel("Test accuracy")
ax.set_title("Accuracy vs. update magnitude (mean $\\pm$ std, 5 seeds)")
ax.legend(fontsize=8)
savefig(fig, "fig3_accuracy_vs_sigma.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 4: per-class negative flip breakdown (failure analysis)
# ---------------------------------------------------------------
byclass = pd.read_csv("results/csv/failure_analysis_by_class.csv")
fig, ax = plt.subplots(figsize=(5.4, 3.6))
x = np.arange(10)
width = 0.35
ax.bar(x - width/2, byclass.naive_flip_count, width, label="NAIVE", color=COLORS["NAIVE"])
ax.bar(x + width/2, byclass.bcwi_flip_count, width, label=f"BCWI ($\\alpha^*={_astar}$)", color=COLORS["BCWI"])
ax.set_xlabel("True digit class")
ax.set_ylabel("Negative flips (summed over 5 seeds, $\\sigma=8$)")
ax.set_title("Negative flips by true class: NAIVE vs. BCWI")
ax.set_xticks(x)
ax.legend(fontsize=8)
savefig(fig, "fig4_failure_by_class.pdf")
plt.close(fig)

print("\nAll figures generated from results/csv/*.csv (no hand-entered values).")
