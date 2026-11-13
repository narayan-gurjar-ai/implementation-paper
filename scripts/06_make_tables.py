"""Generate LaTeX tables + a per-seed consistency check directly from result CSVs."""
import pandas as pd, numpy as np
R = "results/csv/"
OUT = "paper/tables/"
df = pd.read_csv(R + "magnitude_sweep_results.csv")
assert (df.split == "test").all()
sig = sorted(df.sigma.unique())

def cell(sub, col):
    return f"{sub[col].mean():.3f}$\\pm${sub[col].std():.3f}"

for metric, name, cap in [("nfr", "tab_nfr.tex", "Negative flip rate (lower is better) on the held-out test set, mean$\\pm$std over 5 seeds."),
                          ("accuracy", "tab_acc.tex", "Test accuracy (higher is better), mean$\\pm$std over 5 seeds.")]:
    lines = ["\\begin{tabular}{lcccc}", "\\toprule", "$\\sigma$ & OLD (frozen) & NAIVE & PCT ($\\lambda{=}0.25$) & BCWI ($\\alpha^*$) \\\\", "\\midrule"]
    for s in sig:
        row = [f"{s:g}"]
        for m in ["OLD_frozen", "NAIVE", "PCT", "BCWI"]:
            row.append(cell(df[(df.label == m) & (df.sigma == s)], metric))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(OUT + name, "w").write("\n".join(lines))

# Per-seed paired consistency: in how many of 5 seeds is method X's NFR below NAIVE's?
lines = ["\\begin{tabular}{lcc}", "\\toprule", "$\\sigma$ & BCWI $<$ NAIVE (seeds) & PCT $<$ NAIVE (seeds) \\\\", "\\midrule"]
chk = []
for s in sig:
    p = df[df.sigma == s].pivot(index="seed", columns="label", values="nfr")
    b = int((p.BCWI < p.NAIVE).sum()); c = int((p.PCT < p.NAIVE).sum())
    chk.append((s, b, c)); lines.append(f"{s:g} & {b}/5 & {c}/5 \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(OUT + "tab_seedcheck.tex", "w").write("\n".join(lines))
print("seed check (sigma, BCWI<NAIVE, PCT<NAIVE) [ties count as not-less]:", chk)

# Ablation table (validation split)
ab = pd.read_csv(R + "ablation_results.csv"); assert (ab.split == "val").all()
lines = ["\\begin{tabular}{llccc}", "\\toprule", "Method & Setting & Val.\\ acc. & Val.\\ NFR & NFR change vs.\\ NAIVE \\\\", "\\midrule"]
n_nfr = ab[ab.label == "NAIVE"].nfr.mean()
for lab, disp in [("PCT_lambda=0.25", "PCT $\\lambda{=}0.25$"), ("PCT_lambda=1.0", "PCT $\\lambda{=}1$"), ("PCT_lambda=4.0", "PCT $\\lambda{=}4$"),
                  ("BCWI_alpha=0.8", "BCWI $\\alpha{=}0.8$"), ("BCWI_alpha=0.9", "BCWI $\\alpha{=}0.9$"), ("BCWI_alpha=0.95", "BCWI $\\alpha{=}0.95$"), ("BCWI_alpha=0.98", "BCWI $\\alpha{=}0.98$")]:
    sub = ab[ab.label == lab]
    lines.append(f"{disp.split(' ')[0]} & {disp.split(' ',1)[1]} & {sub.accuracy.mean():.3f} & {sub.nfr.mean():.3f} & {100*(sub.nfr.mean()/n_nfr-1):+.1f}\\% \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(OUT + "tab_ablation.tex", "w").write("\n".join(lines))
print("NAIVE val NFR", round(n_nfr, 4), "OLD val acc", round(ab[ab.label=="NAIVE"].acc_old.mean(), 4))
