#!/usr/bin/env bash
# Regenerates every result, figure, table and the paper PDF (CPU only, a few minutes).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs results/csv results/figures paper/figures paper/tables
for s in 00_env_check 00b_pilot_design 01_train_baseline 02_ablation 03_magnitude_sweep 04_failure_analysis 05_make_figures 06_make_tables; do
  echo ">> scripts/$s.py"; python3 scripts/$s.py > results/logs/$s.log 2>&1
done
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "done: paper/main.pdf"
