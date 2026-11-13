# Verified excerpts from Paper 1 used to motivate this study

Source: `main.tex` / compiled `Tracking_LLM_Performance_Drift.pdf`
(previous submission, re-read on 2026-10-20 to ground this project).

> "Bansal et al. [bansal2019updates] were among the first to formalize this
> problem in human-AI interaction terms... They term instances that were
> previously classified correctly but are misclassified after an update
> 'negative flips'... the negative flip rate is
> NFR(f_old, f_new, D) = |{x in D : f_old(x)=y(x) ∧ f_new(x)≠y(x)}| / |D|."
> — Section 2.1 / Equation 2

> "Yan et al. [yan2021positive] extended this idea to image classification
> with positive-congruent training, an objective that explicitly constrains
> a new model to preserve the correct predictions of its predecessor."
> — Section 2.1

> "Schumann et al. [schumann2024backward] proposed backward-compatible
> weight interpolation (BCWI), which blends the weights of an old and a new
> model to reduce negative flips without sacrificing the new model's
> aggregate accuracy gains."
> — Section 2.1

> "...the absence of a shared, standardized drift-reporting metric... makes
> it difficult to compare drift magnitude across studies, providers, or
> model families..." — Section 6 (Methodological Challenges, §5.4 in the
> numbered subsections)

These three verified claims are the direct ancestry of this project's
research question: NFR is the metric (first excerpt), PCT and BCWI are the
two mitigation strategies under comparison (second and third excerpts), and
the absence of a shared comparison protocol across mitigation strategies is
the gap this study addresses (fourth excerpt).

Citation keys below match `paper/references.bib` in this project, carried
over unchanged from Paper 1's bibliography (re-verified, not re-derived):

- bansal2019updates — Bansal, G. et al. (2019). AAAI. doi:10.1609/aaai.v33i01.33012429
- yan2021positive — Yan, S. et al. (2021). CVPR. doi:10.1109/CVPR46437.2021.01408
- schumann2024backward — Schumann, R. et al. (2024). EACL. doi:10.18653/v1/2024.eacl-long.174
- gama2014survey — Gama, J. et al. (2014). ACM Computing Surveys. doi:10.1145/2523813
- chen2024chatgpt — Chen, L., Zaharia, M., Zou, J. (2024). Harvard Data Science Review.
- echterhoff2024muscle — Echterhoff, J. et al. (2024). Findings of EMNLP.
- kirkpatrick2017overcoming — Kirkpatrick, J. et al. (2017). PNAS. doi:10.1073/pnas.1611835114
