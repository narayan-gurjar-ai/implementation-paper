# Data

This project uses **`sklearn.datasets.load_digits`** — the Optical
Recognition of Handwritten Digits dataset (E. Alpaydin, C. Kaynak, UCI
Machine Learning Repository), bundled directly with scikit-learn
(`scikit-learn>=1.0`). It is loaded entirely offline via:

```python
from sklearn.datasets import load_digits
digits = load_digits()
```

- 1,797 samples, 8×8 grayscale images (64 features, pixel range 0–16), 10
  classes (digits 0–9).
- No download, API key, or network access required.
- No additional license restrictions beyond scikit-learn's own (BSD-3-
  Clause); the underlying dataset is public-domain UCI data, standard for
  research and teaching use.

No raw data files are stored in this repository — `src/data.py` loads the
dataset fresh from the installed `scikit-learn` package on every run, so
results are exactly reproducible from a clean `pip install -r
requirements.txt`.

See `configs/experiment_config.yaml` for the exact, frozen split and
corruption parameters applied to this dataset in all reported experiments.
