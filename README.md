# Polynomial Regression Assignment — BT2024154

Polynomial regression models for two problems:

| Problem | Inputs | Allowed degree | Task |
|---|---|---|---|
| var1 — steam turbine Net Power Score | x1…x6 | ≤ 10 | predict `y` |
| var2 — thermal anomaly score | x1, x2, x3 (3-D position) | ≤ 20 | predict `y` |

## Repository layout

```
data/                     train/test CSVs + sample_submission.csv
code/polyreg.py           polynomial features (Legendre/monomial), ridge path, lasso / elastic-net CV, final model class
code/train.py             model selection (degree, penalty) with cross-validation; saves models + CV logs
code/predict.py           inference: writes predictions/<ROLLNO>_pred_var{1,2}.csv
code/make_figures.py      figures used in the report
models/                   trained models (pickle)
results/                  CV logs (json), out-of-fold predictions, figures
predictions/              final prediction files
report/                   PDF report
```

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python code/train.py            # ~5-10 min on a laptop CPU
python code/predict.py          # writes predictions/
python code/make_figures.py
```

All randomness (CV fold splits) is seeded, so results are deterministic.

## Method in one paragraph

Every model is a polynomial in the inputs containing all terms of total degree ≤ d.
Terms are represented in a product-Legendre basis, which spans exactly the same
polynomials as raw monomials but is far better conditioned on [-1, 1]. The degree and
the regularisation strength are chosen by repeated 5-fold cross-validation. Ridge (L2)
gives the full degree-vs-error curve, and Lasso (L1) with a penalty that grows with
term order refines the best region. Elastic net (an L1 + L2 mix) is also tried around
the Lasso optimum and adopted only if it beats the ridge/Lasso winner by more than one
standard error across CV folds. The test set for var1 has many more inputs clipped at
±1 than the training set, so validation errors are importance-weighted to match the
test set's clipping profile before the final model is picked. See the report for details.
