"""Model selection + training for both problems.

Usage:  python code/train.py            (from the repository root)

For each problem it
  1. sweeps every polynomial degree (1..max_degree) with ridge regression,
     5-fold CV, and a log-grid of penalties -> the bias/variance curve;
  2. refines the best region with Lasso + a degree-weighted penalty
     (gamma), which exploits sparsity in the true polynomial;
  3. picks the configuration with the lowest *shift-weighted* CV MSE
     (validation rows re-weighted to match the test set's mix of clipped
     coordinates), refits it on all training data and saves it.

Outputs: models/<var>.pkl, results/<var>_cv.json, results/*.png
"""
import json, os, pickle, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polyreg import (PolyModel, cv_lasso, cv_ridge, kfold, poly_features,
                     shift_weights, boundary_count)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ROLL = "BT2024154"
SEEDS = [0, 1]                 # CV repeated with two different fold splits
K = 5

CONFIG = {
    # max_degree given in the assignment; lasso_grid = (degree, gamma) pairs
    "var1": dict(max_degree=10,
                 lasso_grid=[(d, g) for d in (3, 4, 5, 6) for g in (0, 2, 4, 6)]),
    "var2": dict(max_degree=20,
                 lasso_grid=[(d, g) for d in (6, 7, 8, 9, 10, 12) for g in (0, 2, 4)]),
}
RIDGE_ALPHAS = np.logspace(-8, 4, 37)
LASSO_LAMBDAS = np.logspace(-0.5, -4, 22)


def load(var):
    tr = pd.read_csv(os.path.join(DATA, f"{ROLL}_train_{var}.csv"))
    te = pd.read_csv(os.path.join(DATA, f"{ROLL}_test_{var}.csv"))
    return tr.drop(columns="y").values, tr["y"].values, te.values


def repeated(fn):
    """Average a CV function's (fold x grid) output over all seeds/folds."""
    return np.mean([fn(kfold(N, K, s)).mean(0) for s in SEEDS], axis=0)


def run(var):
    global N
    cfg = CONFIG[var]
    X, y, Xte = load(var)
    N = len(y)
    w = shift_weights(X, Xte)
    vy = y.var()
    log = {"n_train": N, "var_y": vy, "ridge": [], "lasso": []}
    print(f"\n=== {var}: {X.shape[1]} features, {N} rows, var(y)={vy:.3f}")

    # ---- stage 1: ridge degree sweep
    for d in range(1, cfg["max_degree"] + 1):
        t = time.time()
        m = repeated(lambda f: cv_ridge(X, y, d, RIDGE_ALPHAS, f))
        mw = repeated(lambda f: cv_ridge(X, y, d, RIDGE_ALPHAS, f, weights=w))
        # ordinary least squares ~ smallest alpha; training MSE for the curve
        P = PolyModel(d, "ridge", RIDGE_ALPHAS[0]).fit(X, y)
        tr_mse = float(np.mean((P.predict(X) - y) ** 2))
        i = int(mw.argmin())
        rec = dict(degree=d, n_terms=poly_features(X[:1], d).shape[1],
                   train_mse_ols=tr_mse, cv_mse_ols=float(m[0]),
                   cv_mse=float(m.min()), cv_wmse=float(mw[i]), alpha=float(RIDGE_ALPHAS[i]))
        log["ridge"].append(rec)
        print(f" ridge d={d:2d} terms={rec['n_terms']:5d}  cvMSE={rec['cv_mse']:.4f}  "
              f"wMSE={rec['cv_wmse']:.4f}  alpha={rec['alpha']:.1e}  ({time.time()-t:.1f}s)")

    # ---- stage 2: lasso + degree-weighted penalty
    for d, g in cfg["lasso_grid"]:
        t = time.time()
        m = repeated(lambda f: cv_lasso(X, y, d, g, LASSO_LAMBDAS, f))
        mw = repeated(lambda f: cv_lasso(X, y, d, g, LASSO_LAMBDAS, f, weights=w))
        i = int(mw.argmin())
        rec = dict(degree=d, gamma=g, cv_mse=float(m[i]), cv_wmse=float(mw[i]),
                   lam=float(LASSO_LAMBDAS[i]))
        log["lasso"].append(rec)
        print(f" lasso d={d:2d} gamma={g}  cvMSE={rec['cv_mse']:.4f}  wMSE={rec['cv_wmse']:.4f}"
              f"  lam={rec['lam']:.1e}  ({time.time()-t:.1f}s)")

    # ---- selection on shift-weighted CV MSE
    cands = [dict(method="ridge", degree=r["degree"], lam=r["alpha"], gamma=0,
                  cv_mse=r["cv_mse"], cv_wmse=r["cv_wmse"]) for r in log["ridge"]]
    cands += [dict(method="lasso", degree=r["degree"], lam=r["lam"], gamma=r["gamma"],
                   cv_mse=r["cv_mse"], cv_wmse=r["cv_wmse"]) for r in log["lasso"]]
    best = min(cands, key=lambda c: c["cv_wmse"])

    # out-of-fold diagnostics for the chosen model
    oof = np.zeros(N)
    for tr, va in kfold(N, K, 0):
        mdl = PolyModel(best["degree"], best["method"], best["lam"], best["gamma"]).fit(X[tr], y[tr])
        oof[va] = mdl.predict(X[va])
    k = boundary_count(X)
    best["oof_mse"] = float(np.mean((oof - y) ** 2))
    best["oof_r2"] = float(1 - best["oof_mse"] / vy)
    best["oof_mse_by_clipped"] = {int(c): float(np.mean((oof[k == c] - y[k == c]) ** 2))
                                  for c in np.unique(k)}

    model = PolyModel(best["degree"], best["method"], best["lam"], best["gamma"]).fit(X, y)
    best["train_mse"] = float(np.mean((model.predict(X) - y) ** 2))
    best["n_nonzero_terms"] = model.n_nonzero()
    best["n_terms"] = len(model.coef_)
    log["best"] = best
    print(f" >>> chosen: {best}")

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "models", f"{var}.pkl"), "wb") as f:
        pickle.dump(model, f)
    np.save(os.path.join(ROOT, "results", f"{var}_oof.npy"), np.c_[y, oof])
    with open(os.path.join(ROOT, "results", f"{var}_cv.json"), "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["var1", "var2"]:
        run(v)
