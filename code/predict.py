"""Inference: load the trained models and write the prediction files.

Usage:  python code/predict.py          (run code/train.py first)

Writes predictions/<ROLLNO>_pred_var1.csv and <ROLLNO>_pred_var2.csv in the
sample-submission format (a single column `y`, one row per test row, same order).
"""
import os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polyreg  # noqa: F401  (needed to unpickle PolyModel)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLL = "BT2024154"


def main():
    sample = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))
    out_dir = os.path.join(ROOT, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    for var in ("var1", "var2"):
        Xte = pd.read_csv(os.path.join(ROOT, "data", f"{ROLL}_test_{var}.csv")).values
        with open(os.path.join(ROOT, "models", f"{var}.pkl"), "rb") as f:
            model = pickle.load(f)
        pred = model.predict(Xte)
        assert len(pred) == len(sample) and np.isfinite(pred).all()
        sub = pd.DataFrame({c: pred for c in sample.columns})
        path = os.path.join(out_dir, f"{ROLL}_pred_{var}.csv")
        sub.to_csv(path, index=False)
        print(f"{var}: degree={model.degree} method={model.method} -> {path} "
              f"(mean={pred.mean():.3f}, std={pred.std():.3f})")


if __name__ == "__main__":
    main()
