"""Physiology-only baseline on the Line B benchmark split.

Pre-declared protocol (do not change after seeing results):
  * features: the physiological columns only (default: hrv, gsr)
  * standardization fitted on the TRAIN split only
  * two fixed models, no hyperparameter search:
      - logistic regression (C = 1.0)
      - MLP, one hidden layer of 32 units, max_iter = 1000, random_state = 0
  * both are fitted on train; validation accuracy is reported for information;
    the test split is evaluated exactly once per model
  * reports overall test accuracy and accuracy per test participant
  * by default, every row is used, so the test denominator (496) matches the
    EQUITAS-RCMF results; --dedup additionally removes repeated physiology windows
    (a window may be paired with several faces) and should be reported as a
    secondary check only

Usage:
    python physio_baseline.py --csv data\\publishable_scar_production\\multimodal_publishable.csv
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", nargs="+", default=["hrv", "gsr"])
    ap.add_argument("--label", default="threat")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--subject-col", default="physiology_subject")
    ap.add_argument("--window-col", default="window_id")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = set(args.features) | {args.label, args.split_col, args.subject_col}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"Missing columns: {sorted(miss)}. Available: {list(df.columns)}")
    n_raw = len(df)
    if args.dedup and args.window_col in df.columns:
        df = df.drop_duplicates(subset=[args.subject_col, args.window_col, args.split_col])
    print(f"Rows: {n_raw} in file, {len(df)} used" + (" (de-duplicated windows)" if args.dedup else ""))

    parts = {k: df[df[args.split_col] == k] for k in ("train", "val", "test")}
    for k, d in parts.items():
        if d.empty:
            sys.exit(f"Split {k!r} is empty; check --split-col values: {df[args.split_col].unique()}")
        print(f"  {k}: {len(d)} rows, participants {sorted(d[args.subject_col].astype(str).unique())}, "
              f"stress rate {d[args.label].mean():.3f}")

    scaler = StandardScaler().fit(parts["train"][args.features].to_numpy())
    X = {k: scaler.transform(d[args.features].to_numpy()) for k, d in parts.items()}
    y = {k: d[args.label].to_numpy().astype(int) for k, d in parts.items()}

    models = {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000),
        "mlp_32": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=0),
    }
    results = {}
    for name, m in models.items():
        m.fit(X["train"], y["train"])
        val_acc = float((m.predict(X["val"]) == y["val"]).mean())
        pred = m.predict(X["test"])
        test_acc = float((pred == y["test"]).mean())
        per = {}
        t = parts["test"].assign(_pred=pred)
        for s, g in t.groupby(args.subject_col):
            per[str(s)] = {"n": int(len(g)), "acc": float((g["_pred"] == g[args.label]).mean())}
        results[name] = {"val_acc": val_acc, "test_acc": test_acc, "per_participant_test": per}
        print(f"\n{name}: val acc = {val_acc:.4f} | TEST acc = {test_acc:.4f}")
        for s, v in per.items():
            print(f"    {s}: n = {v['n']}, acc = {v['acc']:.4f}")
    maj = max(y["test"].mean(), 1 - y["test"].mean())
    print(f"\nMajority-class test accuracy: {maj:.4f}")
    results["majority_test_acc"] = float(maj)
    results["protocol"] = "pre-declared; see module docstring"
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
