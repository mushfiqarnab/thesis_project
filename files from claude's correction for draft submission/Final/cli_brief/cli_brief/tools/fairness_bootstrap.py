"""Fairness and accuracy metrics with participant-clustered bootstrap CIs,
computed ONLY from real saved model predictions.

Input: a CSV of per-window predictions with columns
    id       unique row identifier (must match across models for --compare)
    subject  participant identifier (the bootstrap resamples participants)
    scar     sensitive attribute (0/1)
    y        true label (0/1)
    p        predicted probability of y=1 for the factual input
    p_cf     (optional) predicted probability of y=1 for the scar-removed input
             (leave empty for rows without a counterfactual)

IMPORTANT: the metric definitions below must match the definitions used in the
thesis code. Compare them with the repository's metric functions before using
the output. If they differ, change this file to match the repository and record
the change; never change the repository's definitions to match this file.

Usage:
    python fairness_bootstrap.py --preds preds_D.csv --B 10000 --seed 0
    python fairness_bootstrap.py --preds preds_D.csv --compare preds_A.csv --B 10000
"""
import argparse
import sys

import numpy as np
import pandas as pd

METRICS = ["accuracy", "dp_gap", "eo_gap", "cf_gap", "worst_group_acc"]


def metrics(df, thr):
    yhat = (df["p"].to_numpy() >= thr).astype(int)
    y = df["y"].to_numpy().astype(int)
    s = df["scar"].to_numpy().astype(int)
    out = {"accuracy": float(np.mean(yhat == y))}

    def rate(mask):
        return float(np.mean(yhat[mask])) if mask.any() else np.nan

    out["dp_gap"] = abs(rate(s == 1) - rate(s == 0))
    tpr_gap = abs(rate((s == 1) & (y == 1)) - rate((s == 0) & (y == 1)))
    fpr_gap = abs(rate((s == 1) & (y == 0)) - rate((s == 0) & (y == 0)))
    out["eo_gap"] = float(np.nanmax([tpr_gap, fpr_gap])) if not (
        np.isnan(tpr_gap) and np.isnan(fpr_gap)) else np.nan

    if "p_cf" in df.columns and df["p_cf"].notna().any():
        m = df["p_cf"].notna().to_numpy()
        out["cf_gap"] = float(np.mean(np.abs(df["p"].to_numpy()[m] - df["p_cf"].to_numpy()[m])))
    else:
        out["cf_gap"] = np.nan

    accs = []
    for sv in (0, 1):
        for yv in (0, 1):
            m = (s == sv) & (y == yv)
            if m.any():
                accs.append(np.mean(yhat[m] == y[m]))
    out["worst_group_acc"] = float(min(accs)) if len(accs) == 4 else np.nan
    return out


def check(df, name):
    need = {"id", "subject", "scar", "y", "p"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"{name}: missing columns {sorted(miss)}")
    if df["id"].duplicated().any():
        sys.exit(f"{name}: duplicate ids")
    for c in ("scar", "y"):
        if not set(df[c].unique()) <= {0, 1}:
            sys.exit(f"{name}: column {c} must be 0/1")
    if not df["p"].between(0, 1).all():
        sys.exit(f"{name}: p must be probabilities in [0, 1]")


def cluster_resample(subjects, rng):
    return rng.choice(subjects, size=len(subjects), replace=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--compare", help="second model's predictions; reports preds minus compare")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    a = pd.read_csv(args.preds)
    check(a, args.preds)
    b = None
    if args.compare:
        b = pd.read_csv(args.compare)
        check(b, args.compare)
        if set(a["id"]) != set(b["id"]):
            sys.exit("The two prediction files do not contain the same ids; a paired comparison is invalid.")
        b = b.set_index("id").loc[a["id"]].reset_index()

    subjects = np.array(sorted(a["subject"].astype(str).unique()))
    if len(subjects) < 10:
        print(f"WARNING: only {len(subjects)} participants. Cluster-bootstrap intervals with so few "
              f"clusters are unstable and tend to be too narrow; report them as indicative only.")

    groups_a = {s: g for s, g in a.groupby(a["subject"].astype(str))}
    groups_b = {s: g for s, g in b.groupby(b["subject"].astype(str))} if b is not None else None

    point_a = metrics(a, args.thr)
    point = ({k: point_a[k] - metrics(b, args.thr)[k] for k in METRICS} if b is not None else point_a)

    rng = np.random.default_rng(args.seed)
    boot = {k: [] for k in METRICS}
    for _ in range(args.B):
        draw = cluster_resample(subjects, rng)
        ra = pd.concat([groups_a[s] for s in draw])
        ma = metrics(ra, args.thr)
        if b is not None:
            rb = pd.concat([groups_b[s] for s in draw])
            mb = metrics(rb, args.thr)
            ma = {k: ma[k] - mb[k] for k in METRICS}
        for k in METRICS:
            boot[k].append(ma[k])

    label = "difference (preds - compare)" if b is not None else "value"
    print(f"Participants: {len(subjects)} | rows: {len(a)} | B = {args.B} | seed = {args.seed} | threshold = {args.thr}")
    print(f"{'metric':<18}{label:>30}{'95% CI (percentile)':>28}{'valid draws':>13}")
    for k in METRICS:
        arr = np.array(boot[k], dtype=float)
        ok = arr[~np.isnan(arr)]
        if np.isnan(point[k]) or len(ok) == 0:
            print(f"{k:<18}{'n/a':>30}")
            continue
        lo, hi = np.percentile(ok, [2.5, 97.5])
        print(f"{k:<18}{point[k]:>30.4f}{f'[{lo:.4f}, {hi:.4f}]':>28}{len(ok):>13}")
    print("Note: draws where a scar/label group was absent are excluded for that metric (see 'valid draws').")


if __name__ == "__main__":
    main()
