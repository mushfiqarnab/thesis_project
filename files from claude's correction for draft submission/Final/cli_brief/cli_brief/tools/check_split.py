"""Check whether any participant appears in more than one data split.

Supports either a split column in the CSV (--split-col) or a JSON file mapping
split names to lists of row indices (--split-json). Fails loudly on any other
format rather than guessing.

Usage:
    python check_split.py --csv data.csv --subject-col subject_id --split-col split
    python check_split.py --csv data.csv --subject-col subject_id --split-json split42.json
"""
import argparse
import json
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--subject-col", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split-col")
    g.add_argument("--split-json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.subject_col not in df.columns:
        sys.exit(f"Column {args.subject_col!r} not found. Columns: {list(df.columns)}")

    if args.split_col:
        if args.split_col not in df.columns:
            sys.exit(f"Column {args.split_col!r} not found.")
        splits = {k: v.index.tolist() for k, v in df.groupby(args.split_col)}
    else:
        with open(args.split_json) as f:
            obj = json.load(f)
        if not (isinstance(obj, dict) and all(isinstance(v, list) for v in obj.values())):
            sys.exit("Unsupported split JSON format: expected {split_name: [row indices]}. "
                     "Inspect the file and adapt this script explicitly; do not guess.")
        splits = obj

    subj = {k: set(df.iloc[idx][args.subject_col].astype(str)) for k, idx in splits.items()}
    print("Rows and participants per split:")
    for k, idx in splits.items():
        print(f"  {k}: {len(idx)} rows, participants = {sorted(subj[k])}")
    names = list(subj)
    overlap_found = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = subj[names[i]] & subj[names[j]]
            if inter:
                overlap_found = True
                print(f"OVERLAP {names[i]} & {names[j]}: {sorted(inter)}")
    print("RESULT:", "window-level split (participants shared across splits)"
          if overlap_found else "participant-level split (no shared participants)")


if __name__ == "__main__":
    main()
