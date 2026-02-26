from pathlib import Path
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", type=str, default="data/csv/multimodal.csv")
    p.add_argument("--out_csv", type=str, default="data/csv/multimodal_10k.csv")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)

    required = ["image_path", "hrv", "gsr", "scar", "threat"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # numeric hygiene
    for c in ["hrv", "gsr", "scar", "threat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["scar"] = df["scar"].astype(int).clip(0, 1)
    df["threat"] = df["threat"].astype(int).clip(0, 1)

    # Preserve original row ids for proper set subtraction
    df = df.reset_index(drop=False).rename(columns={"index": "__rid__"})

    per = args.n // 4
    parts = []

    for s in (0, 1):
        for t in (0, 1):
            g = df[(df["scar"] == s) & (df["threat"] == t)]
            if len(g) == 0:
                continue
            parts.append(g.sample(n=min(per, len(g)), random_state=args.seed))

    out = pd.concat(parts, axis=0) if parts else df.head(0)

    # Top-up if any quadrant was small
    if len(out) < args.n:
        used = set(out["__rid__"].tolist())
        remaining = df[~df["__rid__"].isin(used)]
        need = args.n - len(out)
        if len(remaining) > 0:
            out = pd.concat(
                [out, remaining.sample(n=min(need, len(remaining)), random_state=args.seed)],
                axis=0,
            )

    # Final shuffle + exact size
    out = out.sample(frac=1.0, random_state=args.seed).head(args.n).reset_index(drop=True)

    # Drop internal id column
    out = out.drop(columns=["__rid__"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved {len(out)} rows -> {args.out_csv}")

    # Optional quick sanity print
    ct = out.groupby(["scar", "threat"]).size()
    print("Counts by (scar, threat):")
    print(ct.to_string())

if __name__ == "__main__":
    main()
