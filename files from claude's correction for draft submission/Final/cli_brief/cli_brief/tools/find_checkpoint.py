"""Search local drives for the lost master checkpoint by SHA-256.

Read-only: never moves, renames, or deletes anything.

Usage (PowerShell or bash):
    python find_checkpoint.py --roots C:\\ D:\\ --out checkpoint_search.csv
"""
import argparse
import csv
import hashlib
import os
import sys

TARGET = "4c2dcad470271ad7109ec302a2e4d31eac429cc054ba3d243ce0d39266b3bc5b"
EXTS = (".pt", ".pth", ".ckpt", ".bin", ".zip")
SKIP_DIRS = {"$Recycle.Bin", "Windows", "Program Files", "Program Files (x86)",
             "node_modules", ".git", "__pycache__", "AppData\\Local\\Temp"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--min-mb", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoint_search.csv")
    args = ap.parse_args()

    rows, found = [], []
    for root in args.roots:
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if not name.lower().endswith(EXTS):
                    continue
                p = os.path.join(dirpath, name)
                try:
                    size_mb = os.path.getsize(p) / 1e6
                    if size_mb < args.min_mb:
                        continue
                    digest = sha256(p)
                except OSError as e:
                    print(f"[skip] {p}: {e}", file=sys.stderr)
                    continue
                match = digest == TARGET
                rows.append({"path": p, "size_mb": f"{size_mb:.2f}",
                             "sha256": digest, "match": match})
                if match:
                    found.append(p)
                    print(f"[MATCH] {p}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size_mb", "sha256", "match"])
        w.writeheader()
        w.writerows(rows)
    print(f"Scanned {len(rows)} candidate files; matches: {len(found)}")
    print(f"Full listing written to {args.out}")
    sys.exit(0 if found else 2)


if __name__ == "__main__":
    main()
