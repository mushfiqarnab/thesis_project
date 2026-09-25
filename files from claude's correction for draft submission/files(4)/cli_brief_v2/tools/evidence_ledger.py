"""Build an evidence ledger: every number in the thesis .tex files, and where
(if anywhere) it appears in the repository's output files.

This is a HEURISTIC search. A FOUND row means the string appears in some output
file, not that it is the right quantity. A human must confirm each row.

Usage:
    python evidence_ledger.py --tex-dir thesis/chapters thesis/core \
        --search-dirs outputs logs docs --out evidence_ledger.csv
"""
import argparse
import csv
import os
import re

NUM_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(\s*\\?%)?")
SEARCH_EXTS = (".json", ".csv", ".txt", ".log", ".md", ".yaml", ".yml")
# Numbers that are almost never results (years, section numbers, small ints).
IGNORE = {str(i) for i in range(0, 11)}


def variants(num):
    """Formatting variants of a number as it might appear in an output file."""
    out = {num}
    try:
        x = float(num.replace(",", ""))
    except ValueError:
        return out
    for d in range(0, 7):
        out.add(f"{x:.{d}f}")
    if x > 1:  # a percentage may be stored as a fraction
        for d in range(2, 7):
            out.add(f"{x / 100:.{d}f}")
    return out


def load_corpus(dirs):
    corpus = []
    for d in dirs:
        for dp, _, fns in os.walk(d):
            for fn in fns:
                if fn.lower().endswith(SEARCH_EXTS):
                    p = os.path.join(dp, fn)
                    try:
                        with open(p, encoding="utf-8", errors="ignore") as f:
                            corpus.append((p, f.read()))
                    except OSError:
                        pass
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-dir", nargs="+", required=True)
    ap.add_argument("--search-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="evidence_ledger.csv")
    args = ap.parse_args()

    corpus = load_corpus(args.search_dirs)
    rows = []
    for d in args.tex_dir:
        for dp, _, fns in os.walk(d):
            for fn in fns:
                if not fn.endswith(".tex"):
                    continue
                p = os.path.join(dp, fn)
                with open(p, encoding="utf-8", errors="ignore") as f:
                    for ln, line in enumerate(f, 1):
                        if line.lstrip().startswith("%"):
                            continue
                        for m in NUM_RE.finditer(line):
                            num = m.group(1)
                            if num in IGNORE or re.fullmatch(r"(19|20)\d\d", num):
                                continue
                            hits = [cp for cp, text in corpus
                                    if any(v in text for v in variants(num))]
                            rows.append({
                                "tex_file": p, "line": ln, "number": num,
                                "context": line.strip()[:160],
                                "status": "FOUND" if hits else "NOT_FOUND",
                                "hit_files": ";".join(hits[:5]),
                                "human_confirmed": "",
                            })
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["tex_file", "line", "number", "context", "status",
                            "hit_files", "human_confirmed"])
        w.writeheader()
        w.writerows(rows)
    nf = sum(r["status"] == "NOT_FOUND" for r in rows)
    print(f"{len(rows)} numbers checked; {nf} NOT_FOUND. See {args.out}")


if __name__ == "__main__":
    main()
