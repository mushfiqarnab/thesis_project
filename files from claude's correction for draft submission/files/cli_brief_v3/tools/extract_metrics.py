"""Print every metric block (a dict containing acc and any of dp_abs, eo_max_gap, cf_gap)
found anywhere in one or more benchmark JSON files, with its full key path.

Use this instead of copying numbers by hand. It never rounds silently: it prints the
raw value and a 4-decimal version.

Usage:
    python extract_metrics.py outputs\\reports\\equitas_rcmf_master_benchmark_report.json \\
                              outputs\\reports\\thesis_production_benchmark_report.json
"""
import json
import sys

KEYS = ("acc", "dp_abs", "eo_max_gap", "cf_gap", "count", "latent_diff", "minority_recall")


def walk(obj, path, parent_count=None):
    if isinstance(obj, dict):
        count_here = obj.get("count", parent_count)
        if "acc" in obj and any(k in obj for k in ("dp_abs", "eo_max_gap", "cf_gap")):
            yield path, obj, count_here
        for k, v in obj.items():
            yield from walk(v, path + [str(k)], count_here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + [f"[{i}]"])


def fmt(v):
    return f"{v:.4f} (raw {v!r})" if isinstance(v, float) else repr(v)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for fn in sys.argv[1:]:
        with open(fn, encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n=== {fn} ===")
        top = {k: v for k, v in data.items() if not isinstance(v, (dict, list))} if isinstance(data, dict) else {}
        for k, v in top.items():
            print(f"  [top-level] {k} = {v!r}")
        n = 0
        for path, block, cnt in walk(data, []):
            n += 1
            print("  " + " > ".join(path) + (f"   [count = {cnt}]" if cnt is not None else ""))
            for k in KEYS:
                if k in block:
                    print(f"      {k:<16}{fmt(block[k])}")
        print(f"  ({n} metric blocks)")


if __name__ == "__main__":
    main()
