"""Persist the full per-clip estimator table to outputs/leakage_run/.

The writeup (docs/negative_result_writeup.md, section 4.5) previously recorded
only the two highest CHROM clips. This script re-runs the *unchanged* estimator
comparison harness (src/evaluation/leakage_estimator_comparison.py) and writes
its complete output - the estimator-level summary blocks and the full 12-clip
per-estimator true-correlation table - to outputs/leakage_run/, so the
per-clip values behind the writeup tables are stored evidence, not console
output.

Run from the repository root:

    python scripts/ubfc_leakage/save_per_clip_results.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.leakage_estimator_comparison import build_clips, evaluate

OUT_DIR = ROOT / "outputs" / "leakage_run"
ESTIMATORS = ("POS", "CHROM", "PBV")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = build_clips()
    assert len(clips) == 12, f"expected 12 clips, got {len(clips)}"

    results = []
    for name in ESTIMATORS:
        print(f"Evaluating estimator: {name}", flush=True)
        results.append(evaluate(clips, name))
    by_name = {r["estimator"]: r for r in results}

    csv_path = OUT_DIR / "estimator_per_clip_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "task", "pos_r", "chrom_r", "pbv_r"])
        for i in range(len(clips)):
            subject, task, _ = results[0]["per_clip"][i]
            writer.writerow([
                subject,
                task,
                *(f"{by_name[e]['per_clip'][i][2]:.4f}" for e in ESTIMATORS),
            ])

    txt_path = OUT_DIR / "estimator_comparison_full_output.txt"
    lines = []
    lines.append("LEAKAGE TEST BY ESTIMATOR")
    lines.append("Pre-registered rule: pass iff Mann-Whitney p (True > Null) > 0.05.")
    lines.append("Produced by scripts/ubfc_leakage/save_per_clip_results.py on top of")
    lines.append("src/evaluation/leakage_estimator_comparison.py (code unchanged).")
    lines.append("")
    header = (f"{'Estimator':<10} {'True mean r':>12} {'Null mean r':>12} "
              f"{'Null p95 r':>11} {'U':>9} {'p':>9} {'Verdict':>8} {'pct':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r['estimator']:<10} {r['true_mean']:>12.4f} {r['null_mean']:>12.4f} "
            f"{r['null_p95']:>11.4f} {r['u']:>9.1f} {r['p']:>9.4f} "
            f"{r['verdict']:>8} {r['pct_criterion']:>6}"
        )
    lines.append("")
    lines.append("Per-clip true correlations (max Pearson r over lags within +/-0.5 s):")
    lines.append(f"{'Clip':<8}" + "".join(f"{r['estimator']:>10}" for r in results))
    for i in range(len(clips)):
        subject, task, _ = results[0]["per_clip"][i]
        row = f"{subject + ' ' + task:<8}"
        for r in results:
            row += f"{r['per_clip'][i][2]:>10.4f}"
        lines.append(row)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
