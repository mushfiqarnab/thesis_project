"""Forensic audit of manifest box geometry (2026-09-23).

Motivated by two questions raised during final review:

1. The `3.566283` scale constant has a documented broken chain of custody
   (preregistration, "Data Loss & Chain of Custody Limitation"). The retained
   manifests contain per-frame bbox width (w) and both eye keypoints, so the
   width/IOD ratio implied by the *current-generation* artifacts can be
   computed - with the caveat that these crops were cut USING the constant
   (base_size = iod * 3.566283), so the ratio recovers it by construction and
   is circular, not an independent validation.

2. The alpha=1.0 pass is the arm whose PNGs were overwritten by pass 2. Its
   preserved manifests can be checked for internal consistency with the
   committed crop rule (side = IOD * 3.566283 * 1.5, so w/IOD should be
   ~3.566283 * 1.5 = 5.3494 per frame, alpha smoothing affects the box
   temporal trajectory, not its size rule).

Run from the repository root:

    python scripts/ubfc_leakage/audit_manifest_geometry.py
"""
from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "leakage_run" / "manifest_geometry_audit.txt"

HISTORICAL_C = 3.566283
MARGIN = 1.5


def per_clip(pattern):
    rows_out = []
    for path in sorted(ROOT.glob(pattern)):
        rel = path.relative_to(ROOT).as_posix().split("/")
        rs, n_ok, n_other = [], 0, 0
        for row in csv.DictReader(open(path, newline="")):
            if row["status"] != "OK":
                n_other += 1
                continue
            n_ok += 1
            iod = math.hypot(float(row["rx"]) - float(row["lx"]),
                             float(row["ry"]) - float(row["ly"]))
            if iod > 0:
                rs.append(float(row["w"]) / iod)
        rows_out.append((rel[1], rel[2], n_ok, n_other, rs))
    return rows_out


def main():
    lines = []
    lines.append("MANIFEST BOX-GEOMETRY AUDIT (2026-09-23)")
    lines.append(f"Committed rule: w = IOD * {HISTORICAL_C} * {MARGIN} "
                 f"-> expected per-frame w/IOD = {HISTORICAL_C * MARGIN:.4f}")
    lines.append("Caveat: alpha=0.5 crops were cut using the constant, so their ratios are circular.")
    for pattern, label in [("processed/s*/T*/manifest_alpha1.csv", "alpha=1.0 pass (preserved manifests; PNGs overwritten)"),
                           ("processed/s*/T*/manifest.csv", "alpha=0.5 pass (current artifacts)")]:
        lines.append("")
        lines.append(f"--- {label} ---")
        lines.append(f"{'clip':<8}{'n_OK':>6}{'non-OK':>8}{'mean w/IOD':>12}{'stdev':>9}{'min':>8}{'max':>8}{'implied c':>11}")
        for sid, task, n_ok, n_other, rs in per_clip(pattern):
            q = st.quantiles(rs, n=4)
            lines.append(f"{sid + ' ' + task:<8}{n_ok:>6}{n_other:>8}{st.mean(rs):>12.4f}"
                         f"{st.stdev(rs):>9.4f}{q[0]:>8.3f}{q[2]:>8.3f}{st.mean(rs) / MARGIN:>11.4f}")
    txt = "\n".join(lines) + "\n"
    OUT.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
