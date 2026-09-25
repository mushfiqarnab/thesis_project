"""
reanalyze_o7.py
Re-runs the alpha=1.0 leakage test excluding s4 T2 and s4 T3,
to check whether the earlier p=0.0437 result depended on those
two anomalous clips.

Writes the full result to outputs/leakage_run/o7_reanalysis.txt, and also
echoes it to stdout. Must be run from the repository root: the vendored
pipeline reads processed/ and data/UBFC-Phys/ as relative paths.
"""
import sys
import importlib.util
import itertools
from pathlib import Path
import numpy as np
import scipy.stats as stats

# --- Load the pipeline functions from the vendored (in-repo) script ---
PIPELINE_PATH = Path(__file__).resolve().parent / "scripts" / "ubfc_leakage" / "run_full_pos_pipeline.py"

spec = importlib.util.spec_from_file_location("pos_pipeline", PIPELINE_PATH)
pos_pipeline = importlib.util.module_from_spec(spec)
sys.modules["pos_pipeline"] = pos_pipeline
spec.loader.exec_module(pos_pipeline)

# --- Output capture ---------------------------------------------------------
OUT_PATH = Path(__file__).resolve().parent / "outputs" / "leakage_run" / "o7_reanalysis.txt"
_out_lines = []

def emit(line=""):
    """Print to stdout and buffer the same line for the output file."""
    print(line)
    _out_lines.append(line)

fs_rppg = 10
fs_bvp = 64
fs_target = 30
max_lag_samples = int(0.5 * fs_target)

# IMPORTANT: must be run AFTER alpha=1.0 crops have been regenerated
# by Step 2 (scripts/ubfc_leakage/preprocess_video_mediapipe.py with
# alpha=1.0). The processed/ directory must contain alpha=1.0 crops,
# not alpha=0.5. Verify manifest timestamps before running.

dev_clips_data = []
for s in ["s1", "s2", "s3", "s4"]:
    for t in ["1", "2", "3"]:
        rgb = pos_pipeline.extract_rgb_from_pngs(s, t)
        bvp = pos_pipeline.load_bvp(s, t)
        if rgb is not None and bvp is not None:
            rppg = pos_pipeline.compute_pos_rppg(rgb, fs_rppg)
            dev_clips_data.append({"subject": s, "task": f"T{t}", "rppg": rppg, "bvp": bvp})

for clip in dev_clips_data:
    duration_rppg = len(clip["rppg"]) / fs_rppg
    duration_bvp = len(clip["bvp"]) / fs_bvp
    clip["rppg_30hz"] = pos_pipeline.resample_to_common(clip["rppg"], fs_rppg, fs_target, duration_rppg)
    clip["bvp_30hz"] = pos_pipeline.resample_to_common(clip["bvp"], fs_bvp, fs_target, duration_rppg)
    clip["rppg_filt"] = pos_pipeline.bandpass_filter(clip["rppg_30hz"], fs_target)
    clip["bvp_filt"] = pos_pipeline.bandpass_filter(clip["bvp_30hz"], fs_target)

# --- Full null distribution (all 132 cross pairs, unchanged) ---
null_corrs = []
for c1, c2 in itertools.permutations(dev_clips_data, 2):
    if c1["subject"] == c2["subject"]:
        continue
    r_null = pos_pipeline.max_cross_corr(c1["rppg_filt"], c2["bvp_filt"], max_lag_samples)
    null_corrs.append(r_null)

# --- True matches: ALL 12 (original) ---
true_corrs_all = []
for clip in dev_clips_data:
    r = pos_pipeline.max_cross_corr(clip["rppg_filt"], clip["bvp_filt"], max_lag_samples)
    true_corrs_all.append((clip["subject"], clip["task"], r))

# --- True matches: excluding s4 T2 and s4 T3 (N=10) ---
true_corrs_excl = [r for (s, t, r) in true_corrs_all if not (s == "s4" and t in ("T2", "T3"))]
true_corrs_full = [r for (s, t, r) in true_corrs_all]

emit()
emit("=== Per-clip true-match r values ===")
for s, t, r in true_corrs_all:
    flag = "  <-- EXCLUDED IN RETEST" if (s == "s4" and t in ("T2", "T3")) else ""
    emit(f"  {s} {t}: r = {r:.4f}{flag}")

u_all, p_all = stats.mannwhitneyu(true_corrs_full, null_corrs, alternative="greater")
u_excl, p_excl = stats.mannwhitneyu(true_corrs_excl, null_corrs, alternative="greater")

emit()
emit("=== RESULTS ===")
emit(f"ORIGINAL (N=12): U={u_all:.1f}, p={p_all:.4f}")
emit(f"EXCLUDING s4 T2/T3 (N=10): U={u_excl:.1f}, p={p_excl:.4f}")
emit(f"Did removing s4 T2/T3 flip the result above p=0.05? "
     f"{'YES' if (p_all <= 0.05 < p_excl) else 'NO'}")

# --- Persist to disk --------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text("\n".join(_out_lines) + "\n", encoding="utf-8")
n_bytes = OUT_PATH.stat().st_size
print(f"\n[o7] Wrote {OUT_PATH} ({n_bytes} bytes)")
