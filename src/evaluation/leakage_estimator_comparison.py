"""
Estimator-specificity check for the vision-side leakage finding.

Re-runs the pre-registered leakage test (docs/negative_result_writeup.md, sections
3.3 and 3.4) with three rPPG estimators on *identical* inputs:

    POS   - Plane-Orthogonal-to-Skin (Wang et al. 2017), the estimator used for the
            reported result in the writeup
    CHROM - Chrominance-based (de Haan & Jeanne 2013)
    PBV   - Blood-Volume-Pulse signature (de Haan & van Leest 2014)

Only the estimator changes. Crop extraction, temporal resampling, band-pass
filtering, the lag search and the true/null pairing scheme are identical to the
reported run, so any difference in outcome is attributable to the estimator alone.

Purpose: the reported negative result currently rests on POS alone. This script
tests whether the leakage verdict is estimator-specific.

Run from the repository root:

    python src/evaluation/leakage_estimator_comparison.py
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import cv2
import numpy as np
import scipy.signal as signal
import scipy.stats as stats

SUBJECTS = ["s1", "s2", "s3", "s4"]
TASKS = ["1", "2", "3"]

# Identical to the reported run (scratch/run_full_pos_pipeline.py)
FS_RPPG = 10
FS_BVP = 64
FS_TARGET = 30
WINDOW_SEC = 1.6
BANDPASS = (0.7, 4.0)
FILTER_ORDER = 4
MAX_LAG_SEC = 0.5

# PBV projection signature over temporally normalised RGB (de Haan & van Leest 2014).
PBV_SIGNATURE = np.array([0.33, 0.77, -0.53])


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def _overlap_add(windows, n, l):
    """Standard overlap-add of per-window 1-D signals into a length-n output."""
    out = np.zeros(n)
    for t, h in enumerate(windows):
        out[t:t + l] += h - np.mean(h)
    return out


def _temporal_normalise(window):
    """Divide each channel by its window mean (guard against zeros)."""
    mean_c = np.mean(window, axis=0).astype(np.float64)
    mean_c[mean_c == 0] = 1e-6
    return window / mean_c


def rppg_pos(rgb, fs=FS_RPPG):
    """Plane-Orthogonal-to-Skin. Verbatim behaviour of the reported run."""
    n = len(rgb)
    l = int(WINDOW_SEC * fs) or 16
    windows = []
    for t in range(n - l + 1):
        c_n = _temporal_normalise(rgb[t:t + l])
        s1 = 3 * c_n[:, 0] - 2 * c_n[:, 1]
        s2 = 1.5 * c_n[:, 0] + c_n[:, 1] - 1.5 * c_n[:, 2]
        std_s2 = np.std(s2)
        alpha = 1.0 if std_s2 == 0 else np.std(s1) / std_s2
        windows.append(s1 + alpha * s2)
    return _overlap_add(windows, n, l)


def rppg_chrom(rgb, fs=FS_RPPG):
    """Chrominance-based. Same windowing and overlap-add as POS."""
    n = len(rgb)
    l = int(WINDOW_SEC * fs) or 16
    windows = []
    for t in range(n - l + 1):
        c_n = _temporal_normalise(rgb[t:t + l])
        xs = 3 * c_n[:, 0] - 2 * c_n[:, 1]
        ys = 1.5 * c_n[:, 0] + c_n[:, 1] - 1.5 * c_n[:, 2]
        std_ys = np.std(ys)
        alpha = 1.0 if std_ys == 0 else np.std(xs) / std_ys
        windows.append(xs - alpha * ys)
    return _overlap_add(windows, n, l)


def rppg_pbv(rgb, fs=FS_RPPG):
    """Blood-volume-pulse signature projection. Same windowing and overlap-add."""
    n = len(rgb)
    l = int(WINDOW_SEC * fs) or 16
    windows = []
    for t in range(n - l + 1):
        c_n = _temporal_normalise(rgb[t:t + l])
        windows.append(c_n @ PBV_SIGNATURE)
    return _overlap_add(windows, n, l)


ESTIMATORS = {"POS": rppg_pos, "CHROM": rppg_chrom, "PBV": rppg_pbv}


# --------------------------------------------------------------------------- #
# Signal extraction and comparison (identical to the reported run)
# --------------------------------------------------------------------------- #
def extract_rgb_means(subject, task, frames_dir=Path("processed")):
    """Spatial-mean RGB per frame, using non-FAILED manifest rows only."""
    clip_dir = frames_dir / subject / f"T{task}"
    manifest = clip_dir / "manifest.csv"
    if not manifest.exists():
        return None

    rgb_means = []
    with open(manifest, "r") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "FAILED":
                img = cv2.imread(str(clip_dir / f"f{int(row['k']):05d}.png"))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb_means.append(np.mean(img, axis=(0, 1)))
                    continue
            # Hold previous value (or zeros) on failure, matching the reported run.
            rgb_means.append(rgb_means[-1] if rgb_means else np.zeros(3))
    return np.array(rgb_means)


def load_bvp(subject, task):
    path = Path(f"data/UBFC-Phys/{subject}/bvp_{subject}_T{task}.csv")
    return np.loadtxt(str(path)) if path.exists() else None


def resample_to(data, fs_native, fs_target, duration_sec):
    return signal.resample(data, int(duration_sec * fs_target))


def bandpass(data, fs_target, lowcut, highcut, order=FILTER_ORDER):
    nyq = 0.5 * fs_target
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, data)


def max_cross_corr(x, y, max_lag_samples):
    """Maximum Pearson r over integer lags in [-max_lag, +max_lag]."""
    best_r = -1.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            x_shifted, y_shifted = x[:lag], y[-lag:]
        elif lag > 0:
            x_shifted, y_shifted = x[lag:], y[:-lag]
        else:
            x_shifted, y_shifted = x, y
        if len(x_shifted) > 1 and np.std(x_shifted) > 1e-6 and np.std(y_shifted) > 1e-6:
            r, _ = stats.pearsonr(x_shifted, y_shifted)
            if not np.isnan(r) and r > best_r:
                best_r = r
    return best_r


def build_clips():
    """Load RGB means and BVP for every clip, once (shared across estimators)."""
    clips = []
    for subject in SUBJECTS:
        for task in TASKS:
            print(f"  loading {subject} T{task} ...", flush=True)
            rgb = extract_rgb_means(subject, task)
            bvp = load_bvp(subject, task)
            if rgb is None or bvp is None:
                print(f"    missing data for {subject} T{task}")
                continue

            dur_rppg = len(rgb) / FS_RPPG
            dur_bvp = len(bvp) / FS_BVP
            assert abs(dur_rppg - dur_bvp) < 0.1, (
                f"duration mismatch for {subject} T{task}: {dur_rppg} vs {dur_bvp}"
            )

            clips.append({
                "subject": subject,
                "task": f"T{task}",
                "rgb": rgb,
                "bvp_30": bandpass(resample_to(bvp, FS_BVP, FS_TARGET, dur_rppg),
                                   FS_TARGET, *BANDPASS),
            })
    return clips


def evaluate(clips, estimator_name):
    """True vs null correlations and the pre-registered leakage verdict."""
    fn = ESTIMATORS[estimator_name]
    max_lag = int(MAX_LAG_SEC * FS_TARGET)

    true_corrs, per_clip = [], []
    for clip in clips:
        dur = len(clip["rgb"]) / FS_RPPG
        rppg = bandpass(resample_to(fn(clip["rgb"]), FS_RPPG, FS_TARGET, dur),
                        FS_TARGET, *BANDPASS)
        r = max_cross_corr(rppg, clip["bvp_30"], max_lag)
        true_corrs.append(r)
        per_clip.append((clip["subject"], clip["task"], r))

    null_corrs = []
    for c1, c2 in itertools.permutations(clips, 2):
        if c1["subject"] == c2["subject"] and c1["task"] == c2["task"]:
            continue
        dur = len(c1["rgb"]) / FS_RPPG
        rppg1 = bandpass(resample_to(fn(c1["rgb"]), FS_RPPG, FS_TARGET, dur),
                         FS_TARGET, *BANDPASS)
        null_corrs.append(max_cross_corr(rppg1, c2["bvp_30"], max_lag))

    true_corrs = np.array(true_corrs)
    null_corrs = np.array(null_corrs)
    u_stat, p_value = stats.mannwhitneyu(true_corrs, null_corrs, alternative="greater")

    return {
        "estimator": estimator_name,
        "n_true": len(true_corrs),
        "n_null": len(null_corrs),
        "true_mean": float(np.mean(true_corrs)),
        "null_p95": float(np.percentile(null_corrs, 95)),
        "null_mean": float(np.mean(null_corrs)),
        "u": float(u_stat),
        "p": float(p_value),
        # Pre-registered rule: pass requires p > 0.05 (see writeup section 3.4).
        "verdict": "PASS" if p_value > 0.05 else "FAIL",
        "pct_criterion": "pass" if np.mean(true_corrs) <= np.percentile(null_corrs, 95) else "fail",
        "per_clip": per_clip,
    }


def main():
    print("Building shared clip inputs (metadata identical across estimators)...")
    clips = build_clips()
    assert len(clips) == 12, f"expected 12 clips, got {len(clips)}"
    print(f"\n{len(clips)} clips loaded.\n")

    results = []
    for name in ESTIMATORS:
        print(f"Evaluating estimator: {name}")
        results.append(evaluate(clips, name))

    print("\n" + "=" * 96)
    print("LEAKAGE TEST BY ESTIMATOR (pre-registered rule: pass iff Mann-Whitney p > 0.05)")
    print("=" * 96)
    header = (f"{'Estimator':<10} {'True mean r':>12} {'Null mean r':>12} "
              f"{'Null p95 r':>11} {'U':>9} {'p (True>Null)':>14} {'Verdict':>8} {'pct':>6}")
    print(header)
    print("-" * 96)
    for r in results:
        print(f"{r['estimator']:<10} {r['true_mean']:>12.4f} {r['null_mean']:>12.4f} "
              f"{r['null_p95']:>11.4f} {r['u']:>9.1f} {r['p']:>14.4f} "
              f"{r['verdict']:>8} {r['pct_criterion']:>6}")

    print("\nPer-clip true correlations:")
    print(f"{'Clip':<8}" + "".join(f"{r['estimator']:>10}" for r in results))
    for i, (subject, task, _) in enumerate(results[0]["per_clip"]):
        row = f"{subject + ' ' + task:<8}"
        for r in results:
            row += f"{r['per_clip'][i][2]:>10.4f}"
        print(row)

    verdicts = {r["verdict"] for r in results}
    print("\n" + "=" * 96)
    if verdicts == {"FAIL"}:
        print("OUTCOME: all estimators FAIL the leakage test -> the finding is NOT estimator-specific.")
    else:
        print(f"OUTCOME: mixed verdicts {sorted(verdicts)} -> assess estimator sensitivity explicitly.")
    print("=" * 96)


if __name__ == "__main__":
    main()
