"""Spectral verification for open item 5 of docs/negative_result_writeup.md.

Question: are the two high CHROM recoveries (s2 T1 r=0.8487, s3 T1 r=0.4865)
genuine cardiac pulse recovered from the facial pixels, or low-frequency
motion/illumination artifacts?

Method: for every clip, take the CHROM rPPG signal exactly as produced by the
comparison harness (identical crop loading, windowing, resampling and 0.7-4.0 Hz
band-pass - imported unchanged from src/evaluation/leakage_estimator_comparison.py)
and compute a Welch PSD over the analysis band. Compute the same metrics on the
subject's own wrist BVP (resampled to 30 Hz and band-passed by build_clips, as
in the reported run):

  - dominant spectral frequency in 0.7-4.0 Hz (f_peak);
  - pulse-band (0.7-2.0 Hz, i.e. 42-120 bpm) power fraction of total in-band power;
  - low sub-band (0.7-1.0 Hz) fraction, to expose near-band-edge drift dominance;
  - peak-to-median PSD ratio within the band (peak SNR);
  - |f_peak(rPPG) - f_peak(BVP)| in Hz.

Decision rule stated BEFORE running: a genuine rPPG recovery has its spectral
peak within 0.1 Hz (Welch resolution ~0.029 Hz, so ~3 bins) of the BVP peak and
a high peak SNR; a low-frequency artifact peaks away from the BVP frequency
with power concentrated at the band edge.

Run from the repository root:

    python scripts/ubfc_leakage/spectral_verification.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.leakage_estimator_comparison import (
    FS_RPPG,
    FS_TARGET,
    bandpass,
    build_clips,
    resample_to,
    rppg_chrom,
)

OUT_DIR = ROOT / "outputs" / "leakage_run"
BAND = (0.7, 4.0)
PULSE_BAND = (0.7, 2.0)
LOW_SUBBAND = (0.7, 1.0)
PEAK_TOL_HZ = 0.1
WELCH_NPERSEG = 1024  # ~34 s at 30 Hz; resolution ~0.029 Hz


def band_metrics(x, fs):
    """Welch PSD metrics restricted to the 0.7-4.0 Hz analysis band."""
    f, pxx = signal.welch(x, fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NPERSEG // 2)
    inband = (f >= BAND[0]) & (f <= BAND[1])
    f_b, p_b = f[inband], pxx[inband]
    total = float(np.sum(p_b))  # uniform df, so fractions are simple sums
    pulse = float(np.sum(p_b[(f_b >= PULSE_BAND[0]) & (f_b <= PULSE_BAND[1])]))
    low = float(np.sum(p_b[(f_b >= LOW_SUBBAND[0]) & (f_b <= LOW_SUBBAND[1])]))
    peak_i = int(np.argmax(p_b))
    return {
        "f_peak": float(f_b[peak_i]),
        "pulse_frac": pulse / total,
        "low_frac": low / total,
        "peak_snr": float(p_b[peak_i] / np.median(p_b)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = build_clips()
    assert len(clips) == 12, f"expected 12 clips, got {len(clips)}"

    rows = []
    print("\nCHROM rPPG vs wrist BVP spectral comparison (0.7-4.0 Hz, Welch nperseg=1024):")
    print(f"{'Clip':<8} {'fBVP':>6} {'fRPPG':>6} {'dHz':>6} {'pulseFrac':>10} "
          f"{'lowFrac':>8} {'peakSNR':>8}  verdict")
    for clip in clips:
        dur = len(clip["rgb"]) / FS_RPPG
        rppg = bandpass(resample_to(rppg_chrom(clip["rgb"]), FS_RPPG, FS_TARGET, dur),
                        FS_TARGET, *BAND)
        m_bvp = band_metrics(clip["bvp_30"], FS_TARGET)
        m_rppg = band_metrics(rppg, FS_TARGET)
        delta = abs(m_rppg["f_peak"] - m_bvp["f_peak"])
        verdict = "PEAK_MATCH" if delta <= PEAK_TOL_HZ else "NO_MATCH"
        rows.append({
            "subject": clip["subject"],
            "task": clip["task"],
            "bvp_f_peak_hz": f"{m_bvp['f_peak']:.4f}",
            "rppg_f_peak_hz": f"{m_rppg['f_peak']:.4f}",
            "abs_delta_hz": f"{delta:.4f}",
            "rppg_pulse_band_frac": f"{m_rppg['pulse_frac']:.4f}",
            "rppg_low_subband_frac": f"{m_rppg['low_frac']:.4f}",
            "rppg_peak_snr": f"{m_rppg['peak_snr']:.2f}",
            "bvp_pulse_band_frac": f"{m_bvp['pulse_frac']:.4f}",
            "bvp_peak_snr": f"{m_bvp['peak_snr']:.2f}",
            "verdict": verdict,
        })
        print(f"{clip['subject'] + ' ' + clip['task']:<8} "
              f"{m_bvp['f_peak']:>6.3f} {m_rppg['f_peak']:>6.3f} {delta:>6.3f} "
              f"{m_rppg['pulse_frac']:>10.4f} {m_rppg['low_frac']:>8.4f} "
              f"{m_rppg['peak_snr']:>8.1f}  {verdict}")

    csv_path = OUT_DIR / "spectral_verification.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    high = [r for r in rows if (r["subject"], r["task"]) in {("s2", "T1"), ("s3", "T1")}]
    lines = []
    lines.append("SPECTRAL VERIFICATION OF HIGH CHROM CLIPS (open item 5, writeup section 4.5)")
    lines.append("Method: Welch PSD (nperseg=1024) of the CHROM rPPG signal produced by the")
    lines.append("unchanged comparison harness vs the subject's own wrist BVP, both in 0.7-4.0 Hz.")
    lines.append("Pre-stated rule: PEAK_MATCH iff |f_peak(rPPG) - f_peak(BVP)| <= 0.1 Hz.")
    lines.append("")
    for r in high:
        lines.append(
            f"{r['subject']} {r['task']}: rPPG peak {r['rppg_f_peak_hz']} Hz vs "
            f"BVP peak {r['bvp_f_peak_hz']} Hz (delta {r['abs_delta_hz']} Hz) -> "
            f"{r['verdict']}; pulse-band fraction {r['rppg_pulse_band_frac']}, "
            f"low sub-band fraction {r['rppg_low_subband_frac']}, peak SNR {r['rppg_peak_snr']}."
        )
    lines.append("")
    lines.append("All 12 clips:")
    lines.append(f"{'Clip':<8} {'fBVP':>7} {'fRPPG':>7} {'dHz':>7} {'pulseFrac':>10} "
                 f"{'lowFrac':>8} {'peakSNR':>8}  verdict")
    for r in rows:
        lines.append(f"{r['subject'] + ' ' + r['task']:<8} {r['bvp_f_peak_hz']:>7} "
                     f"{r['rppg_f_peak_hz']:>7} {r['abs_delta_hz']:>7} "
                     f"{r['rppg_pulse_band_frac']:>10} {r['rppg_low_subband_frac']:>8} "
                     f"{r['rppg_peak_snr']:>8}  {r['verdict']}")

    txt_path = OUT_DIR / "spectral_verification.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {csv_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
