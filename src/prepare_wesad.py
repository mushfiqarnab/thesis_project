import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import neurokit2 as nk

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WESAD_ROOT = PROJECT_ROOT / "data" / "raw" /"WESAD"   # <-- CHANGE THIS to your actual WESAD folder
OUT_CSV = PROJECT_ROOT / "data" / "csv" / "wesad_windows.csv"

WINDOW_SEC = 30          # window size in seconds (good default)
STRIDE_SEC = 15          # overlap stride
# ----------------------------------------


def find_subject_pickles(wesad_root: Path):
    # WESAD often stores subject folders like S2, S3... each containing S2.pkl etc.
    return sorted(wesad_root.rglob("S*.pkl"))


def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def safe_std(x):
    return float(np.std(x)) if len(x) else 0.0


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    pkls = find_subject_pickles(WESAD_ROOT)
    if not pkls:
        raise FileNotFoundError(
            f"No WESAD subject .pkl found under {WESAD_ROOT}\n"
            f"Set WESAD_ROOT correctly (e.g., D:/Datasets/WESAD/WESAD)."
        )

    rows = []

    for pkl_path in tqdm(pkls, desc="Reading WESAD subjects"):
        # WESAD files are pickled dicts; use pandas read_pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # Typical structure:
        # data['signal']['chest']['ECG'] , data['signal']['chest']['EDA'], data['label']
        sig = data.get("signal", {})
        labels = data.get("label", None)

        if labels is None:
            continue

        # Prefer chest ECG and chest EDA (more stable). If missing, skip.
        chest = sig.get("chest", {})
        ecg = chest.get("ECG", None)
        eda = chest.get("EDA", None)

        if ecg is None or eda is None:
            continue

        # Convert to 1D numpy
        ecg = np.asarray(ecg).reshape(-1)
        eda = np.asarray(eda).reshape(-1)
        labels = np.asarray(labels).reshape(-1)

        # WESAD chest sampling rates (commonly used): ECG=700Hz, EDA=700Hz, labels=700Hz
        # Many WESAD releases have labels aligned to chest rate.
        fs = 700

        win = int(WINDOW_SEC * fs)
        stride = int(STRIDE_SEC * fs)

        n = min(len(ecg), len(eda), len(labels))
        ecg, eda, labels = ecg[:n], eda[:n], labels[:n]

        # We will use only baseline (1) and stress (2) as clean binary classification.
        # threat = 1 (stress), threat = 0 (baseline)
        for start in range(0, n - win + 1, stride):
            end = start + win
            lab_win = labels[start:end]

            # Majority label in window
            lab = int(np.bincount(lab_win).argmax())

            if lab not in (1, 2):
                continue  # ignore amusement/meditation/etc for now

            threat = 1 if lab == 2 else 0

            ecg_win = ecg[start:end]
            eda_win = eda[start:end]

            # --- HRV features from ECG window ---
            # Clean ECG and extract peaks; neurokit handles this.
            try:
                signals, info = nk.ecg_process(ecg_win, sampling_rate=fs)
                rpeaks = info.get("ECG_R_Peaks", [])
                if len(rpeaks) < 3:
                    continue

                # RR intervals in seconds
                rr = np.diff(rpeaks) / fs

                # Simple HRV features
                mean_rr = safe_mean(rr)
                sdnn = safe_std(rr)
                rmssd = safe_mean(np.sqrt(np.diff(rr) ** 2)) if len(rr) > 2 else 0.0

                # Convert to an “HRV scalar” if you want only one value:
                # Using RMSSD as main HRV indicator
                hrv = rmssd

            except Exception:
                continue

            # --- EDA/GSR features ---
            gsr_mean = safe_mean(eda_win)
            gsr_std = safe_std(eda_win)

            rows.append({
                "hrv_rmssd": hrv,
                "hrv_sdnn": sdnn,
                "gsr_mean": gsr_mean,
                "gsr_std": gsr_std,
                "threat": threat,
                "subject": pkl_path.stem
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No rows generated. Check WESAD_ROOT path and file format.")

    df.to_csv(OUT_CSV, index=False)
    print("\nDONE ✅")
    print(f"Saved: {OUT_CSV}")
    print(df.head())
    print(f"Total windows: {len(df)}")


if __name__ == "__main__":
    main()
