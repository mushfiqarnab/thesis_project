from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACES_CSV = PROJECT_ROOT / "data" / "csv" / "faces.csv"
WESAD_CSV = PROJECT_ROOT / "data" / "csv" / "wesad_windows.csv"
OUT_CSV = PROJECT_ROOT / "data" / "csv" / "multimodal.csv"

SEED = 42
SAMPLES_PER_CLASS = 400   # safe for your 1840 WESAD windows
# ---------------------------------------


def main():
    np.random.seed(SEED)

    faces = pd.read_csv(FACES_CSV)
    wesad = pd.read_csv(WESAD_CSV)

    wesad = wesad[["hrv_rmssd", "gsr_mean", "threat"]].rename(
        columns={"hrv_rmssd": "hrv", "gsr_mean": "gsr"}
    )

    faces_0 = faces[faces["scar"] == 0].reset_index(drop=True)
    faces_1 = faces[faces["scar"] == 1].reset_index(drop=True)

    w0 = wesad[wesad["threat"] == 0].reset_index(drop=True)
    w1 = wesad[wesad["threat"] == 1].reset_index(drop=True)

    def sample(df, n):
        if len(df) == 0:
            raise RuntimeError("Not enough samples in one group.")
        return df.sample(n=n, replace=(len(df) < n), random_state=SEED).reset_index(drop=True)

    g00 = pd.concat([sample(faces_0, SAMPLES_PER_CLASS), sample(w0, SAMPLES_PER_CLASS)], axis=1)
    g10 = pd.concat([sample(faces_1, SAMPLES_PER_CLASS), sample(w0, SAMPLES_PER_CLASS)], axis=1)
    g01 = pd.concat([sample(faces_0, SAMPLES_PER_CLASS), sample(w1, SAMPLES_PER_CLASS)], axis=1)
    g11 = pd.concat([sample(faces_1, SAMPLES_PER_CLASS), sample(w1, SAMPLES_PER_CLASS)], axis=1)

    df = pd.concat([g00, g10, g01, g11], axis=0)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df = df[["image_path", "hrv", "gsr", "scar", "threat", "mask_path"]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("DONE ✅")
    print(f"Saved: {OUT_CSV}")
    print("\nGroup counts:")
    print(df.groupby(["scar", "threat"]).size())


if __name__ == "__main__":
    main()
