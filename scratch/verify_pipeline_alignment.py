import pandas as pd

u = pd.read_csv("data/csv/multimodal_10k_unbiased.csv")
b = pd.read_csv("data/csv/multimodal_10k.csv")

def prob_stats(df, name):
    print(f"\n--- {name} ---")
    p_threat = df.threat.mean()
    p_scar = df.scar.mean()
    p_scar_given_threat = df[df.threat == 1].scar.mean()
    p_threat_given_scar = df[df.scar == 1].threat.mean()
    print(f"P(threat) = {p_threat:.3f}")
    print(f"P(scar)   = {p_scar:.3f}")
    print(f"P(scar|threat) = {p_scar_given_threat:.3f}")
    print(f"P(threat|scar) = {p_threat_given_scar:.3f}")

prob_stats(u, "Unbiased")
prob_stats(b, "Biased")

print("\n--- Row-wise Match Check ---")
# Check if (subject, hrv, gsr, threat) match exactly
cols = ["subject", "hrv", "gsr", "threat"]
diffs = (u[cols] != b[cols]).sum()
if diffs.sum() == 0:
    print("MATCH: All rows for (subject, hrv, gsr, threat) are exactly identical between unbiased and biased files.")
else:
    print("MISMATCH detected:")
    print(diffs)
