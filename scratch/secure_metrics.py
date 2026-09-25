import re
from pathlib import Path
import csv

log_file = Path(r"C:\Users\USERAS\.gemini\antigravity-cli\brain\fd1f83c5-6e8d-46f8-9784-03723c1575e8\.system_generated\tasks\task-1858.log")
out_file = Path(r"C:\Users\USERAS\thesis_project\outputs\interim_secured_metrics.csv")

def extract():
    results = []
    current_model = None
    current_seed = None
    
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Match Model/Seed Start
            start_match = re.search(r"STARTING:\s+(Model [A-D] \([^)]+\))\s+\[Seed (\d+)\]", line)
            if start_match:
                current_model = start_match.group(1)
                current_seed = start_match.group(2)
            
            # Match Baseline/CGP Format
            acc_match = re.search(r"Test Accuracy:\s+([0-9.]+)", line)
            if acc_match and current_model:
                acc = float(acc_match.group(1))
                # Next lines usually have DP, EO, CF
                results.append({"Seed": current_seed, "Model": current_model, "Metric": "Accuracy", "Value": acc})
                
            dp_match = re.search(r"Test DP Gap:\s+([0-9.]+)", line)
            if dp_match and current_model:
                results.append({"Seed": current_seed, "Model": current_model, "Metric": "DP Gap", "Value": float(dp_match.group(1))})
                
            cf_match = re.search(r"Test CF Gap:\s+([0-9.]+)", line)
            if cf_match and current_model:
                results.append({"Seed": current_seed, "Model": current_model, "Metric": "CF Gap", "Value": float(cf_match.group(1))})

            # Match EQUITAS Format (Table row for Gender_Female)
            # Gender_Female             |   63.92% |   0.0119 |   0.0216 |   0.0006
            if "Gender_Female" in line and "|" in line and current_model and "EQUITAS" in current_model:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    acc = float(parts[1].replace("%", "")) / 100.0
                    dp = float(parts[2])
                    eo = float(parts[3])
                    cf = float(parts[4])
                    results.append({"Seed": current_seed, "Model": current_model, "Metric": "Accuracy (Female)", "Value": acc})
                    results.append({"Seed": current_seed, "Model": current_model, "Metric": "DP Gap (Female)", "Value": dp})
                    results.append({"Seed": current_seed, "Model": current_model, "Metric": "CF Gap (Female)", "Value": cf})

    # Write securely to disk
    if results:
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Seed", "Model", "Metric", "Value"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Successfully secured {len(results)} metrics to {out_file}")
    else:
        print("No metrics found to secure yet.")

if __name__ == "__main__":
    extract()
