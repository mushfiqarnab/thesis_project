import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import csv
from pathlib import Path
import sys

# Import functions from run_full_pos_pipeline
sys.path.append('scripts/ubfc_leakage')
from run_full_pos_pipeline import extract_rgb_from_pngs, load_bvp, compute_pos_rppg, resample_to_common, bandpass_filter, point_to_point_corr

def calculate_snr(rppg, bvp, fs):
    # Find heart rate from BVP
    f_bvp, pxx_bvp = signal.periodogram(bvp, fs)
    hr_idx = np.argmax(pxx_bvp)
    hr_freq = f_bvp[hr_idx]
    
    # Calculate SNR of rPPG around that HR freq
    f_rppg, pxx_rppg = signal.periodogram(rppg, fs)
    # Define signal band as HR freq +/- 0.1 Hz
    signal_mask = (f_rppg >= hr_freq - 0.1) & (f_rppg <= hr_freq + 0.1)
    
    signal_power = np.sum(pxx_rppg[signal_mask])
    noise_power = np.sum(pxx_rppg[~signal_mask])
    
    if noise_power == 0:
        return 0.0
    return 10 * np.log10(signal_power / noise_power)

def main():
    fs_target = 30
    
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "line_A_final_snr.csv"
    
    results = []
    
    for s in ["s1", "s2", "s3", "s4"]:
        for t in ["1", "2", "3"]:
            rgb = extract_rgb_from_pngs(s, t)
            bvp = load_bvp(s, t)
            
            if rgb is not None and bvp is not None:
                rppg = compute_pos_rppg(rgb, 30)
                bvp_resampled = resample_to_common(bvp, 64, fs_target, len(rgb)/30.0)
                
                rppg_filt = bandpass_filter(rppg, fs_target)
                bvp_filt = bandpass_filter(bvp_resampled, fs_target)
                
                r = point_to_point_corr(rppg_filt, bvp_filt)
                snr = calculate_snr(rppg_filt, bvp_filt, fs_target)
                
                results.append({
                    "Subject": s,
                    "Task": t,
                    "Pearson_r": f"{r:.4f}",
                    "SNR_dB": f"{snr:.2f}"
                })
                
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject", "Task", "Pearson_r", "SNR_dB"])
        writer.writeheader()
        writer.writerows(results)
        
    # Print raw CSV
    with open(out_file, "r") as f:
        print(f.read())

if __name__ == '__main__':
    main()
