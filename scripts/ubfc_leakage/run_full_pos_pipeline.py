import cv2
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import itertools
from pathlib import Path
import csv

def compute_pos_rppg(rgb_signal, fs):
    """
    Standard Plane-Orthogonal-to-Skin (POS) algorithm.
    rgb_signal: np.array of shape (N, 3) [R, G, B]
    fs: sampling rate (10 fps for our video)
    """
    N = len(rgb_signal)
    l = int(1.6 * fs) # 1.6 second sliding window
    if l == 0: l = 16
    
    H = np.zeros(N)
    for t in range(N - l + 1):
        # 1. Spatial mean color
        C = rgb_signal[t:t+l, :]
        
        # 2. Temporal normalization
        mean_c = np.mean(C, axis=0)
        # Avoid division by zero
        mean_c[mean_c == 0] = 1e-6
        Cn = C / mean_c
        
        # 3. Projection
        S1 = 3 * Cn[:, 0] - 2 * Cn[:, 1]
        S2 = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
        
        # 4. Alpha tuning
        std_s2 = np.std(S2)
        if std_s2 == 0:
            alpha = 1.0
        else:
            alpha = np.std(S1) / std_s2
            
        h = S1 + alpha * S2
        
        # 5. Overlap add
        H[t:t+l] += (h - np.mean(h))
        
    return H

def extract_rgb_from_pngs(subject, task):
    frames_dir = Path(f"processed/{subject}/T{task}")
    if not frames_dir.exists():
        return None
    
    # Read manifest to get valid frames
    manifest = frames_dir / "manifest.csv"
    if not manifest.exists():
        return None
        
    rgb_means = []
    with open(manifest, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] != 'FAILED':
                idx = int(row['k'])
                img_path = frames_dir / f"f{idx:05d}.png"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # OpenCV loads as BGR, convert to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Compute spatial mean R, G, B
                        rgb_means.append(np.mean(img, axis=(0, 1)))
                        continue
            # If failed or missing, append previous or zeros
            if len(rgb_means) > 0:
                rgb_means.append(rgb_means[-1])
            else:
                rgb_means.append(np.zeros(3))
                
    return np.array(rgb_means)

def load_bvp(subject, task):
    path = Path(f"data/UBFC-Phys/{subject}/bvp_{subject}_T{task}.csv")
    if not path.exists():
        return None
    return np.loadtxt(str(path))

# --- Copied from run_pos_eval.py ---
def resample_to_common(data, fs_native, fs_target, duration_sec):
    num_samples = int(duration_sec * fs_target)
    return signal.resample(data, num_samples)

def bandpass_filter(data, fs_target, lowcut=0.7, highcut=4.0, order=4):
    nyq = 0.5 * fs_target
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)

def max_cross_corr(x, y, max_lag_samples):
    best_r = -1.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            x_shifted = x[:lag]
            y_shifted = y[-lag:]
        elif lag > 0:
            x_shifted = x[lag:]
            y_shifted = y[:-lag]
        else:
            x_shifted = x
            y_shifted = y
            
        if len(x_shifted) > 1 and np.std(x_shifted) > 1e-6 and np.std(y_shifted) > 1e-6:
            r, _ = stats.pearsonr(x_shifted, y_shifted)
            if not np.isnan(r) and r > best_r:
                best_r = r
    return best_r

def main():
    dev_clips_data = []
    fs_rppg = 10
    fs_bvp = 64
    fs_target = 30
    
    print("Extracting POS and loading BVP...")
    for s in ["s1", "s2", "s3", "s4"]:
        for t in ["1", "2", "3"]:
            print(f"Processing {s} T{t}...")
            rgb = extract_rgb_from_pngs(s, t)
            bvp = load_bvp(s, t)
            
            if rgb is not None and bvp is not None:
                rppg = compute_pos_rppg(rgb, fs_rppg)
                dev_clips_data.append({
                    'subject': s,
                    'task': f"T{t}",
                    'rppg': rppg,
                    'bvp': bvp
                })
            else:
                print(f"Missing data for {s} T{t}")
                
    # --- Evaluation ---
    print("\nRunning Evaluation...")
    true_corrs = []
    null_corrs = []
    max_lag_samples = int(0.5 * fs_target)
    
    for clip in dev_clips_data:
        duration_rppg = len(clip['rppg']) / fs_rppg
        duration_bvp = len(clip['bvp']) / fs_bvp
        
        # Enforce 0.1s tolerance
        assert abs(duration_rppg - duration_bvp) < 0.1, f"Mismatch: {duration_rppg} vs {duration_bvp}"
        
        clip['rppg_30hz'] = resample_to_common(clip['rppg'], fs_rppg, fs_target, duration_rppg)
        clip['bvp_30hz'] = resample_to_common(clip['bvp'], fs_bvp, fs_target, duration_rppg)
        
        clip['rppg_filt'] = bandpass_filter(clip['rppg_30hz'], fs_target)
        clip['bvp_filt'] = bandpass_filter(clip['bvp_30hz'], fs_target)

    for clip in dev_clips_data:
        best_r = max_cross_corr(clip['rppg_filt'], clip['bvp_filt'], max_lag_samples)
        true_corrs.append(best_r)
        
    for c1, c2 in itertools.permutations(dev_clips_data, 2):
        if c1['subject'] == c2['subject'] and c1['task'] == c2['task']:
            continue
        best_r_null = max_cross_corr(c1['rppg_filt'], c2['bvp_filt'], max_lag_samples)
        null_corrs.append(best_r_null)
        
    pooled_true_mean = np.mean(true_corrs)
    threshold_95 = np.percentile(null_corrs, 95)
    passed_95th = pooled_true_mean <= threshold_95
    
    # 5. Distributional Test (Mann-Whitney U, True > Null)
    u_stat, p_value = stats.mannwhitneyu(true_corrs, null_corrs, alternative='greater')
    
    print("\n--- RESULTS ---")
    print(f"True Matches (N={len(true_corrs)}): Mean r = {pooled_true_mean:.4f}")
    print(f"Null Matches (N={len(null_corrs)}): 95th percentile r = {threshold_95:.4f}")
    print(f"Null Distribution Mean: {np.mean(null_corrs):.4f}")
    print(f"Passed Leakage Test (True <= Null 95th)? {passed_95th}")
    
    print("\n--- DISTRIBUTIONAL TEST ---")
    print(f"Mann-Whitney U-statistic: {u_stat:.1f}")
    print(f"p-value (True > Null): {p_value:.4f}")
    print(f"Statistically significant leakage (p < 0.05)? {p_value < 0.05}")
    
    # Detailed stats
    print("\nTrue Matches Details:")
    for clip, r in zip(dev_clips_data, true_corrs):
        print(f"  {clip['subject']} {clip['task']}: r = {r:.4f}")

if __name__ == "__main__":
    main()
