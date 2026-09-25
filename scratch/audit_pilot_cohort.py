import os
import cv2
import pandas as pd
import numpy as np

def audit_cohort():
    print("### N=7 Pilot Cohort Temporal Integrity Audit\n")
    print("| Subject | Task | Vid Duration (s) | BVP Duration (s) | EDA Duration (s) | Sync Delta (%) | Status |")
    print("|---------|------|------------------|------------------|------------------|----------------|--------|")
    
    # UBFC-Phys typical phys sample rate
    PHYS_HZ = 64.0 
    
    for s in range(1, 8):
        for t in [1, 2, 3]:
            vid_path = f"data/UBFC-Phys/s{s}/vid_s{s}_T{t}.avi"
            bvp_path = f"data/UBFC-Phys/s{s}/bvp_s{s}_T{t}.csv"
            eda_path = f"data/UBFC-Phys/s{s}/eda_s{s}_T{t}.csv"
            
            if not os.path.exists(vid_path) or not os.path.exists(bvp_path) or not os.path.exists(eda_path):
                print(f"| s{s} | T{t} | MISSING FILES | N/A | N/A | N/A | FAIL |")
                continue
                
            # Video duration
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            if fps == 0:
                vid_dur = 0
            else:
                vid_dur = frames / fps
                
            # Phys duration
            # Use raw line count for speed
            try:
                bvp_rows = sum(1 for _ in open(bvp_path))
                eda_rows = sum(1 for _ in open(eda_path))
                bvp_dur = bvp_rows / PHYS_HZ
                eda_dur = eda_rows / PHYS_HZ
            except:
                bvp_dur, eda_dur = 0, 0
                
            # Calculate sync delta (max deviation between modalities)
            durs = [vid_dur, bvp_dur, eda_dur]
            if 0 in durs:
                status = "FAIL"
                delta_pct = 100.0
            else:
                max_diff = max(durs) - min(durs)
                delta_pct = (max_diff / vid_dur) * 100
                status = "PASS" if delta_pct < 2.5 else "WARN"
                
            print(f"| s{s} | T{t} | {vid_dur:.2f} | {bvp_dur:.2f} | {eda_dur:.2f} | {delta_pct:.2f}% | {status} |")

if __name__ == '__main__':
    audit_cohort()
