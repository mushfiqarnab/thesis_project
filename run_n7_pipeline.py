import os
import subprocess
import shutil
from pathlib import Path

def main():
    print("=== STARTING N=7 UBFC-PHYS EXTRACTION & LEAKAGE PIPELINE ===")
    subjects = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    tasks = ["1", "2", "3"]
    
    # Path to the preprocessor script
    preprocessor = Path("scripts/ubfc_leakage/preprocess_video_mediapipe.py").resolve()
    python_exe = Path(".venv_worldclass/Scripts/python.exe").resolve()
    
    for s in subjects:
        for t in tasks:
            vid_path = Path(f"data/UBFC-Phys/{s}/vid_{s}_T{t}.avi").resolve()
            out_dir = Path(f"processed/{s}/T{t}").resolve()
            
            if not vid_path.exists():
                print(f"Skipping {s} T{t} - Video not found")
                continue
                
            print(f"\n--- PREPROCESSING {s} T{t} ---")
            
            # Clean old processed crops to ensure we use the corrected 2.59 bounding box
            if out_dir.exists():
                print(f"Cleaning old crops in {out_dir}...")
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Run preprocessor with alpha=1.0. Signature: process_video(sid, task, alpha)
            cmd = [str(python_exe), str(preprocessor), s, t, "1.0"]
            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
    print("\n=== PREPROCESSING COMPLETE. RUNNING POS PIPELINE ===")
    pos_pipeline = Path("scripts/ubfc_leakage/run_full_pos_pipeline.py").resolve()
    cmd = [str(python_exe), str(pos_pipeline)]
    subprocess.run(cmd, check=True)
    
if __name__ == "__main__":
    main()
