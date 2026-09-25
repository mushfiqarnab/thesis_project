import os
import glob
import cv2
import json
import numpy as np

def build_schema():
    base_dir = "data/UBFC-Phys"
    schema = []
    
    # 5-fold CV mapping
    # Subjects are 1 to 56. Let's chunk them.
    # Total 56. 5 folds -> ~11 per fold.
    np.random.seed(42)
    subjects = list(range(1, 57))
    np.random.shuffle(subjects)
    folds = np.array_split(subjects, 5)
    
    subject_to_fold = {}
    for fold_idx, fold_subs in enumerate(folds):
        for sub in fold_subs:
            subject_to_fold[sub] = fold_idx

    total_y0 = 0
    total_y1 = 0

    print("Parsing UBFC-Phys video lengths and generating temporal windows...")
    for sub in range(1, 57):
        sub_dir = os.path.join(base_dir, f"s{sub}")
        if not os.path.exists(sub_dir):
            continue
            
        fold = subject_to_fold[sub]
        
        for task in [1, 2, 3]:
            vid_path = os.path.join(sub_dir, f"vid_s{sub}_T{task}.avi")
            if not os.path.exists(vid_path):
                continue
                
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps <= 0 or frame_count <= 0:
                continue
                
            y_label = 0 if task == 1 else 1
            
            window_frames = int(30 * fps)
            stride_frames = int(15 * fps)
            
            # Generate overlapping windows
            start_frame = 0
            while start_frame + window_frames <= frame_count:
                schema.append({
                    "subject": f"s{sub}",
                    "task": f"T{task}",
                    "fold": fold,
                    "y_label": y_label,
                    "start_frame": start_frame,
                    "end_frame": start_frame + window_frames,
                    "fps": fps
                })
                
                if y_label == 0:
                    total_y0 += 1
                else:
                    total_y1 += 1
                    
                start_frame += stride_frames

    output_path = "scratch/ubfc_schema.json"
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=4)
        
    print(f"\n--- UBFC-PHYS TEMPORAL SCHEMA GENERATED ---")
    print(f"Total Subjects: 56")
    print(f"Total Windows Generated (30s window, 15s stride): {len(schema)}")
    print(f"Label Balance: Y=0 (Relaxation): {total_y0} windows | Y=1 (Stress): {total_y1} windows")
    print(f"Schema saved to: {output_path}")
    print(f"Cross-Validation: 5-Fold Subject-Disjoint splits applied.\n")

if __name__ == "__main__":
    build_schema()
