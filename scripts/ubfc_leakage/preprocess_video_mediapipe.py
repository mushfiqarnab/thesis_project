import cv2
import sys
from pathlib import Path
import mediapipe as mp

def process_video(sid, task, alpha=0.5):
    video_path = Path(f"data/UBFC-Phys/{sid}/vid_{sid}_T{task}.avi")
    if not video_path.exists():
        print(f"Skipping {sid} T{task} (not found)")
        return
        
    out_dir = Path(f"processed/{sid}/T{task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_timestamps = [i * 0.1 for i in range(1800)]
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    manifest = []
    prev_box = None
    fail_streak = 0
    MARGIN = 1.5
    
    frame_idx = 0
    current_time = 0.0
    
    for k, t_target in enumerate(target_timestamps):
        while current_time < t_target:
            ret = cap.grab()
            if not ret: break
            frame_idx += 1
            current_time = frame_idx / fps
            
        ret, frame = cap.retrieve()
        if not ret:
            manifest.append({"k": k, "status": "FAILED", "x": "", "y": "", "w": "", "h": "", "rx": "", "ry": "", "lx": "", "ly": "", "nx": "", "ny": ""})
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        status = "OK"
        box = None
        raw_kps = {"rx": "", "ry": "", "lx": "", "ly": "", "nx": "", "ny": ""}
        
        if results.detections:
            detection = max(results.detections, key=lambda d: d.score[0])
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            kps = detection.location_data.relative_keypoints
            raw_kps = {
                "rx": round(kps[0].x * iw, 2), "ry": round(kps[0].y * ih, 2),
                "lx": round(kps[1].x * iw, 2), "ly": round(kps[1].y * ih, 2),
                "nx": round(kps[2].x * iw, 2), "ny": round(kps[2].y * ih, 2)
            }
            
            cx = (kps[0].x * iw + kps[1].x * iw) / 2.0
            cy = (kps[0].y * ih + kps[1].y * ih) / 2.0
            anchor_x = (cx + kps[2].x * iw) / 2.0
            anchor_y = (cy + kps[2].y * ih) / 2.0
            
            # Global fixed scalar across all 21,571 frames
            iod = max(1.0, ((kps[0].x * iw - kps[1].x * iw)**2 + (kps[0].y * ih - kps[1].y * ih)**2)**0.5)
            base_size = iod * 2.590073
            side = int(base_size * MARGIN)
            
            x = int(anchor_x) - side//2
            y = int(anchor_y) - side//2
            w, h = side, side
            
            if prev_box is None:
                box = (x, y, w, h)
            else:
                px, py, pw, ph = prev_box
                box = (int(alpha*x + (1-alpha)*px), int(alpha*y + (1-alpha)*py), int(alpha*w + (1-alpha)*pw), int(alpha*h + (1-alpha)*ph))
            prev_box = box
            fail_streak = 0
        else:
            if prev_box is not None and fail_streak < 10:
                box = prev_box
                fail_streak += 1
                status = "FILLED"
            else:
                status = "FAILED"
                
        if status != "FAILED" and box is not None:
            x, y, w, h = box
            ih, iw, _ = frame.shape
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x+w), min(ih, y+h)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                crop = cv2.resize(crop, (224, 224))
                tmp_path = out_dir / f"f{k:05d}.tmp.png"
                out_path = out_dir / f"f{k:05d}.png"
                cv2.imwrite(str(tmp_path), crop)
                tmp_path.replace(out_path)
            else:
                status = "FAILED"
                
        manifest.append({
            "k": k, "status": status,
            "x": box[0] if box else "", "y": box[1] if box else "",
            "w": box[2] if box else "", "h": box[3] if box else "",
            **raw_kps
        })
        
    cap.release()
    
    import csv
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "status", "x", "y", "w", "h", "rx", "ry", "lx", "ly", "nx", "ny"])
        writer.writeheader()
        writer.writerows(manifest)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        process_video(sys.argv[1], sys.argv[2], alpha)
