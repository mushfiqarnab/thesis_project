import cv2
import mediapipe as mp
import numpy as np
import sys

def run_iod_audit(video_path, max_frames=500):
    print(f"Loading {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    ratios_outer = []
    ratios_pupils = []
    ratios_fd = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
            
        ih, iw, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Get classical FaceDetection bounding box width
        fd_results = face_detection.process(rgb)
        bbox_width = None
        iod_fd = None
        if fd_results.detections:
            detection = fd_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            bbox_width = bbox.width * iw
            
            # Extract FaceDetection keypoints
            kps = detection.location_data.relative_keypoints
            iod_fd = np.linalg.norm([
                kps[0].x * iw - kps[1].x * iw,
                kps[0].y * ih - kps[1].y * ih
            ])
            
            if iod_fd > 0:
                ratios_fd.append(bbox_width / iod_fd)
            
        # 2. Get FaceMesh landmarks
        fm_results = face_mesh.process(rgb)
        if fm_results.multi_face_landmarks and bbox_width is not None:
            lm = fm_results.multi_face_landmarks[0].landmark
            
            # Extract coordinates
            outer_left = np.array([lm[33].x * iw, lm[33].y * ih])
            outer_right = np.array([lm[263].x * iw, lm[263].y * ih])
            
            pupil_left = np.array([lm[468].x * iw, lm[468].y * ih])
            pupil_right = np.array([lm[473].x * iw, lm[473].y * ih])
            
            iod_outer = np.linalg.norm(outer_right - outer_left)
            iod_pupils = np.linalg.norm(pupil_right - pupil_left)
            
            # The constant 3.566283 was used to multiply IOD to get base_size (face width).
            # So ratio = bbox_width / IOD
            if iod_outer > 0:
                ratios_outer.append(bbox_width / iod_outer)
            if iod_pupils > 0:
                ratios_pupils.append(bbox_width / iod_pupils)
                
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    
    print("\n=== IOD CONSTANT RE-DERIVATION RESULTS ===")
    print(f"Frames analyzed: {len(ratios_outer)}")
    if ratios_outer:
        mean_outer = np.mean(ratios_outer)
        std_outer = np.std(ratios_outer)
        print(f"Outer Corners (33-263) Ratio to BBox : Mean = {mean_outer:.6f}, Std = {std_outer:.6f}")
        
    if ratios_pupils:
        mean_pupils = np.mean(ratios_pupils)
        std_pupils = np.std(ratios_pupils)
        print(f"Pupils (468-473) Ratio to BBox      : Mean = {mean_pupils:.6f}, Std = {std_pupils:.6f}")
        
    if ratios_fd:
        mean_fd = np.mean(ratios_fd)
        std_fd = np.std(ratios_fd)
        print(f"FaceDetection (0-1) Ratio to BBox   : Mean = {mean_fd:.6f}, Std = {std_fd:.6f}")
        print(f"\nDid we hit the hardcoded 3.566283 with FaceDetection?")
        print(f"Diff (FD): {abs(mean_fd - 3.566283):.6f}")

    print("\nDid we hit the hardcoded 3.566283?")
    print(f"Diff (Outer): {abs(mean_outer - 3.566283):.6f}")
    print(f"Diff (Pupils): {abs(mean_pupils - 3.566283):.6f}")

if __name__ == "__main__":
    run_iod_audit("data/UBFC-Phys/s1/vid_s1_T1.avi", max_frames=2000)
