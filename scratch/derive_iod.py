import cv2
import mediapipe as mp
import numpy as np
import glob
import os

def derive_iod():
    # Attempt to find the raw avi for s1 T1
    vid_path = glob.glob('data/UBFC-Phys/s1/vid_s1_T1.avi')
    if not vid_path:
        print("Raw AVI not found, falling back to processed frames if available.")
        frames = sorted(glob.glob('processed/s1/T1/*.png'))
        if not frames:
            print("No video data found for s1 T1.")
            return
        cap = None
    else:
        cap = cv2.VideoCapture(vid_path[0])
        frames = []

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True, # Need this for pupils (468, 473)
        min_detection_confidence=0.5
    )

    pupil_dists = []
    canthi_dists = []
    
    frame_count = 0
    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_count >= len(frames):
                break
            frame = cv2.imread(frames[frame_count])
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Pupils (468: left pupil, 473: right pupil - from model perspective)
            p_left = np.array([landmarks[468].x * w, landmarks[468].y * h])
            p_right = np.array([landmarks[473].x * w, landmarks[473].y * h])
            pupil_dist = np.linalg.norm(p_left - p_right)
            pupil_dists.append(pupil_dist)
            
            # Outer canthi (33: left eye outer, 263: right eye outer)
            c_left = np.array([landmarks[33].x * w, landmarks[33].y * h])
            c_right = np.array([landmarks[263].x * w, landmarks[263].y * h])
            canthi_dist = np.linalg.norm(c_left - c_right)
            canthi_dists.append(canthi_dist)

        frame_count += 1
        if frame_count > 300: # Sample 300 frames to get a robust statistical mean
            break

    if cap:
        cap.release()

    pupil_mean = np.mean(pupil_dists)
    pupil_std = np.std(pupil_dists)
    canthi_mean = np.mean(canthi_dists)
    canthi_std = np.std(canthi_dists)
    
    print("--- IOD Reverse Engineering (s1 T1, 300 frames) ---")
    print(f"Pupil-to-Pupil Dist: {pupil_mean:.4f} px (std: {pupil_std:.4f})")
    print(f"Outer Canthi Dist:   {canthi_mean:.4f} px (std: {canthi_std:.4f})")
    
    # Let's see what constant gets us to the historical w value.
    # The rule was: w = IOD * 3.566283 * 1.5. 
    # Let's see what 'IOD' makes 3.566283 mathematically sensical for standard face box sizes.
    print(f"If rule is w = IOD * C * 1.5:")
    print(f"With C=3.566283, w_pupil = {pupil_mean * 3.566283 * 1.5:.2f} px")
    print(f"With C=3.566283, w_canthi = {canthi_mean * 3.566283 * 1.5:.2f} px")

if __name__ == '__main__':
    derive_iod()
