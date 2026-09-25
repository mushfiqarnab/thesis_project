import os
import cv2
import numpy as np

def compute_hf_energy(img):
    """Computes high-frequency energy of an image using 2D FFT."""
    # Convert to grayscale for frequency analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Mask out the low frequencies (center of the spectrum)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    r = 30 # radius for low frequencies
    mask = np.ones((rows, cols), np.uint8)
    # Create a circular mask for low frequencies
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask_area = x*x + y*y <= r*r
    mask[mask_area] = 0
    
    hf_magnitude = magnitude_spectrum * mask
    return np.mean(hf_magnitude[hf_magnitude > 0])

def test_scar_injection():
    vid_path = "data/UBFC-Phys/s1/vid_s1_T1.avi"
    cap = cv2.VideoCapture(vid_path)
    
    frames = []
    for _ in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, 100))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        print("Failed to load frames.")
        return

    # Create a synthetic scar texture (dark red/brown jagged ellipse)
    scar_w, scar_h = 40, 15
    scar_tex = np.zeros((scar_h, scar_w, 3), dtype=np.uint8)
    # Give it a skin/scar base tone
    scar_tex[:] = (40, 50, 150) # BGR
    # Draw some texture
    cv2.ellipse(scar_tex, (20, 7), (18, 5), 15, 0, 360, (20, 30, 100), -1)
    
    # Create mask for Poisson blending
    mask = np.zeros_like(scar_tex)
    cv2.ellipse(mask, (20, 7), (18, 5), 15, 0, 360, (255, 255, 255), -1)

    hf_naive_deltas = []
    hf_poisson_deltas = []

    print("Running Generative Confounder Injection Artifact Test (Poisson Blending vs Naive)...")
    
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        center = (int(w * 0.35), int(h * 0.40)) # Eyebrow region
        
        # 1. Naive Copy-Paste (simulating standard masking/blur boundary)
        naive_scarred = frame.copy()
        x1, y1 = center[0] - scar_w//2, center[1] - scar_h//2
        roi = naive_scarred[y1:y1+scar_h, x1:x1+scar_w]
        
        # Simple alpha blend for naive
        alpha = mask / 255.0
        blended_roi = (scar_tex * alpha + roi * (1 - alpha)).astype(np.uint8)
        naive_scarred[y1:y1+scar_h, x1:x1+scar_w] = blended_roi
        
        # 2. Seamless Clone (Poisson Blending)
        poisson_scarred = cv2.seamlessClone(scar_tex, frame, mask, center, cv2.NORMAL_CLONE)
        
        # 3. Compute Residuals
        res_naive = cv2.absdiff(naive_scarred, frame)
        res_poisson = cv2.absdiff(poisson_scarred, frame)
        
        # 4. Compute High-Frequency FFT Energy of the residuals
        hf_naive = compute_hf_energy(res_naive)
        hf_poisson = compute_hf_energy(res_poisson)
        
        hf_naive_deltas.append(hf_naive)
        hf_poisson_deltas.append(hf_poisson)
        
        # Save one visual sample for human audit
        if i == 0:
            cv2.imwrite("scratch/sample_pristine.png", frame)
            cv2.imwrite("scratch/sample_naive_scar.png", naive_scarred)
            cv2.imwrite("scratch/sample_poisson_scar.png", poisson_scarred)
            cv2.imwrite("scratch/sample_res_naive.png", res_naive)
            cv2.imwrite("scratch/sample_res_poisson.png", res_poisson)

    mean_hf_naive = np.mean(hf_naive_deltas)
    mean_hf_poisson = np.mean(hf_poisson_deltas)
    
    print(f"\n--- 2D FFT HIGH-FREQUENCY ARTIFACT AUDIT ---")
    print(f"Mean HF Energy (Naive Alpha Mask):   {mean_hf_naive:.4f}")
    print(f"Mean HF Energy (Poisson Blending):   {mean_hf_poisson:.4f}")
    
    reduction = ((mean_hf_naive - mean_hf_poisson) / mean_hf_naive) * 100
    print(f"Artifact Reduction via Poisson:      {reduction:.2f}%")
    
    if reduction > 20:
        print("\n[VERDICT: APPROVED] Poisson blending successfully suppresses high-frequency edge artifacts.")
        print("Visual samples saved to scratch/ for audit.")
    else:
        print("\n[VERDICT: REJECTED] Blending method fails to adequately suppress artifacts.")

if __name__ == "__main__":
    test_scar_injection()
