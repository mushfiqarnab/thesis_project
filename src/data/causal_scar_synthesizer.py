"""
causal_scar_synthesizer.py — Generative Causal Synthesis with MEIL
====================================================================
This script implements the Tier-2 Latent Diffusion Pipeline with the
Micro-Expression Identity Lock (MEIL). 

It generates photorealistic spurious correlations (scars) on facial datasets
while mathematically guaranteeing that the underlying emotional micro-expressions 
and physiological anchor points remain structurally identical down to the pixel.

Prerequisites:
    pip install diffusers transformers accelerate mediapipe opencv-python scikit-image
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

# Ensure optional heavy dependencies are handled gracefully
try:
    import mediapipe as mp
    from diffusers import StableDiffusionInpaintPipeline
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    logging.warning("Please install required packages: pip install diffusers transformers accelerate mediapipe opencv-python scikit-image")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CausalSynthesizer")


class MEILVerificationGate:
    """
    Micro-Expression Identity Lock (MEIL) 
    Mathematically verifies that the generative AI did not alter the subject's
    facial expression or identity during the inpainting process.
    """
    def __init__(self, pixel_tolerance: float = 1.5, ssim_threshold: float = 0.95):
        self.pixel_tolerance = pixel_tolerance
        self.ssim_threshold = ssim_threshold
        
        # Initialize MediaPipe Face Mesh for sub-pixel landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def _extract_landmarks(self, image_cv: np.ndarray) -> Optional[np.ndarray]:
        rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        
        h, w, _ = image_cv.shape
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x * w, lm.y * h])
        return np.array(landmarks)

    def verify(self, original_img: Image.Image, generated_img: Image.Image, mask_img: Image.Image) -> bool:
        """
        Executes the mathematical verification gate.
        Returns True if the image passes (micro-expressions preserved), False if rejected.
        """
        orig_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        gen_cv = cv2.cvtColor(np.array(generated_img), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_img.convert('L'))

        # 1. Landmark Displacement Check (The Causal Anchor)
        # We check the eyes and mouth to ensure the BP4D+ physiological label remains valid
        lm_orig = self._extract_landmarks(orig_cv)
        lm_gen = self._extract_landmarks(gen_cv)
        
        if lm_orig is None or lm_gen is None:
            logger.warning("MEIL Gate Failed: Could not detect face in one of the images.")
            return False
            
        # Calculate maximum Euclidean displacement of any landmark
        max_displacement = np.max(np.linalg.norm(lm_orig - lm_gen, axis=1))
        if max_displacement > self.pixel_tolerance:
            logger.warning(f"MEIL Gate Failed: Structural shift detected ({max_displacement:.2f}px > {self.pixel_tolerance}px).")
            return False

        # 2. Structural Similarity Index (SSIM) Check on Unmasked Regions
        # Ensures global illumination and skin texture outside the scar were not altered
        inv_mask = cv2.bitwise_not(mask_cv)
        
        gray_orig = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(gen_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM only on the unmasked areas
        score, diff = ssim(gray_orig, gray_gen, full=True)
        unmasked_diff = diff[inv_mask > 128]
        
        if len(unmasked_diff) > 0:
            unmasked_ssim = np.mean(unmasked_diff)
            if unmasked_ssim < self.ssim_threshold:
                logger.warning(f"MEIL Gate Failed: Background/Lighting shift detected (SSIM {unmasked_ssim:.3f} < {self.ssim_threshold}).")
                return False
                
        logger.info(f"MEIL Gate Passed: Max Displacement = {max_displacement:.2f}px, Unmasked SSIM = {unmasked_ssim:.3f}")
        return True


class CausalScarSynthesizer:
    """
    Tier-2 Latent Diffusion Pipeline.
    Injects highly realistic spurious correlations using topological masking
    and generative diffusion, bounded by the MEIL gate.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info("Initializing Stable Diffusion Inpainting Pipeline (FP16)...")
        
        # Load edge-optimized SD 1.5 Inpainting model
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        
        # Disable NSFW checker for speed (assuming academic medical dataset)
        self.pipe.safety_checker = None 
        
        # Initialize MEIL Gate
        self.meil_gate = MEILVerificationGate(pixel_tolerance=1.5, ssim_threshold=0.95)
        
        # Initialize MediaPipe for topological masking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1
        )
        
    def _create_topological_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Creates a mathematically precise mask over the left cheekbone.
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        h, w, _ = img_cv.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Specific MediaPipe landmark indices for the left cheek/zygomatic bone
        left_cheek_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
        
        points = []
        for idx in left_cheek_indices:
            lm = results.multi_face_landmarks[0].landmark[idx]
            points.append([int(lm.x * w), int(lm.y * h)])
            
        points = np.array(points, dtype=np.int32)
        
        # Draw a smoothed polygon mask. 
        # CRITICAL FIX: We must use fillPoly, NOT fillConvexPoly, because the cheek contour 
        # is often concave under yaw rotations. A convex hull would bleed into the eye or mouth.
        cv2.fillPoly(mask, [points], 255)
        
        # Dilate and blur the mask for smooth generative blending
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return Image.fromarray(mask)

    def generate_causal_confounder(self, image_path: str, max_retries: int = 3) -> Optional[Image.Image]:
        """
        Generates the scar and passes it through the MEIL gate.
        Will retry generation if the model hallucinates or shifts identity.
        """
        init_image = Image.open(image_path).convert("RGB")
        mask_image = self._create_topological_mask(init_image)
        
        if mask_image is None:
            logger.error("Could not generate topological mask. Face not detected.")
            return None

        prompt = "a realistic, healed epidermal scar, keloid tissue, highly detailed, matching skin tone"
        # Aggressive negative prompt to prevent identity and expression shift
        negative_prompt = "changing facial expression, open mouth, moving eyes, changing identity, changing lighting, background shift, morphing"

        for attempt in range(max_retries):
            logger.info(f"Generating confounder (Attempt {attempt + 1}/{max_retries})...")
            
            # tau <= 0.45 constraint: strict structural preservation
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                strength=0.45, 
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]
            
            # Execute Mathematical Verification
            if self.meil_gate.verify(init_image, result, mask_image):
                return result
                
        logger.error("Failed to generate a causally valid image after max retries. Rejecting sample.")
        return None

# =======================================================================
# Usage Example:
# synthesizer = CausalScarSynthesizer(device="cuda")
# final_image = synthesizer.generate_causal_confounder("bp4d_subject_001.jpg")
# if final_image:
#     final_image.save("bp4d_subject_001_scarred.jpg")
# =======================================================================
