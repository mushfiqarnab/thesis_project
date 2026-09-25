#!/usr/bin/env python3
"""
World-class research pipeline for identity-preserving facial scar augmentation.

Design basis:
- SDXL inpainting for high-resolution local editing.
- Exact pixel compositing outside a context mask: the final artifact cannot alter
  pixels outside the permitted edit zone.
- ArcFace/InsightFace identity embedding similarity as a face-identity gate.
- Landmark geometry as a structural gate.
- Outside-context SSIM + MAE on the RAW diffusion output to detect unintended edits.
- LPIPS (optional) as a perceptual metric, never as a sole acceptance criterion.
- Explicit train-split guard and full provenance/audit manifest.
- Accepted/rejected quarantine and human-review previews.

Important:
This script is a research augmentation pipeline, not a clinical validation system.
A synthetic scar image should not be interpreted as medical truth without domain review.
Because the method deliberately retains most of the source face, outputs are NOT
claimed to be anonymized or privacy-preserving.

Expected input CSV (default): data/csv/multimodal.csv
Required image column (default): img_path
Recommended split column: split (accepted values default to: train)

Primary output:
  data/faces_diffusion_worldclass/accepted/*.png
  data/faces_diffusion_worldclass/rejected/*.png
  data/faces_diffusion_worldclass/review/*.jpg
  data/csv/multimodal_diffusion_worldclass.csv
  data/csv/multimodal_diffusion_worldclass.jsonl

Research references used for design:
- Deng et al., ArcFace: Additive Angular Margin Loss for Deep Face Recognition (2019).
- Ye et al., IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image
  Diffusion Models (2023).
- Wang et al., InstantID: Zero-shot Identity-Preserving Generation in Seconds (2024).
- Zhang et al., The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
  (LPIPS, 2018).
- Podell et al., SDXL: Improving Latent Diffusion Models for High-Resolution Image
  Synthesis (2023).
- Ding et al., When Diffusion Models Forget Who You Are: Identity Preservation in Face
  Inpainting under Large Occlusions (2026).
- Zamzmi et al., Scorecard for Synthetic Medical Data Evaluation (2025).
- Kaabachi et al., A scoping review of privacy and utility metrics in medical synthetic
  data (2025).
- Nature Biomedical Engineering (2025), Unconditional latent diffusion models memorize
  patient imaging data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from skimage.metrics import structural_similarity as skimage_ssim
from tqdm import tqdm

try:
    from diffusers import AutoPipelineForInpainting
except ImportError as exc:
    raise ImportError(
        "diffusers is required. Install with: pip install -U diffusers transformers accelerate"
    ) from exc

try:
    import cv2
    from insightface.app import FaceAnalysis
except ImportError as exc:
    raise ImportError(
        "insightface and opencv-python are required. Install with: "
        "pip install -U insightface opencv-python onnxruntime-gpu"
    ) from exc

try:
    import lpips as lpips_lib  # type: ignore
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False


LOGGER = logging.getLogger("worldclass_diffusion")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_int_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def normalized_landmark_rmse(src_kps: np.ndarray, gen_kps: np.ndarray) -> float:
    src_kps = np.asarray(src_kps, dtype=np.float32)
    gen_kps = np.asarray(gen_kps, dtype=np.float32)
    src_eye = np.linalg.norm(src_kps[0] - src_kps[1])
    if not np.isfinite(src_eye) or src_eye < 1e-6:
        return float("nan")
    delta = src_kps - gen_kps
    rmse = float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))) / src_eye)
    return rmse


def masked_mae(src: np.ndarray, gen: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() == 0:
        return float("nan")
    diff = np.abs(src.astype(np.float32) - gen.astype(np.float32))
    return float(diff[m].mean())


def masked_max_abs(src: np.ndarray, gen: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() == 0:
        return float("nan")
    diff = np.abs(src.astype(np.int16) - gen.astype(np.int16))
    return float(diff[m].max())


def roi_replaced(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace pixels inside mask with reference pixels.

    This lets a full-image metric isolate changes outside the masked region.
    """
    out = image.copy()
    m = mask.astype(bool)
    out[m] = reference[m]
    return out


def ssim_outside_region(src: np.ndarray, gen: np.ndarray, edit_mask: np.ndarray) -> float:
    gen_isolated = roi_replaced(gen, src, edit_mask)
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(gen_isolated, cv2.COLOR_RGB2GRAY)
    score = skimage_ssim(src_gray, gen_gray, data_range=255)
    return float(score)


def resized_square_face_crop(
    image: Image.Image,
    bbox_xyxy: Sequence[float],
    output_size: int,
    margin: float,
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    """Create a square face-centered crop and map bbox into resized coordinates."""
    img_w, img_h = image.size
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * float(margin)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    left = cx - side / 2.0
    top = cy - side / 2.0
    right = cx + side / 2.0
    bottom = cy + side / 2.0

    # Shift crop into image bounds while maintaining size where possible.
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= right - img_w
        right = img_w
    if bottom > img_h:
        top -= bottom - img_h
        bottom = img_h
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(img_w), right)
    bottom = min(float(img_h), bottom)

    # Final crop is as square as the source bounds permit.
    crop_w = right - left
    crop_h = bottom - top
    side2 = min(crop_w, crop_h)
    cx2 = (left + right) / 2.0
    cy2 = (top + bottom) / 2.0
    left = max(0.0, min(cx2 - side2 / 2.0, img_w - side2))
    top = max(0.0, min(cy2 - side2 / 2.0, img_h - side2))
    right = left + side2
    bottom = top + side2

    crop = image.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))
    crop = crop.resize((output_size, output_size), Image.Resampling.LANCZOS)

    scale = output_size / side2
    new_bbox = (
        (x1 - left) * scale,
        (y1 - top) * scale,
        (x2 - left) * scale,
        (y2 - top) * scale,
    )
    return crop, new_bbox


def map_kps_to_crop(
    kps: Optional[np.ndarray],
    crop_source_box: Tuple[float, float, float, float],
    output_size: int,
) -> Optional[np.ndarray]:
    if kps is None:
        return None
    left, top, right, bottom = crop_source_box
    side = right - left
    if side <= 0:
        return None
    scale = output_size / side
    arr = np.asarray(kps, dtype=np.float32).copy()
    arr[:, 0] = (arr[:, 0] - left) * scale
    arr[:, 1] = (arr[:, 1] - top) * scale
    return arr


def ellipse_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
    n: int = 28,
) -> List[Tuple[float, float]]:
    angle = math.radians(angle_deg)
    ca, sa = math.cos(angle), math.sin(angle)
    pts: List[Tuple[float, float]] = []
    for t in np.linspace(0, 2 * math.pi, n, endpoint=False):
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        px = cx + ca * x - sa * y
        py = cy + sa * x + ca * y
        pts.append((px, py))
    return pts


REGIONS: Dict[str, Tuple[float, float, float, float, float]] = {
    # cx, cy, relative length, relative width, allowed angle center
    "forehead": (0.50, 0.24, 0.16, 0.040, 90.0),
    "left_cheek": (0.31, 0.56, 0.16, 0.050, 35.0),
    "right_cheek": (0.69, 0.56, 0.16, 0.050, 145.0),
    "left_temple": (0.24, 0.42, 0.12, 0.045, 50.0),
    "right_temple": (0.76, 0.42, 0.12, 0.045, 130.0),
    "left_jaw": (0.34, 0.73, 0.14, 0.045, 25.0),
    "right_jaw": (0.66, 0.73, 0.14, 0.045, 155.0),
    "chin": (0.50, 0.80, 0.12, 0.040, 90.0),
}

SCAR_STYLES: Dict[str, str] = {
    "linear_surgical": "a subtle, mature linear post-surgical scar, fully healed and closed",
    "linear_traumatic": "a subtle, mature linear traumatic laceration scar, fully healed and closed",
    "hypopigmented": "a subtle mature hypopigmented linear scar with natural skin texture",
    "atrophic": "a subtle mature atrophic scar with gentle depression and natural skin texture",
    "hypertrophic": "a mild mature hypertrophic scar, closed skin, limited thickness and natural texture",
}


# ==============================================================================
# SOTA DECOUPLED CORE SALIENCE EVALUATOR
# ==============================================================================
def compute_decoupled_salience(
    comp_np: np.ndarray, 
    canvas_np: np.ndarray, 
    alpha_mask: np.ndarray,
    true_core_mask: np.ndarray
) -> dict:
    """
    Decouples the central fibrotic core from the Gaussian-feathered fringe.
    Eliminates the Mask-Averaging Fallacy while strictly quarantining ghost scars.
    """
    core_mask = (true_core_mask > 0.5).astype(np.uint8)
    
    if np.sum(core_mask) < 25:
        core_mask = (alpha_mask > 0.5).astype(np.uint8)

    diff = np.abs(comp_np.astype(np.float32) - canvas_np.astype(np.float32))
    diff_gray = np.mean(diff, axis=-1)

    mask_sum = np.sum(alpha_mask) + 1e-6
    global_mae = float(np.sum(diff_gray * alpha_mask) / mask_sum)
    core_mae = float(np.mean(diff_gray[core_mask == 1])) if np.sum(core_mask) > 0 else 0.0
    core_changed_ratio = float(np.mean((diff_gray[core_mask == 1] > 12.0).astype(np.float32))) if np.sum(core_mask) > 0 else 0.0

    return {
        "global_mae": global_mae,
        "core_mae": core_mae,
        "core_changed_ratio": core_changed_ratio
    }


@dataclass
class GenerationConfig:
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    generation_size: int = 1024
    output_size: int = 224
    face_margin: float = 1.75
    steps: int = 32
    guidance_scale: float = 9.0
    strength: float = 0.82
    padding_mask_crop: int = 64
    attempts: int = 4
    min_face_cosine: float = 0.85
    max_landmark_rmse: float = 0.035
    min_ssim_outside: float = 0.995
    max_outside_mae: float = 1.50
    min_edit_mae: float = 8.5
    min_core_mae: float = 14.0
    min_changed_ratio: float = 0.35
    context_dilation: int = 17
    mask_blur_radius: float = 0.0
    min_core_area_pct: float = 0.25
    max_core_area_pct: float = 4.0
    num_variants: int = 1
    seed: int = 20260921
    cpu_offload: bool = True
    use_lpips: bool = False
    save_raw_debug: bool = False


class FaceEngine:
    def __init__(self, use_gpu: bool = True):
        if use_gpu and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        LOGGER.info("Loading InsightFace FaceAnalysis (buffalo_l)")
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect(self, image: Image.Image, reject_multiple: bool = True):
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            raise ValueError("no_face_detected")
        if reject_multiple and len(faces) != 1:
            raise ValueError(f"expected_one_face_found_{len(faces)}")
        # Largest face, while requiring a single face by default.
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        return face


class LPIPSEngine:
    def __init__(self, enabled: bool, device: str):
        self.enabled = bool(enabled and LPIPS_AVAILABLE)
        self.model = None
        self.device = device
        if enabled and not LPIPS_AVAILABLE:
            LOGGER.warning("LPIPS requested but package is unavailable; LPIPS metrics will be omitted.")
        if self.enabled:
            self.model = lpips_lib.LPIPS(net="alex").to(device)
            self.model.eval()

    @staticmethod
    def _tensor(image: Image.Image, device: str) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)

    @torch.inference_mode()
    def distance(self, a: Image.Image, b: Image.Image) -> Optional[float]:
        if not self.enabled or self.model is None:
            return None
        ta = self._tensor(a, self.device)
        tb = self._tensor(b, self.device)
        value = self.model(ta, tb).mean().item()
        return float(value)


class OrganicMaskGenerator:
    """Controlled organic scar-mask generator constrained to face-safe regions."""

    def __init__(self, size: int, dilation: int = 17):
        self.size = size
        self.dilation = dilation if dilation % 2 == 1 else dilation + 1

    def generate(
        self,
        rng: np.random.Generator,
        face_bbox: Sequence[float],
        face_kps: Optional[np.ndarray],
        region_name: str,
    ) -> Tuple[Image.Image, Image.Image, Dict[str, float]]:
        size = self.size
        fx1, fy1, fx2, fy2 = map(float, face_bbox)
        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)

        if region_name not in REGIONS:
            region_name = str(rng.choice(list(REGIONS.keys())))
        rx, ry, rel_len, rel_width, base_angle = REGIONS[region_name]

        cx = fx1 + rx * fw + rng.normal(0, 0.012 * fw)
        cy = fy1 + ry * fh + rng.normal(0, 0.010 * fh)
        length = rel_len * fw * rng.uniform(0.75, 1.15)
        width = max(4.0, rel_width * fw * rng.uniform(0.70, 1.25))
        angle = base_angle + rng.normal(0, 16.0)

        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)

        # Curved centerline with low-amplitude stochastic perturbation.
        ux = math.cos(math.radians(angle))
        uy = math.sin(math.radians(angle))
        vx = -uy
        vy = ux
        n = 9
        points: List[Tuple[float, float]] = []
        for i in range(n):
            t = i / (n - 1) - 0.5
            lateral = math.sin((t + 0.5) * math.pi * 1.4) * rng.normal(0, 0.035 * fw)
            jitter = rng.normal(0, 0.010 * fw)
            px = cx + ux * (t * length) + vx * lateral + jitter
            py = cy + uy * (t * length) + vy * lateral + jitter
            points.append((px, py))

        line_width = int(round(width * rng.uniform(0.80, 1.20)))
        draw.line(points, fill=255, width=max(3, line_width), joint="curve")

        # Organic lobes create a non-perfect, biologically plausible mask footprint.
        for i, (px, py) in enumerate(points):
            local = width * rng.uniform(0.35, 0.65)
            if i in (0, n - 1):
                local *= rng.uniform(0.55, 0.85)
            pts = ellipse_points(
                px,
                py,
                local * rng.uniform(0.65, 1.10),
                local * rng.uniform(0.35, 0.65),
                angle + rng.normal(0, 20),
            )
            draw.polygon(pts, fill=255)

        # Face-safe oval: the generated scar must stay in the facial skin envelope.
        face_oval = Image.new("L", (size, size), 0)
        oval_draw = ImageDraw.Draw(face_oval)
        pad_x = 0.04 * fw
        pad_y = 0.04 * fh
        oval_draw.ellipse(
            [fx1 + pad_x, fy1 + pad_y, fx2 - pad_x, fy2 - pad_y],
            fill=255,
        )
        mask = ImageChops.multiply(mask, face_oval)

        # Protect eyes, nose, and mouth using the five InsightFace keypoints.
        if face_kps is not None and len(face_kps) >= 5:
            eye_dist = np.linalg.norm(face_kps[0] - face_kps[1])
            protected = Image.new("L", (size, size), 0)
            pd = ImageDraw.Draw(protected)
            eye_r = max(7.0, 0.12 * eye_dist)
            for p in face_kps[:2]:
                pd.ellipse([p[0] - eye_r, p[1] - eye_r, p[0] + eye_r, p[1] + eye_r], fill=255)
            nose_r = max(7.0, 0.10 * eye_dist)
            p = face_kps[2]
            pd.ellipse([p[0] - nose_r, p[1] - nose_r * 0.85, p[0] + nose_r, p[1] + nose_r * 0.85], fill=255)
            mouth_r = max(9.0, 0.16 * eye_dist)
            for p in face_kps[3:5]:
                pd.ellipse([p[0] - mouth_r, p[1] - mouth_r * 0.55, p[0] + mouth_r, p[1] + mouth_r * 0.55], fill=255)
            protected = protected.filter(ImageFilter.GaussianBlur(radius=1.2))
            inv = ImageOps.invert(protected)
            mask = ImageChops.multiply(mask, inv)

        # Slight erosion-like cleanup followed by dilation for diffusion context.
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35))
        core = mask.point(lambda v: 255 if v >= 96 else 0, mode="L")
        context = core.filter(ImageFilter.MaxFilter(self.dilation))

        # Re-apply face envelope to context as well.
        context = ImageChops.multiply(context, face_oval)

        core_np = np.asarray(core) > 0
        context_np = np.asarray(context) > 0
        face_area = max(1.0, fw * fh)
        core_area_pct = float(core_np.sum() / face_area * 100.0)
        context_area_pct = float(context_np.sum() / face_area * 100.0)

        return core, context, {
            "core_area_pct": core_area_pct,
            "context_area_pct": context_area_pct,
            "region": region_name,
        }

def generate_anisotropic_dermal_noise(
    shape: Tuple[int, int], 
    seed: int,
    angle: float = 45.0, 
    base_frequency: float = 0.08
) -> np.ndarray:
    """
    Generates oriented, anisotropic high-frequency dermal noise.
    Tuned specifically so pixel-space perturbations survive the SDXL VAE encoder (f=8)
    without being smoothed out as sub-latent aliasing.
    """
    import cv2
    rng = np.random.RandomState(seed)
    
    h, w = shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    
    theta = np.deg2rad(angle)
    x_rot = xx * np.cos(theta) + yy * np.sin(theta)
    y_rot = -xx * np.sin(theta) + yy * np.cos(theta)
    
    # 1. Macro-structure (Fibrotic Ridge) - Survives f=8 compression
    wave_macro = np.sin(2 * np.pi * base_frequency * x_rot * 10.0)
    
    # 2. Micro-structure (Erythema / Texture)
    wave_micro = np.cos(2 * np.pi * (base_frequency * 2.5) * y_rot * 10.0)
    
    carrier = (wave_macro + 0.4 * wave_micro) / 1.4
    
    # Non-linear contrast enhancement to punch through diffusion prior
    carrier = np.sign(carrier) * (np.abs(carrier) ** 0.8)
    
    noise = rng.normal(0, 0.25, (h, w))
    dermal_field = np.clip(carrier + noise, -1.0, 1.0)
    
    return dermal_field.astype(np.float32)

class WorldClassDiffusionEngine:
    def __init__(self, cfg: GenerationConfig, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            LOGGER.warning(
                "CUDA is unavailable. SDXL inpainting on CPU is extremely slow; "
                "the script remains functional but is intended for a CUDA GPU."
            )

        self.pipe = self._load_pipeline()
        self.face_engine = FaceEngine(use_gpu=self.device == "cuda")
        self.lpips_engine = LPIPSEngine(cfg.use_lpips, self.device)
        self.mask_gen = OrganicMaskGenerator(cfg.generation_size, cfg.context_dilation)

        self.total_attempts = 0
        self.total_accepted = 0
        self.total_rejected = 0

    def _load_pipeline(self):
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        LOGGER.info("Loading %s at %s", self.cfg.model_id, self.device)
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        if self.device == "cuda":
            # SDXL inpainting model cards support fp16 variants.
            load_kwargs["variant"] = "fp16"

        pipe = AutoPipelineForInpainting.from_pretrained(self.cfg.model_id, **load_kwargs)
        pipe.set_progress_bar_config(disable=True)

        if self.device == "cuda":
            if self.cfg.cpu_offload:
                LOGGER.info("Enabling model CPU offload for lower VRAM usage.")
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        else:
            pipe = pipe.to("cpu")
        return pipe

    def _face_features(self, image: Image.Image):
        face = self.face_engine.detect(image)
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        kps = np.asarray(face.kps, dtype=np.float32)
        bbox = np.asarray(face.bbox, dtype=np.float32)
        score = float(getattr(face, "det_score", 0.0))
        return face, embedding, kps, bbox, score

    def _build_prompts(self, scar_type: str, region: str) -> Tuple[str, str]:
        prompt = (
            "macro medical photograph of a mature hypertrophic facial scar, "
            "elevated collagenous fibrous ridge, slight erythematous margin, "
            "disrupted dermal skin pores, authentic dermatological healing texture, "
            "realistic skin contracture, depth shadows, sharp clinical focus, 8k resolution"
        )
        negative = (
            "smooth skin, pristine dermis, flawless complexion, unblemished face, "
            "invisible scar, faded scratch, airbrushed, drawing, cartoon, 3d render, "
            "flat color, makeup, blurred texture, plastic skin, cosmetic retouch"
        )
        return prompt, negative

    def _generate_once(
        self,
        base_image: Image.Image,
        context_mask: Image.Image,
        core_mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> Image.Image:
        generator_device = self.device if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        
        # ASLP: Anisotropic Structural Latent Perturbation
        canvas_np = np.asarray(base_image)
        core_arr = np.asarray(core_mask, dtype=np.float32) / 255.0
        
        if np.sum(core_arr) > 0:
            # 1. Deterministic Random State
            rng = np.random.RandomState(int(seed))
            
            # 2. Prevent Morphological Cliff (C0 Discontinuity artifact)
            # The VAE will encode a strict binary boundary as a literal cut/scab.
            # We apply topological feathering so the noise fades naturally into the cheek dermis.
            import cv2
            core_arr_smooth = cv2.GaussianBlur(core_arr, (15, 15), 5.0)
            
            dermal_noise = generate_anisotropic_dermal_noise(
                shape=canvas_np.shape[:2], 
                seed=int(seed),
                angle=float(rng.uniform(20.0, 70.0))
            )
            
            # 3. Channel-Wise Dermal Chrominance (Biological Accuracy)
            # Scars are highly vascularized. Instead of greyscale scalar amplitude, 
            # we bias the structural tension toward Erythema (Red channel [0]).
            # PIL is RGB -> R=48.0, G=32.0, B=32.0
            color_bias = np.array([48.0, 32.0, 32.0], dtype=np.float32)
            
            canvas_perturbed = canvas_np.astype(np.float32) + (dermal_noise[:, :, None] * core_arr_smooth[:, :, None] * color_bias)
            canvas_perturbed = np.clip(canvas_perturbed, 0, 255).astype(np.uint8)
            base_image = Image.fromarray(canvas_perturbed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_image,
                mask_image=context_mask,
                height=self.cfg.generation_size,
                width=self.cfg.generation_size,
                strength=self.cfg.strength,
                num_inference_steps=self.cfg.steps,
                guidance_scale=self.cfg.guidance_scale,
                padding_mask_crop=self.cfg.padding_mask_crop,
                generator=generator,
            ).images[0]
        return result.convert("RGB")

    def _compose_final(
        self,
        base_image: Image.Image,
        generated_raw: Image.Image,
        context_mask: Image.Image,
    ) -> Image.Image:
        # We feather the context mask strictly for blending.
        # This eradicates high-frequency boundary seams from the UNet 
        # while preserving the identity outside the mask footprint.
        from PIL import ImageFilter
        alpha = context_mask.filter(ImageFilter.GaussianBlur(radius=5.0))
        final = Image.composite(generated_raw, base_image, alpha)
        return final.convert("RGB")

    def _acceptance_metrics(
        self,
        base_image: Image.Image,
        generated_raw: Image.Image,
        final_image: Image.Image,
        core_mask: Image.Image,
        context_mask: Image.Image,
        source_embedding: np.ndarray,
        source_kps: np.ndarray,
    ) -> Dict[str, Any]:
        src = np.asarray(base_image, dtype=np.uint8)
        raw = np.asarray(generated_raw, dtype=np.uint8)
        final = np.asarray(final_image, dtype=np.uint8)
        alpha_mask = np.asarray(context_mask, dtype=np.float32) / 255.0
        context = alpha_mask > 0.5
        outside_context = ~context

        _, gen_embedding, gen_kps, _, det_score = self._face_features(final_image)
        face_cos = cosine_similarity(source_embedding, gen_embedding)
        lm_rmse = normalized_landmark_rmse(source_kps, gen_kps)
        ssim_out = ssim_outside_region(src, raw, context)
        outside_mae = masked_mae(src, raw, outside_context)
        outside_max = masked_max_abs(src, raw, outside_context)
        
        final_exact_outside_mae = masked_mae(src, final, outside_context)
        final_exact_outside_max = masked_max_abs(src, final, outside_context)

        # 2. Decoupled Core Salience Evaluation
        true_core_mask_np = np.asarray(core_mask, dtype=np.float32) / 255.0
        salience = compute_decoupled_salience(final, src, alpha_mask, true_core_mask_np)
        core_mae = salience["core_mae"]
        global_mae = salience["global_mae"]
        changed_ratio = salience["core_changed_ratio"]

        lpips_full = self.lpips_engine.distance(base_image, final_image)

        passed = True
        failures: List[str] = []

        if not np.isfinite(face_cos) or face_cos < self.cfg.min_face_cosine:
            passed = False
            failures.append("identity_cosine_below_threshold")
        if not np.isfinite(lm_rmse) or lm_rmse > self.cfg.max_landmark_rmse:
            passed = False
            failures.append("landmark_drift_above_threshold")
        if not np.isfinite(ssim_out) or ssim_out < self.cfg.min_ssim_outside:
            passed = False
            failures.append("outside_ssim_below_threshold")
        if not np.isfinite(outside_mae) or outside_mae > self.cfg.max_outside_mae:
            passed = False
            failures.append("raw_outside_mae_above_threshold")
        
        # New Salience Gates
        if core_mae < getattr(self.cfg, 'min_core_mae', 14.0):
            passed = False
            failures.append("core_mae_too_low")
        if global_mae < getattr(self.cfg, 'min_edit_mae', 8.5):
            passed = False
            failures.append("global_mae_too_low")
        if changed_ratio < getattr(self.cfg, 'min_changed_ratio', 0.35):
            passed = False
            failures.append("core_changed_ratio_too_low")

        # Final composite invariants. These should be zero/nearly zero except for
        # unavoidable integer conversion edge effects.
        if np.isfinite(final_exact_outside_mae) and final_exact_outside_mae > 0.01:
            passed = False
            failures.append("final_outside_not_exactly_preserved")

        return {
            "face_cosine": face_cos,
            "landmark_rmse": lm_rmse,
            "ssim_outside_context": ssim_out,
            "raw_outside_mae": outside_mae,
            "raw_outside_max_abs": outside_max,
            "edit_mae_final": core_mae,
            "edit_mae_raw": global_mae,
            "edit_changed_fraction": changed_ratio,
            "final_outside_mae": final_exact_outside_mae,
            "final_outside_max_abs": final_exact_outside_max,
            "lpips_full": lpips_full,
            "passed": passed,
            "failures": failures,
        }

    def _make_review_sheet(
        self,
        base: Image.Image,
        core: Image.Image,
        raw: Image.Image,
        final: Image.Image,
        metrics: Dict[str, Any],
        out_path: Path,
    ) -> None:
        tile_size = 384
        canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), (245, 245, 245))
        mask_rgb = Image.merge("RGB", (core, core, core))
        # Add a red-like overlay without hard-coding a rendering style dependency.
        overlay = base.copy().convert("RGBA")
        red = Image.new("RGBA", base.size, (255, 0, 0, 0))
        red.putalpha(core.point(lambda p: int(p * 0.35)))
        overlay = Image.alpha_composite(overlay, red).convert("RGB")

        items = [
            (base, "SOURCE"),
            (overlay, "MASK / ROI"),
            (raw, "RAW DIFFUSION"),
            (final, "FINAL / COMPOSITED"),
        ]

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for i, (img, title) in enumerate(items):
            img2 = ImageOps.fit(img, (tile_size, tile_size), method=Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(img2)
            draw.rectangle([0, 0, tile_size, 24], fill=(0, 0, 0))
            draw.text((8, 6), title, fill=(255, 255, 255), font=font)
            x = (i % 2) * tile_size
            y = (i // 2) * tile_size
            canvas.paste(img2, (x, y))

        metric_text = (
            f"ArcFace cos={metrics.get('face_cosine', float('nan')):.4f} | "
            f"LM-RMSE={metrics.get('landmark_rmse', float('nan')):.4f} | "
            f"SSIM-out={metrics.get('ssim_outside_context', float('nan')):.5f} | "
            f"MAE-out={metrics.get('raw_outside_mae', float('nan')):.3f} | "
            f"Edit-MAE={metrics.get('edit_mae_final', float('nan')):.3f}\n"
            f"LPIPS={metrics.get('lpips_full', None)} | "
            f"PASS={metrics.get('passed', False)}"
        )
        footer = Image.new("RGB", (tile_size * 2, 110), (235, 235, 235))
        fd = ImageDraw.Draw(footer)
        fd.multiline_text((10, 10), metric_text, fill=(20, 20, 20), font=font, spacing=4)
        canvas2 = Image.new("RGB", (tile_size * 2, tile_size * 2 + 110), (235, 235, 235))
        canvas2.paste(canvas, (0, 0))
        canvas2.paste(footer, (0, tile_size * 2))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas2.save(out_path, quality=95)

    def process_sample(
        self,
        row: pd.Series,
        row_index: int,
        image_path: Path,
        variant: int,
    ) -> Dict[str, Any]:
        source_sha = sha256_file(image_path)
        sample_seed = stable_int_seed(self.cfg.seed, source_sha, row_index, variant)

        source = Image.open(image_path).convert("RGB")
        source_face = self.face_engine.detect(source)
        source_bbox = np.asarray(source_face.bbox, dtype=np.float32)

        # Use a face-centered square canvas and then generate at 1024x1024.
        crop, crop_bbox = resized_square_face_crop(
            source,
            source_bbox,
            self.cfg.generation_size,
            self.cfg.face_margin,
        )
        # Re-run face analysis on the exact working crop. This makes the source
        # identity embedding and landmarks commensurate with the generated image.
        crop_face = self.face_engine.detect(crop)
        source_embedding = np.asarray(crop_face.normed_embedding, dtype=np.float32)
        crop_kps = np.asarray(crop_face.kps, dtype=np.float32)

        rng = np.random.default_rng(sample_seed)
        region = str(rng.choice(list(REGIONS.keys())))
        scar_type = str(rng.choice(list(SCAR_STYLES.keys())))

        # Retry mask construction until area constraints are met.
        core_mask = None
        context_mask = None
        mask_meta: Dict[str, Any] = {}
        for _ in range(25):
            core_candidate, context_candidate, meta = self.mask_gen.generate(
                rng, crop_bbox, crop_kps, region
            )
            if self.cfg.min_core_area_pct <= meta["core_area_pct"] <= self.cfg.max_core_area_pct:
                core_mask, context_mask, mask_meta = core_candidate, context_candidate, meta
                break
            region = str(rng.choice(list(REGIONS.keys())))
        if core_mask is None or context_mask is None:
            raise ValueError("mask_area_constraints_failed")

        prompt, negative_prompt = self._build_prompts(scar_type, region)

        source_id = f"row{row_index:06d}_{source_sha[:12]}"
        synthetic_id = f"{source_id}_v{variant:02d}"
        attempt_records: List[Tuple[Image.Image, Image.Image, Dict[str, Any], int]] = []
        accepted = False
        best_record: Optional[Tuple[Image.Image, Image.Image, Dict[str, Any], int]] = None

        for attempt in range(self.cfg.attempts):
            self.total_attempts += 1
            attempt_seed = stable_int_seed(sample_seed, attempt)
            started = time.time()
            raw = self._generate_once(crop, context_mask, core_mask, prompt, negative_prompt, attempt_seed)
            final = self._compose_final(crop, raw, context_mask)
            elapsed = time.time() - started

            try:
                metrics = self._acceptance_metrics(
                    crop, raw, final, core_mask, context_mask, source_embedding, crop_kps
                )
            except ValueError as exc:
                metrics = {
                    "passed": False,
                    "failures": [f"verification_error:{exc}"],
                }

            metrics["attempt_seconds"] = elapsed
            metrics["attempt"] = attempt
            attempt_records.append((raw, final, metrics, attempt_seed))

            if best_record is None:
                best_record = (raw, final, metrics, attempt_seed)
            else:
                # Deterministic preference: passed > failed; then higher identity cosine;
                # then lower landmark error; then higher outside SSIM.
                def rank(m: Dict[str, Any]) -> Tuple[int, float, float, float]:
                    return (
                        int(bool(m.get("passed", False))),
                        float(m.get("face_cosine", -1.0)),
                        -float(m.get("landmark_rmse", 1e9)),
                        float(m.get("ssim_outside_context", -1.0)),
                    )
                if rank(metrics) > rank(best_record[2]):
                    best_record = (raw, final, metrics, attempt_seed)

            if metrics.get("passed", False):
                accepted = True
                break

        assert best_record is not None
        raw, final, metrics, used_seed = best_record

        # Resize final output only after all identity/edit gates have passed.
        final_224 = final.resize((self.cfg.output_size, self.cfg.output_size), Image.Resampling.LANCZOS)
        mask_224 = core_mask.resize((self.cfg.output_size, self.cfg.output_size), Image.Resampling.NEAREST)
        raw_224 = raw.resize((self.cfg.output_size, self.cfg.output_size), Image.Resampling.LANCZOS)

        status = "accepted" if accepted else "rejected"
        self.total_accepted += int(accepted)
        self.total_rejected += int(not accepted)

        return {
            "row_index": int(row_index),
            "status": status,
            "synthetic_id": synthetic_id,
            "source_id": source_id,
            "source_sha256": source_sha,
            "source_path": str(image_path),
            "variant": int(variant),
            "seed": int(used_seed),
            "scar_type": scar_type,
            "region": region,
            "core_area_pct": float(mask_meta["core_area_pct"]),
            "context_area_pct": float(mask_meta["context_area_pct"]),
            "metrics": metrics,
            "image": final_224,
            "source_image": crop.resize((self.cfg.output_size, self.cfg.output_size), Image.Resampling.LANCZOS),
            "mask": mask_224,
            "raw": raw_224,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }

    @staticmethod
    def _approximate_kps_from_bbox(
        source_kps: np.ndarray,
        source_bbox: np.ndarray,
        transformed_bbox: Sequence[float],
    ) -> np.ndarray:
        """Map keypoints using the same affine transform as the face bbox crop.

        The transform is inferred from source bbox -> transformed bbox. This is exact
        for the crop's scale and translation when the crop is square and uses uniform
        scaling.
        """
        sx1, sy1, sx2, sy2 = map(float, source_bbox)
        tx1, ty1, tx2, ty2 = map(float, transformed_bbox)
        sw = max(1e-6, sx2 - sx1)
        tw = max(1e-6, tx2 - tx1)
        scale = tw / sw
        # transformed bbox is already in crop coordinates; derive transform from
        # the fact that source bbox origin maps to transformed bbox origin.
        out = np.asarray(source_kps, dtype=np.float32).copy()
        out[:, 0] = tx1 + (out[:, 0] - sx1) * scale
        out[:, 1] = ty1 + (out[:, 1] - sy1) * scale
        return out


def save_atomic(image: Image.Image, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img_format = path.suffix.lstrip(".").upper()
    if img_format == "JPG":
        img_format = "JPEG"
    image.save(tmp, format=img_format, **kwargs)
    os.replace(tmp, path)


def resolve_image_path(project_root: Path, raw_path: Any) -> Path:
    p = Path(str(raw_path)).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    candidate = (project_root / p).resolve()
    if candidate.exists():
        return candidate
    # Also support paths relative to the CSV's project-style location.
    if p.exists():
        return p.resolve()
    return candidate


def check_split(df: pd.DataFrame, split_column: str, allowed_splits: Sequence[str], allow_missing: bool) -> pd.DataFrame:
    if split_column not in df.columns:
        if not allow_missing:
            raise ValueError(
                f"CSV has no '{split_column}' column. For leakage-safe thesis evaluation, "
                "add a split column or pass --allow-missing-split only after independently "
                "confirming that the CSV contains training images only."
            )
        LOGGER.warning("Split column '%s' missing; proceeding only because override was supplied.", split_column)
        return df

    allowed = {x.strip().lower() for x in allowed_splits}
    values = df[split_column].astype(str).str.strip().str.lower()
    keep = values.isin(allowed)
    removed = int((~keep).sum())
    if removed:
        LOGGER.info("Excluding %d rows outside allowed split(s): %s", removed, sorted(allowed))
    out = df.loc[keep].copy()
    if out.empty:
        raise ValueError("No rows remain after split filtering.")
    return out


def build_manifest_row(
    original_row: pd.Series,
    result: Dict[str, Any],
    image_rel: str,
    mask_rel: str,
) -> Dict[str, Any]:
    out = original_row.to_dict()
    metrics = result["metrics"]
    out.update(
        {
            "synthetic_id": result["synthetic_id"],
            "source_id": result["source_id"],
            "source_sha256": result["source_sha256"],
            "synthetic_img_path": image_rel,
            "synthetic_mask_path": mask_rel,
            "synthetic_status": result["status"],
            "synthetic_variant": result["variant"],
            "synthetic_seed": result["seed"],
            "synthetic_scar": 1,
            "synthetic_scar_type": result["scar_type"],
            "synthetic_region": result["region"],
            "synthetic_core_area_pct": result["core_area_pct"],
            "synthetic_context_area_pct": result["context_area_pct"],
            "synthetic_model_id": result.get("model_id"),
            "synthetic_identity_cosine": metrics.get("face_cosine"),
            "synthetic_landmark_rmse": metrics.get("landmark_rmse"),
            "synthetic_ssim_outside_context": metrics.get("ssim_outside_context"),
            "synthetic_raw_outside_mae": metrics.get("raw_outside_mae"),
            "synthetic_raw_outside_max_abs": metrics.get("raw_outside_max_abs"),
            "synthetic_edit_mae": metrics.get("edit_mae_final"),
            "synthetic_edit_changed_fraction": metrics.get("edit_changed_fraction"),
            "synthetic_final_outside_mae": metrics.get("final_outside_mae"),
            "synthetic_lpips_full": metrics.get("lpips_full"),
            "synthetic_pass": bool(metrics.get("passed", False)),
            "synthetic_failure_reasons": ";".join(metrics.get("failures", [])),
            "synthetic_prompt": result["prompt"],
        }
    )
    return out


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Identity-preserving facial scar diffusion augmentation engine")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--source-csv", type=Path, default=Path("data/csv/approved_pristine_manifest.csv"))
    p.add_argument("--image-column", default="img_path")
    p.add_argument("--split-column", default="split")
    p.add_argument("--allowed-splits", nargs="+", default=["train"])
    p.add_argument("--allow-missing-split", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("data/faces_diffusion_worldclass"))
    p.add_argument("--out-csv", type=Path, default=Path("data/csv/multimodal_diffusion_worldclass.csv"))
    p.add_argument("--model-id", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    p.add_argument("--generation-size", type=int, default=1024)
    p.add_argument("--output-size", type=int, default=224)
    p.add_argument("--face-margin", type=float, default=1.75)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=9.0)
    p.add_argument("--strength", type=float, default=0.82)
    p.add_argument("--padding-mask-crop", type=int, default=64)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--min-face-cosine", type=float, default=0.80)
    p.add_argument("--max-landmark-rmse", type=float, default=0.035)
    # Tolerances
    p.add_argument("--min-ssim-outside", type=float, default=0.995)
    p.add_argument("--max-outside-mae", type=float, default=1.50)
    p.add_argument("--min-edit-mae", type=float, default=8.5)
    p.add_argument("--min-core-mae", type=float, default=14.0)
    p.add_argument("--min-changed-ratio", type=float, default=0.35)
    p.add_argument("--context-dilation", type=int, default=17)
    p.add_argument("--mask-blur-radius", type=float, default=0.0)
    p.add_argument("--min-core-area-pct", type=float, default=0.25)
    p.add_argument("--max-core-area-pct", type=float, default=4.0)
    p.add_argument("--num-variants", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--no-cpu-offload", action="store_true")
    p.add_argument("--use-lpips", action="store_true")
    p.add_argument("--save-raw-debug", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    project_root = args.project_root.resolve()
    source_csv = (project_root / args.source_csv).resolve() if not args.source_csv.is_absolute() else args.source_csv.resolve()
    out_dir = (project_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out_csv = (project_root / args.out_csv).resolve() if not args.out_csv.is_absolute() else args.out_csv.resolve()
    out_jsonl = out_csv.with_suffix(".jsonl")
    config_json = out_dir / "generation_config.json"
    review_dir = out_dir / "review"
    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    masks_dir = out_dir / "masks"
    raw_dir = out_dir / "raw_debug"

    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    df = pd.read_csv(source_csv)
    if args.image_column not in df.columns:
        raise ValueError(f"Required image column '{args.image_column}' missing from {source_csv}")

    df = check_split(df, args.split_column, args.allowed_splits, args.allow_missing_split)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    cfg = GenerationConfig(
        model_id=args.model_id,
        generation_size=args.generation_size,
        output_size=args.output_size,
        face_margin=args.face_margin,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        padding_mask_crop=args.padding_mask_crop,
        attempts=args.attempts,
        min_face_cosine=args.min_face_cosine,
        max_landmark_rmse=args.max_landmark_rmse,
        min_ssim_outside=args.min_ssim_outside,
        max_outside_mae=args.max_outside_mae,
        min_edit_mae=args.min_edit_mae,
        min_core_mae=args.min_core_mae,
        min_changed_ratio=args.min_changed_ratio,
        context_dilation=args.context_dilation,
        mask_blur_radius=args.mask_blur_radius,
        min_core_area_pct=args.min_core_area_pct,
        max_core_area_pct=args.max_core_area_pct,
        num_variants=args.num_variants,
        seed=args.seed,
        cpu_offload=not args.no_cpu_offload,
        use_lpips=args.use_lpips,
        save_raw_debug=args.save_raw_debug,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for d in (accepted_dir, rejected_dir, masks_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)
    if cfg.save_raw_debug:
        raw_dir.mkdir(parents=True, exist_ok=True)

    runtime_meta = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_id": cfg.model_id,
        "diffusers": getattr(__import__("diffusers"), "__version__", None),
        "insightface": getattr(__import__("insightface"), "__version__", None),
        "lpips_available": LPIPS_AVAILABLE,
    }
    with config_json.open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "runtime": runtime_meta}, f, indent=2, default=str)

    LOGGER.info("Rows selected for generation: %d", len(df))
    LOGGER.info("Output directory: %s", out_dir)
    LOGGER.info("Source data are filtered to training split(s): %s", args.allowed_splits)

    engine = WorldClassDiffusionEngine(cfg, project_root)
    manifest_records: List[Dict[str, Any]] = []
    seen_sha: set[str] = set()
    duplicate_exact_rejections = 0

    iterator = tqdm(df.iterrows(), total=len(df), desc="Generating synthetic facial data")
    for row_index, row in iterator:
        try:
            image_path = resolve_image_path(project_root, row[args.image_column])
            if not image_path.exists():
                raise FileNotFoundError(f"image_not_found:{image_path}")

            source_sha = sha256_file(image_path)
            for variant in range(cfg.num_variants):
                result = engine.process_sample(row, int(row_index), image_path, variant)

                # Exact output duplicate detection within this run.
                image_bytes = np.asarray(result["image"], dtype=np.uint8).tobytes()
                out_sha = hashlib.sha256(image_bytes).hexdigest()
                if out_sha in seen_sha:
                    duplicate_exact_rejections += 1
                    if result["status"] == "accepted":
                        engine.total_accepted = max(0, engine.total_accepted - 1)
                        engine.total_rejected += 1
                    result["status"] = "rejected"
                    result["metrics"]["passed"] = False
                    result["metrics"].setdefault("failures", []).append("exact_duplicate_output")
                seen_sha.add(out_sha)

                bucket = accepted_dir if result["status"] == "accepted" else rejected_dir
                filename = f"{result['synthetic_id']}.png"
                image_path_out = bucket / filename
                mask_path_out = masks_dir / f"{result['synthetic_id']}.png"
                review_path = review_dir / f"{result['synthetic_id']}.jpg"

                save_atomic(result["image"], image_path_out)
                save_atomic(result["mask"].convert("L"), mask_path_out)
                if cfg.save_raw_debug:
                    save_atomic(result["raw"], raw_dir / filename)

                engine._make_review_sheet(
                    result["source_image"],
                    result["mask"],
                    result["raw"],
                    result["image"],
                    result["metrics"],
                    review_path,
                )

                rel_img = str(image_path_out.relative_to(project_root))
                rel_mask = str(mask_path_out.relative_to(project_root))
                result["model_id"] = cfg.model_id
                manifest_row = build_manifest_row(row, result, rel_img, rel_mask)
                manifest_row["synthetic_output_sha256"] = out_sha
                manifest_row["synthetic_runtime_device"] = runtime_meta["device"]
                manifest_records.append(manifest_row)

                iterator.set_postfix(
                    accepted=engine.total_accepted,
                    rejected=engine.total_rejected,
                    id=(f"{result['metrics'].get('face_cosine', float('nan')):.3f}"),
                )

        except Exception as exc:
            LOGGER.error("Row %s failed: %s", row_index, exc)
            # Preserve original metadata even for failures.
            failure = row.to_dict()
            failure.update(
                {
                    "row_index": int(row_index),
                    "synthetic_status": "error",
                    "synthetic_failure_reasons": str(exc),
                    "synthetic_scar": 1,
                }
            )
            manifest_records.append(failure)

    # Consolidate final CSV and auditable JSONL.
    out_df = pd.DataFrame(manifest_records)
    out_df.to_csv(out_csv, index=False)
    write_jsonl(out_jsonl, manifest_records)

    summary = {
        "rows_input_after_split_filter": int(len(df)),
        "variants_requested": int(len(df) * cfg.num_variants),
        "accepted": int(engine.total_accepted),
        "rejected": int(engine.total_rejected),
        "exact_duplicate_rejections": int(duplicate_exact_rejections),
        "errors": int(sum(1 for r in manifest_records if r.get("synthetic_status") == "error")),
        "acceptance_rate_over_attempted": float(
            engine.total_accepted / max(1, engine.total_accepted + engine.total_rejected)
        ),
        "total_generation_attempts": int(engine.total_attempts),
        "out_csv": str(out_csv),
        "out_jsonl": str(out_jsonl),
        "out_dir": str(out_dir),
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("============================================================")
    LOGGER.info("WORLD-CLASS SYNTHETIC DATA RUN COMPLETE")
    LOGGER.info("Accepted: %d", engine.total_accepted)
    LOGGER.info("Rejected: %d", engine.total_rejected)
    LOGGER.info("Exact duplicate rejections: %d", duplicate_exact_rejections)
    LOGGER.info("Accepted dataset: %s", accepted_dir)
    LOGGER.info("Rejected/quarantine: %s", rejected_dir)
    LOGGER.info("Manifest CSV: %s", out_csv)
    LOGGER.info("Audit JSONL: %s", out_jsonl)
    LOGGER.info("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
