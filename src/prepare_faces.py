import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# CONFIG (edit these if needed)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # folder: prethesis2/
SRC_DIR = PROJECT_ROOT / "data" / "src_faces"       # input images (FFHQ/CelebA)
OUT_CLEAN = PROJECT_ROOT / "data" / "faces_clean"
OUT_SCAR = PROJECT_ROOT / "data" / "faces_synth_scar"
OUT_MASK = PROJECT_ROOT / "data" / "scar_masks"
OUT_CSV = PROJECT_ROOT / "data" / "csv" / "faces.csv"

MAX_IMAGES = 1000          # process at most this many images (change later)
OUTPUT_SIZE = 256          # resize output to 256x256 (matches your FFHQ 256)
SCAR_PROB = 0.50           # probability to generate scar version per clean image
SEED = 42                  # reproducibility


def ensure_dirs():
    for d in [SRC_DIR, OUT_CLEAN, OUT_SCAR, OUT_MASK, OUT_CSV.parent]:
        d.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def resize_to_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    # If input is already face-cropped (FFHQ/CelebA aligned), simple resize works.
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)


def draw_scar_mask(h: int, w: int) -> np.ndarray:
    """
    Create a SOFT scar mask (0..255) in eyebrow region.
    Includes hard constraints so scars never become too long or too thick.
    """
    # Constraints (tune once, then forget)
    MAX_AREA_FRAC = 0.0040   # max fraction of image pixels covered by scar (~0.4%)
    MIN_AREA_FRAC = 0.0004   # too small scars are invisible (~0.04%)
    MAX_TRIES = 20

    for _ in range(MAX_TRIES):
        mask = np.zeros((h, w), dtype=np.uint8)

        # Eyebrow region
        y_base = random.randint(int(0.26 * h), int(0.38 * h))

        # Control scar length explicitly (shorter = safer)
        length_frac = random.uniform(0.10, 0.22)   # 10%–22% of width
        length = int(length_frac * w)

        x_center = random.randint(int(0.40 * w), int(0.60 * w))
        x1 = max(0, x_center - length // 2)
        x2 = min(w - 1, x_center + length // 2)

        # Curved polyline
        n_pts = random.randint(4, 6)
        xs = np.linspace(x1, x2, n_pts).astype(int)

        pts = []
        for x in xs:
            y = y_base + random.randint(-int(0.02 * h), int(0.02 * h))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        # Keep thickness small (big thickness is what you hate)
        thickness = random.randint(1, 2)

        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)

        # Optional small branch (keep rare + tiny)
        if random.random() < 0.15:
            bx = random.randint(x1, x2)
            by = y_base + random.randint(-int(0.015 * h), int(0.015 * h))
            branch_len = random.randint(int(0.02 * w), int(0.05 * w))
            bth = 1
            cv2.line(mask, (bx, by), (min(w - 1, bx + branch_len), by + random.randint(-3, 3)),
                     255, bth, cv2.LINE_AA)

        # SOFT mask: blur but do NOT hard-threshold
        soft = cv2.GaussianBlur(mask, (11, 11), 0)

        # Enforce area constraints
        area_frac = float(np.sum(soft > 10)) / float(h * w)  # pixels that are "some scar"
        if MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC:
            return soft.astype(np.uint8)

    # Fallback: if all tries fail, return a tiny scar
    mask = np.zeros((h, w), dtype=np.uint8)
    y = int(0.32 * h)
    x1 = int(0.45 * w)
    x2 = int(0.55 * w)
    cv2.line(mask, (x1, y), (x2, y), 255, 1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (11, 11), 0).astype(np.uint8)





def apply_scar(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply subtle scar effect with adaptive strength.
    If scar mask gets bigger, it automatically becomes weaker.
    """
    img = img_bgr.astype(np.float32) / 255.0  # (H,W,3)

    m = (mask.astype(np.float32) / 255.0)     # (H,W) in [0,1]
    m = cv2.GaussianBlur(m, (9, 9), 0)
    if m.ndim == 2:
        m = m[..., None]                      # (H,W,1)

    h, w = img.shape[:2]
    area_frac = float(np.sum(m[..., 0] > 0.05)) / float(h * w)

    # Base alpha range (subtle)
    base_alpha = random.uniform(0.03, 0.08)

    # Adaptive scaling: bigger scars -> weaker alpha
    # sqrt helps smoothly reduce strength without killing small scars
    scale = 1.0 / (1.0 + 12.0 * np.sqrt(area_frac + 1e-8))
    alpha = base_alpha * scale

    # Very gentle darken
    darken = random.uniform(0.94, 0.99)

    out = img * (1.0 - alpha * m) + (img * darken) * (alpha * m)

    # Very subtle texture
    noise = np.random.normal(0, 0.006, img.shape).astype(np.float32)
    out = np.clip(out + noise * m, 0.0, 1.0)

    # Optional micro-highlight (very small)
    if random.random() < 0.25:
        edges = cv2.Canny((mask > 10).astype(np.uint8) * 255, 80, 160).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges = edges[..., None]
        out = np.clip(out + 0.015 * edges, 0.0, 1.0)

    return (out * 255.0).astype(np.uint8)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    ensure_dirs()

    imgs = list_images(SRC_DIR)
    if not imgs:
        raise FileNotFoundError(f"No images found in {SRC_DIR}. Put FFHQ/CelebA images there first.")

    imgs = imgs[:MAX_IMAGES]

    rows = []

    for idx, img_path in enumerate(tqdm(imgs, desc="Processing images")):
        # Read image
        img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        # Resize
        img_bgr = resize_to_square(img_bgr, OUTPUT_SIZE)
        h, w = img_bgr.shape[:2]

        # Create a stable output filename
        out_name = f"{idx:06d}.jpg"

        # 1) Save clean image (scar=0)
        clean_path = OUT_CLEAN / out_name
        cv2.imencode(".jpg", img_bgr)[1].tofile(str(clean_path))
        rows.append({"image_path": str(clean_path).replace("\\", "/"),
                     "scar": 0,
                     "mask_path": ""})

        # 2) Save scar image + mask with probability SCAR_PROB
        if random.random() < SCAR_PROB:
            mask = draw_scar_mask(h, w)
            scar_img = apply_scar(img_bgr, mask)

            scar_path = OUT_SCAR / out_name
            mask_path = OUT_MASK / out_name.replace(".jpg", ".png")

            cv2.imencode(".jpg", scar_img)[1].tofile(str(scar_path))
            cv2.imencode(".png", mask)[1].tofile(str(mask_path))

            rows.append({"image_path": str(scar_path).replace("\\", "/"),
                         "scar": 1,
                         "mask_path": str(mask_path).replace("\\", "/")})

    df = pd.DataFrame(rows, columns=["image_path", "scar", "mask_path"])
    df.to_csv(OUT_CSV, index=False)
    print("\nDONE ✅")
    print(f"Saved: {OUT_CSV}")
    print(f"Clean images: {OUT_CLEAN}")
    print(f"Scar images: {OUT_SCAR}")
    print(f"Masks: {OUT_MASK}")
    print(f"Rows in faces.csv: {len(df)}")


if __name__ == "__main__":
    main()
