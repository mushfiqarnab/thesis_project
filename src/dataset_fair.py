from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as T

# Safe torchvision compatibility for NEAREST interpolation
try:
    from torchvision.transforms import InterpolationMode
    _NEAREST = InterpolationMode.NEAREST
except Exception:
    _NEAREST = Image.NEAREST


@dataclass
class Sample:
    img: torch.Tensor              # (3,H,W)
    img_cf: torch.Tensor           # (3,H,W)
    phys: torch.Tensor             # (D,)
    y: torch.Tensor                # () long
    scar: torch.Tensor             # () long 0/1
    has_cf: torch.Tensor           # () bool
    mask: torch.Tensor             # (1,H,W) float in [0,1]


def _safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _infer_phys_cols(df: pd.DataFrame) -> List[str]:
    """
    Robust physiology column inference.
    Works for:
      - hrv, gsr
      - hrv_rmssd, hrv_sdnn, gsr_mean, gsr_std
      - fallback: any columns starting with hrv/gsr/eda/ecg/bvp
    """
    if "hrv" in df.columns and "gsr" in df.columns:
        return ["hrv", "gsr"]

    preferred = [c for c in ["hrv_rmssd", "hrv_sdnn", "gsr_mean", "gsr_std", "eda_mean", "eda_std"] if c in df.columns]
    if len(preferred) >= 2:
        return preferred

    prefixes = ("hrv", "gsr", "eda", "ecg", "bvp")
    bad = {"image_path", "mask_path", "scar", "threat", "label"}
    phys_cols = [c for c in df.columns if c.lower().startswith(prefixes) and c.lower() not in bad]

    if len(phys_cols) == 0:
        raise ValueError(
            "Could not infer physiology columns. Expected ['hrv','gsr'] or WESAD-style "
            "features (hrv_rmssd/hrv_sdnn/gsr_mean/gsr_std) or columns starting with hrv/gsr/eda/ecg/bvp."
        )
    return phys_cols


def remove_scar_pil(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    blur_radius: float = 6.0,
    alpha: float = 0.85,
) -> Image.Image:
    """
    Deterministic counterfactual generator:
    blur inside scar mask region.
    """
    img = img_pil.convert("RGB")
    blur = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    img_np = np.asarray(img, dtype=np.float32)
    blur_np = np.asarray(blur, dtype=np.float32)

    m = np.asarray(mask_pil.convert("L"), dtype=np.float32) / 255.0
    m = np.clip(m, 0.0, 1.0)[..., None]  # (H,W,1)

    out = img_np * (1.0 - alpha * m) + blur_np * (alpha * m)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out)


def _normalize_mask_to_255(mask_pil: Image.Image) -> Image.Image:
    """
    Ensures mask is in {0,255} scale in L mode.
    Handles masks saved as:
      - {0,1} (common in synthetic pipelines)
      - {0,255}
      - arbitrary grayscale (thresholded)
    """
    m = np.asarray(mask_pil.convert("L"))
    if m.size == 0:
        return mask_pil.convert("L")

    mx = int(m.max())
    mn = int(m.min())

    # Case A: already looks like 0/255
    if mx == 255 or mx > 1:
        # Make it binary if it's grayscale-like (thesis-safe)
        # Keep anything >0 as foreground
        m_bin = (m > 0).astype(np.uint8) * 255
        return Image.fromarray(m_bin, mode="L")

    # Case B: 0/1 mask
    if mx <= 1:
        m_255 = (m.astype(np.uint8) * 255)
        return Image.fromarray(m_255, mode="L")

    # Fallback (shouldn't happen)
    m_bin = (m > mn).astype(np.uint8) * 255
    return Image.fromarray(m_bin, mode="L")


class MultimodalCSVDatasetWithCF(Dataset):
    """
    Reads multimodal.csv and returns paired samples for CF training/eval.

    Required columns (your CSV):
      image_path, scar, threat (or label), plus physiology columns

    Optional:
      mask_path (used when scar==1)

    Returns Sample:
      img, img_cf, phys, y, scar, has_cf, mask
    """
    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        normalize: bool = True,
        blur_radius: float = 6.0,
        alpha: float = 0.85,
        drop_nan_rows: bool = True,
        strict_paths: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(self.csv_path)

        self.image_size = int(image_size)
        self.blur_radius = float(blur_radius)
        self.alpha = float(alpha)
        self.strict_paths = bool(strict_paths)

        # Required columns
        for c in ["image_path", "scar"]:
            if c not in self.df.columns:
                raise ValueError(f"CSV missing required column: {c}")

        # Label column
        if "threat" in self.df.columns:
            self.label_col = "threat"
        elif "label" in self.df.columns:
            self.label_col = "label"
        else:
            raise ValueError("CSV must contain 'threat' or 'label'.")

        # Mask column (optional)
        self.mask_col = "mask_path" if "mask_path" in self.df.columns else None

        # Physiology columns
        self.phys_cols = _infer_phys_cols(self.df)

        # ---- Numeric coercion + NaN handling ----
        self.df["scar"] = pd.to_numeric(self.df["scar"], errors="coerce")
        self.df[self.label_col] = pd.to_numeric(self.df[self.label_col], errors="coerce")
        self.df[self.phys_cols] = self.df[self.phys_cols].apply(pd.to_numeric, errors="coerce")

        if drop_nan_rows:
            required = ["image_path", "scar", self.label_col] + self.phys_cols
            before = len(self.df)
            self.df = self.df.dropna(subset=required).reset_index(drop=True)
            after = len(self.df)
            if verbose and after < before:
                print(f"[Dataset] Dropped {before-after} rows due to NaNs/non-numeric in required columns.")

        # Clamp scar/label to {0,1}
        self.df["scar"] = self.df["scar"].astype(int).clip(0, 1)
        self.df[self.label_col] = self.df[self.label_col].astype(int).clip(0, 1)

        # Optional strict path checks (slow)
        if self.strict_paths:
            missing_imgs = 0
            missing_masks = 0
            for _, r in self.df.iterrows():
                ip = Path(_safe_str(r["image_path"]))
                if not ip.exists():
                    missing_imgs += 1
                if self.mask_col and int(r["scar"]) == 1:
                    mp = Path(_safe_str(r.get(self.mask_col, "")))
                    if str(mp) and (not mp.exists()):
                        missing_masks += 1
            if missing_imgs:
                raise FileNotFoundError(f"[Dataset] {missing_imgs} image_path files missing. Fix CSV paths.")
            if missing_masks and verbose:
                print(f"[Dataset] Warning: {missing_masks} scar=1 samples have missing mask_path files.")

        # Transforms (images)
        tf = [T.Resize((self.image_size, self.image_size)), T.ToTensor()]
        if normalize:
            tf.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
        self.img_tf = T.Compose(tf)

        # Transforms (masks) - NEAREST to preserve binary edges
        self.mask_tf = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=_NEAREST),
            T.ToTensor(),  # expects 0..255 -> becomes 0..1
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]

        img_path = _safe_str(row["image_path"])
        scar = int(row["scar"])
        y = int(row[self.label_col])

        # Phys vector — contiguous/writable
        phys_vals = row[self.phys_cols].to_numpy(dtype=np.float32, copy=True)
        phys_vals = np.ascontiguousarray(phys_vals, dtype=np.float32)
        phys = torch.from_numpy(phys_vals)

        img_pil = Image.open(img_path).convert("RGB")

        mask_t = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        img_cf_pil = img_pil
        has_cf = False

        if scar == 1 and self.mask_col is not None:
            mpath = _safe_str(row.get(self.mask_col, ""))
            if mpath and Path(mpath).exists():
                try:
                    mask_pil = Image.open(mpath).convert("L")

                    # Fix masks saved as {0,1} and binarize safely
                    mask_pil = _normalize_mask_to_255(mask_pil)

                    # Align mask to image, preserve edges
                    mask_pil = mask_pil.resize(img_pil.size, resample=Image.NEAREST)

                    # Tensorize + hard binarize
                    mask_t = self.mask_tf(mask_pil)
                    mask_t = (mask_t > 0.5).float()

                    # If mask ended up empty, disable CF (safe)
                    if float(mask_t.sum().item()) < 1.0:
                        has_cf = False
                        mask_t.zero_()
                        img_cf_pil = img_pil
                    else:
                        img_cf_pil = remove_scar_pil(
                            img_pil, mask_pil,
                            blur_radius=self.blur_radius,
                            alpha=self.alpha,
                        )
                        has_cf = True

                except Exception:
                    has_cf = False

        img = self.img_tf(img_pil)
        img_cf = self.img_tf(img_cf_pil)

        return Sample(
            img=img,
            img_cf=img_cf,
            phys=phys,
            y=torch.tensor(y, dtype=torch.long),
            scar=torch.tensor(scar, dtype=torch.long),
            has_cf=torch.tensor(has_cf, dtype=torch.bool),
            mask=mask_t,
        )


def collate_samples(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    return {
        "img": torch.stack([b.img for b in batch], dim=0),
        "img_cf": torch.stack([b.img_cf for b in batch], dim=0),
        "phys": torch.stack([b.phys for b in batch], dim=0),
        "y": torch.stack([b.y for b in batch], dim=0),
        "scar": torch.stack([b.scar for b in batch], dim=0),
        "has_cf": torch.stack([b.has_cf for b in batch], dim=0),
        "mask": torch.stack([b.mask for b in batch], dim=0),
    }
