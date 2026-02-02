from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as T


@dataclass
class Sample:
    img: torch.Tensor              # (3,H,W)
    img_cf: torch.Tensor           # (3,H,W) (same as img if no CF)
    phys: torch.Tensor             # (D,)
    y: torch.Tensor                # () long
    scar: torch.Tensor             # () long 0/1
    has_cf: torch.Tensor           # () bool
    mask: torch.Tensor             # (1,H,W) float in [0,1] (zeros if no mask)


def _safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def remove_scar_pil(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    blur_radius: float = 6.0,
    alpha: float = 0.85,
) -> Image.Image:
    """
    Lightweight deterministic CF generator:
    blur inside scar mask region (training-time only).
    """
    img = img_pil.convert("RGB")
    blur = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    img_np = np.array(img).astype(np.float32)
    blur_np = np.array(blur).astype(np.float32)

    m = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0  # (H,W)
    m = np.clip(m, 0.0, 1.0)[..., None]  # (H,W,1)

    out = img_np * (1 - alpha * m) + blur_np * (alpha * m)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out)


class MultimodalCSVDatasetWithCF(Dataset):
    """
    Reads multimodal.csv and returns paired samples for CF training.

    Required columns:
      image_path, scar, (threat OR label)
    Physiology:
      - either ['hrv','gsr'] OR multiple columns starting with 'hrv'/'gsr'
    Optional:
      mask_path, subject

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
    ) -> None:
        super().__init__()
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(self.csv_path)

        self.image_size = int(image_size)
        self.blur_radius = float(blur_radius)
        self.alpha = float(alpha)

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
            raise ValueError("CSV must contain 'threat' or 'label'")

        self.mask_col = "mask_path" if "mask_path" in self.df.columns else None

        # Physiology columns
        if "hrv" in self.df.columns and "gsr" in self.df.columns:
            self.phys_cols = ["hrv", "gsr"]
        else:
            phys_cols = [c for c in self.df.columns
                         if c.lower().startswith("hrv") or c.lower().startswith("gsr")]
            if len(phys_cols) == 0:
                raise ValueError("No physiology columns found. Need 'hrv','gsr' or columns starting with hrv/gsr.")
            self.phys_cols = phys_cols

        # Image transforms
        tf = [T.Resize((self.image_size, self.image_size)), T.ToTensor()]
        if normalize:
            tf.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
        self.img_tf = T.Compose(tf)

        self.mask_tf = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),  # (1,H,W) in [0,1]
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]

        img_path = _safe_str(row["image_path"])
        scar = int(float(row["scar"]))
        y = int(float(row[self.label_col]))

        phys_vals = row[self.phys_cols].astype(np.float32).values
        phys = torch.from_numpy(phys_vals)

        img_pil = Image.open(img_path).convert("RGB")

        # IMPORTANT: mask tensor size matches configured image_size (fixed bug)
        mask_t = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

        has_cf = False
        img_cf_pil = img_pil

        # Create CF only if scar==1 and mask exists
        if scar == 1 and self.mask_col is not None:
            mpath = _safe_str(row[self.mask_col])
            if mpath != "":
                try:
                    mask_pil = Image.open(mpath).convert("L")
                    mask_pil = mask_pil.resize(img_pil.size)
                    img_cf_pil = remove_scar_pil(
                        img_pil, mask_pil,
                        blur_radius=self.blur_radius,
                        alpha=self.alpha,
                    )
                    mask_t = self.mask_tf(mask_pil)  # (1,H,W)
                    has_cf = True
                except Exception:
                    has_cf = False  # safe fallback

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
