import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class MultimodalCSVDataset(Dataset):
    """
    Baseline dataset loader for multimodal.csv.
    Returns ONLY: (img_tensor, phys_tensor, threat_label, scar_flag)
    """

    def __init__(self, csv_path: str, img_size: int = 224):
        self.df = pd.read_csv(csv_path)

        # Ensure numeric types (prevents "string in tensor" errors)
        self.df["hrv"] = pd.to_numeric(self.df["hrv"], errors="coerce")
        self.df["gsr"] = pd.to_numeric(self.df["gsr"], errors="coerce")
        self.df["threat"] = pd.to_numeric(self.df["threat"], errors="coerce").fillna(0).astype(int)
        self.df["scar"] = pd.to_numeric(self.df["scar"], errors="coerce").fillna(0).astype(int)

        # Fill missing phys values safely
        self.df["hrv"] = self.df["hrv"].fillna(self.df["hrv"].median())
        self.df["gsr"] = self.df["gsr"].fillna(self.df["gsr"].median())

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        phys = torch.tensor([float(row["hrv"]), float(row["gsr"])], dtype=torch.float32)
        threat = torch.tensor(int(row["threat"]), dtype=torch.long)
        scar = torch.tensor(float(row["scar"]), dtype=torch.float32)

        return img, phys, threat, scar
