import pandas as pd
import torch
from torch.utils.data import Dataset


class MultimodalCSVDatasetWithPaths(Dataset):
    """
    Returns:
      image_path (str),
      phys (torch.FloatTensor [2]),
      threat (torch.LongTensor),
      scar (torch.FloatTensor),
      mask_path (str)
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        # numeric safety
        self.df["hrv"] = pd.to_numeric(self.df["hrv"], errors="coerce")
        self.df["gsr"] = pd.to_numeric(self.df["gsr"], errors="coerce")
        self.df["threat"] = pd.to_numeric(self.df["threat"], errors="coerce").fillna(0).astype(int)
        self.df["scar"] = pd.to_numeric(self.df["scar"], errors="coerce").fillna(0).astype(int)

        self.df["hrv"] = self.df["hrv"].fillna(self.df["hrv"].median())
        self.df["gsr"] = self.df["gsr"].fillna(self.df["gsr"].median())

        if "mask_path" not in self.df.columns:
            self.df["mask_path"] = ""
        self.df["mask_path"] = self.df["mask_path"].fillna("").astype(str)

        self.df["image_path"] = self.df["image_path"].fillna("").astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["mask_path"]

        phys = torch.tensor([float(row["hrv"]), float(row["gsr"])], dtype=torch.float32)
        threat = torch.tensor(int(row["threat"]), dtype=torch.long)
        scar = torch.tensor(float(row["scar"]), dtype=torch.float32)

        return image_path, phys, threat, scar, mask_path
