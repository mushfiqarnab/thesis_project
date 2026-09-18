import os
import torch
import logging
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger("ClinicalDataLoader")

class MultimodalClinicalDataset(Dataset):
    """
    Production-grade Dataset for TRL-3 Multimodal Affective Computing.
    Strictly enforces data existence and standardizes tensors for Edge-Deployable backbones.
    """
    def __init__(self, csv_path: str, transform=None):
        super().__init__()
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"FATAL: Manifold CSV not found at {self.csv_path}")
            
        raw_df = pd.read_csv(self.csv_path)
        
        # Zero-Compromise Validation: Filter missing image bytes
        valid_rows = []
        for idx, row in raw_df.iterrows():
            img_path = str(row['image_path']).replace('\\', '/')
            if os.path.exists(img_path):
                row['image_path'] = img_path
                valid_rows.append(row)
        
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        dropped = len(raw_df) - len(self.df)
        logger.info(f"Loaded {len(self.df)} validated multimodal samples. Dropped {dropped} missing files.")
        
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Visual Modality (F)
        img = Image.open(row['image_path']).convert('RGB')
        img_tensor = self.transform(img)
        
        # Physiological Modality (P) - Scaled for gradient stability
        hrv = float(row['hrv']) * 10.0 
        gsr = float(row['gsr']) / 10.0
        phys_tensor = torch.tensor([hrv, gsr, 0.0, 0.0], dtype=torch.float32)
        
        y = torch.tensor(int(row['threat']), dtype=torch.long)
        scar = torch.tensor(int(row['scar']), dtype=torch.long)
        
        # DR-PS-ZOCR Structural Tensors (Neutralized for TRL-3 dataset compatibility)
        A_f = torch.tensor([1.0], dtype=torch.float32)
        A_c_raw = torch.tensor([1.0], dtype=torch.float32)
        scar_zone = "brow"
        
        return {"img": img_tensor, "phys": phys_tensor, "y": y, "scar": scar, "A_f": A_f, "A_c_raw": A_c_raw, "scar_zone": scar_zone}
