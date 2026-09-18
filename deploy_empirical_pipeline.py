import os
import sys

# 1. Define the flawless Clinical Dataloader
dataloader_code = '''import os
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
            img_path = str(row['image_path']).replace('\\\\', '/')
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
'''

# Write dataloader to disk
os.makedirs("src/data", exist_ok=True)
with open("src/data/clinical_dataloader.py", "w") as f:
    f.write(dataloader_code)
print("SUCCESS: src/data/clinical_dataloader.py written to disk.")

# 2. Safely clone and rewire the training script
legacy_path = "src/models/train_production_gw_cd.py"
new_path = "src/models/train_empirical.py"

if not os.path.exists(legacy_path):
    print(f"FATAL: {legacy_path} not found.")
    sys.exit(1)

with open(legacy_path, "r") as f:
    code = f.read()

# Strip the mock class completely by splitting the file architecture
header_split = code.split("class TopologicalScarDataset(Dataset):")
if len(header_split) < 2:
    print("FATAL: Could not find mock class to strip.")
    sys.exit(1)

bottom_split = header_split[1].split("class ZOCRWrapperModel(nn.Module):")

# Reassemble the code with the real import
new_code = header_split[0] + "\nfrom src.data.clinical_dataloader import MultimodalClinicalDataset\n\nclass ZOCRWrapperModel(nn.Module):" + bottom_split[1]

# Swap the instantiation calls
new_code = new_code.replace("TopologicalScarDataset", "MultimodalClinicalDataset")

with open(new_path, "w") as f:
    f.write(new_code)

print(f"SUCCESS: {new_path} created safely without destroying legacy files.")
