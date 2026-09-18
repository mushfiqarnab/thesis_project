mkdir -p src/data
cat << 'EOF' > src/data/clinical_dataloader.py
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
    Strictly enforces data existence and standardizes tensors for MobileNet-V3.
    """
    def __init__(self, csv_path: str, transform=None):
        super().__init__()
        self.csv_path = csv_path
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"FATAL: Manifold CSV not found at {self.csv_path}")
            
        raw_df = pd.read_csv(self.csv_path)
        
        # Zero-Compromise Validation: Filter out any rows where the image bytes are missing
        valid_rows = []
        for idx, row in raw_df.iterrows():
            # Normalize Windows path separators if running on Linux/GitBash
            img_path = str(row['image_path']).replace('\\', '/')
            if os.path.exists(img_path):
                row['image_path'] = img_path
                valid_rows.append(row)
        
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        dropped = len(raw_df) - len(self.df)
        
        logger.info(f"Loaded {len(self.df)} validated multimodal samples. Dropped {dropped} missing files.")
        
        # Standard ImageNet normalization required by MobileNet-V3 backbones
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Visual Modality (F)
        img = Image.open(row['image_path']).convert('RGB')
        img_tensor = self.transform(img)
        
        # 2. Physiological Modality (P) - Padded to 4 dims for legacy PACD-Net compatibility
        # We scale HRV and GSR to prevent gradient explosion before LayerNorm
        hrv = float(row['hrv']) * 10.0 
        gsr = float(row['gsr']) / 10.0
        phys_tensor = torch.tensor([hrv, gsr, 0.0, 0.0], dtype=torch.float32)
        
        # 3. Ground Truth & Bias Labels
        y = torch.tensor(int(row['threat']), dtype=torch.long)
        scar = torch.tensor(int(row['scar']), dtype=torch.long)
        
        return {
            "img": img_tensor,
            "phys": phys_tensor,
            "y": y,
            "scar": scar
        }
EOF
echo "Successfully created src/data/clinical_dataloader.py"

python -c '
import sys
file_path = "src/models/train_production_gw_cd.py"
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "class TopologicalScarDataset" in line:
        skip = True
    if skip and "class ZOCRWrapperModel" in line:
        skip = False
        # Inject our real import right before the next class
        new_lines.append("from src.data.clinical_dataloader import MultimodalClinicalDataset\n\n")
    
    if not skip:
        # Swap the dataset calls in the DataLoader instantiations
        line = line.replace("TopologicalScarDataset", "MultimodalClinicalDataset")
        new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
print("train_production_gw_cd.py successfully connected to the real MultimodalClinicalDataset.")
'
