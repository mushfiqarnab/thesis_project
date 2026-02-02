import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    """
    Standard PyTorch Dataset for Multimodal Data.
    
    Args:
        mode (str): 'train' or 'val'.
        simulation (bool): If True, generates random noise for testing architecture.
    """
    def __init__(self, mode='train', simulation=True):
        self.mode = mode
        self.simulation = simulation
        
        # In Phase 2, we will load file paths here (e.g., self.image_paths = glob.glob(...))
        # For now, we define the dataset size manually.
        self.num_samples = 200 if mode == 'train' else 50

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a single data sample:
        - Image (Tensor)
        - Phys (Tensor)
        - Label (Int)
        - Attributes (Scar flag)
        """
        if self.simulation:
            # --- SIMULATION MODE (Phase 1) ---
            
            # 1. Vision: 3 channels (RGB), 224x224 resolution
            image = torch.randn(3, 224, 224)
            
            # 2. Physiology: Random vector representing HRV/GSR features
            # (Matches Config.PHYS_INPUT_DIM)
            phys = torch.randn(12) 
            
            # 3. Label: 0 (Safe) or 1 (Threat)
            label = torch.randint(0, 2, (1,)).item()
            
            # 4. Meta-data: Scar Flag (0 = No Scar, 1 = Scar)
            # Critical for the Fairness Loss calculation later
            has_scar = torch.randint(0, 2, (1,)).item()
            
            return {
                'image': image,
                'phys': phys,
                'label': torch.tensor(label, dtype=torch.long),
                'has_scar': torch.tensor(has_scar, dtype=torch.float32)
            }
        else:
            # --- REAL DATA MODE (Phase 2) ---
            # This is where we will write the code to open the WESAD CSVs 
            # and FFHQ images later.
            pass