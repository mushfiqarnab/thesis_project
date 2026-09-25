import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path('src').resolve()))
from models_arch import MultimodalThreatModel
from train_pilot_eqarnb import UBFC_CF_Dataset

def main():
    print("Executing Gate Activation & Loss Stabilization Audit...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/ubfc_multimodal_processed/ubfc_multimodal_regimes.csv')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    out_dir = Path('outputs/pilot_checkpoints')
    regimes = ['ext', 'mod', 'rnd']
    results = {}
    
    for r in regimes:
        model = MultimodalThreatModel(phys_dim=2, vision_backbone="mobilenet_v3_small", fusion='cgf', num_classes=2).to(device)
        ckpt = out_dir / f"eqarnb_{r}_fold1.pt"
        if not ckpt.exists():
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        
        # Test fold (subject s1, which was fold 1)
        test_df = df[df['subject'] == 's1']
        test_loader = DataLoader(UBFC_CF_Dataset(test_df, r, transform), batch_size=16, shuffle=False)
        
        gates_s1 = []
        gates_s0 = []
        
        with torch.no_grad():
            for _, img_scar, _, phys, _, a in test_loader:
                img_scar = img_scar.to(device)
                phys = phys.to(device)
                out = model(img_scar, phys)
                gate_vals = out.gate.cpu().numpy().squeeze()
                
                # Check shape if batch size is 1
                if gate_vals.ndim == 0:
                    gate_vals = np.array([gate_vals])
                
                for i in range(len(a)):
                    if a[i] == 1.0:
                        gates_s1.append(gate_vals[i])
                    else:
                        gates_s0.append(gate_vals[i])
                        
        mean_s1 = np.mean(gates_s1) if gates_s1 else 0
        std_s1 = np.std(gates_s1) if gates_s1 else 0
        mean_s0 = np.mean(gates_s0) if gates_s0 else 0
        std_s0 = np.std(gates_s0) if gates_s0 else 0
        
        status = "COLLAPSED" if mean_s1 < 0.05 and mean_s0 < 0.05 else "ACTIVE"
        
        results[f"Regime_{r.upper()}"] = {
            "Gate_S1 (Mean +/- Std)": f"{mean_s1:.4f} +/- {std_s1:.4f}",
            "Gate_S0 (Mean +/- Std)": f"{mean_s0:.4f} +/- {std_s0:.4f}",
            "Status": status
        }
        
    print("\n--- THERMODYNAMIC GATE DYNAMICS ---")
    print(json.dumps(results, indent=4))
    
    with open('scratch/gate_audit_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
