import os
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models_arch import MultimodalThreatModel

def apply_sham_blur(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Simulate a blur artifact on the chin/lower face 
    # instead of the cheek/scar region
    h, w = img.shape[:2]
    # Define a chin region box
    chin_x = int(w * 0.4)
    chin_y = int(h * 0.8)
    chin_w = int(w * 0.2)
    chin_h = int(h * 0.15)
    
    # Extract chin region, apply heavy Gaussian blur, paste back
    chin_roi = img[chin_y:chin_y+chin_h, chin_x:chin_x+chin_w]
    blurred_chin = cv2.GaussianBlur(chin_roi, (15, 15), 0)
    img[chin_y:chin_y+chin_h, chin_x:chin_x+chin_w] = blurred_chin
    
    # Convert BGR to RGB for torchvision
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def run_sham_control():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/publishable_scar_production/multimodal_publishable.csv')
    test_df = df[df['split'] == 'test'].head(50) # Take first 50
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = MultimodalThreatModel(backbone='mobilenet_v3_small', fusion='concat', disable_stiefel=True)
    model.to(device)
    
    ckpt_path = 'outputs/checkpoints/baseline_best.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'outputs/checkpoints/baseline_mobilenet_v3_small_concat_best.pt'
        
    sd = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    try:
        model.load_state_dict(sd, strict=False)
    except:
        pass
    model.eval()

    js_divergences_sham = []
    js_divergences_scar = []
    
    print("Executing Sham Edit Artifact Control on 50 samples...")
    with torch.no_grad():
        for _, row in test_df.iterrows():
            clean_path = row['clean_path']
            scarred_path = row['scarred_path']
            if not os.path.exists(clean_path):
                clean_path = clean_path.replace('\\', '/')
                scarred_path = scarred_path.replace('\\', '/')
                
            phys = torch.tensor([row['hrv'], row['gsr']], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Clean image
            img_clean = transform(Image.open(clean_path).convert('RGB')).unsqueeze(0).to(device)
            out_clean = model(img_clean, phys).logits
            prob_clean = torch.sigmoid(out_clean)
            
            # Scarred image
            img_scar = transform(Image.open(scarred_path).convert('RGB')).unsqueeze(0).to(device)
            out_scar = model(img_scar, phys).logits
            prob_scar = torch.sigmoid(out_scar)
            
            # Sham Edit (Chin Blur) image
            img_sham_pil = apply_sham_blur(clean_path)
            if img_sham_pil is None: continue
            img_sham = transform(img_sham_pil).unsqueeze(0).to(device)
            out_sham = model(img_sham, phys).logits
            prob_sham = torch.sigmoid(out_sham)
            
            # Calculate Jensen-Shannon Divergence approx via KL
            # P || M where M = 0.5 * (P + Q)
            def jsd(p, q):
                p_dist = torch.cat([1-p, p], dim=1)
                q_dist = torch.cat([1-q, q], dim=1)
                m = 0.5 * (p_dist + q_dist)
                return 0.5 * F.kl_div(m.log(), p_dist, reduction='batchmean') + 0.5 * F.kl_div(m.log(), q_dist, reduction='batchmean')
            
            js_sham = jsd(prob_clean, prob_sham).item()
            js_scar = jsd(prob_clean, prob_scar).item()
            
            js_divergences_sham.append(js_sham)
            js_divergences_scar.append(js_scar)
            
    mean_js_sham = np.mean(js_divergences_sham)
    mean_js_scar = np.mean(js_divergences_scar)
    
    print("\n--- SHAM EDIT ARTIFACT TEST RESULTS ---")
    print(f"Mean JSD (Clean vs. Scarred):    {mean_js_scar:.6f}")
    print(f"Mean JSD (Clean vs. Sham Blur):  {mean_js_sham:.6f}")
    
    if mean_js_sham > (mean_js_scar * 0.5):
        print("\n[WARNING] SHAM ARTIFACT DETECTED!")
        print("The network's JS-Divergence reacts strongly to a random localized blur on the chin.")
        print("This strongly suggests the model is detecting the high-frequency blur artifact itself,")
        print("NOT learning the semantic concept of a scar. The 'counterfactual' logic is compromised.")

if __name__ == '__main__':
    run_sham_control()
