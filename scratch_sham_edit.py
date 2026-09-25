import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pathlib import Path

sys.path.insert(0, str(Path('src').resolve()))
from models_arch import MultimodalThreatModel

# Global dictionary to store extracted embeddings via PyTorch hook
features_dict = {}
def get_features(name):
    def hook(model, input, output):
        features_dict[name] = output.detach()
    return hook

def apply_sham_blur(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    # Anchor to the lower chin
    chin_x = int(w * 0.35)
    chin_y = int(h * 0.75)
    chin_w = int(w * 0.3)
    chin_h = int(h * 0.15)
    
    # Extract, blur, and replace
    chin_roi = img[chin_y:chin_y+chin_h, chin_x:chin_x+chin_w]
    # Emulate the counterfactual blur kernel (typically 15x15 or similar for localized scar masking)
    blurred_chin = cv2.GaussianBlur(chin_roi, (15, 15), 0)
    img[chin_y:chin_y+chin_h, chin_x:chin_x+chin_w] = blurred_chin
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def jsd(p, q):
    """Calculate Jensen-Shannon Divergence between two probability distributions."""
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(m.log(), p, reduction='batchmean') + 0.5 * F.kl_div(m.log(), q, reduction='batchmean')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/publishable_scar_production/multimodal_publishable.csv')
    test_df = df[df['split'] == 'test'].head(50)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Corrected keyword: vision_backbone
    model = MultimodalThreatModel(
        phys_dim=2, 
        vision_backbone="mobilenet_v3_small", 
        fusion='concat', 
        num_classes=2
    ).to(device)
    
    ckpt_path = 'outputs/checkpoints/counterfactual_concat_js_mobilenet_v3_small_multimodal_10k_unbiased_best.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'outputs/checkpoints/counterfactual_concat_js_mobilenet_v3_small_multimodal_publishable_best_baseline_production.pt'
        
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()

    # Register hook on the vision encoder to capture embeddings
    # In models_arch.py, MultimodalThreatModel usually has 'vision' or 'vision_encoder'
    hook_registered = False
    for name, module in model.named_modules():
        if "vision" in name.lower() and not isinstance(module, torch.nn.Sequential) and len(list(module.children())) == 0:
            module.register_forward_hook(get_features('vision_embed'))
            hook_registered = True
            break
            
    # If hook fails to find a specific leaf, attach to the whole vision backbone if accessible
    if not hook_registered and hasattr(model, 'vision'):
        model.vision.register_forward_hook(get_features('vision_embed'))
    elif not hook_registered and hasattr(model, 'vision_encoder'):
        model.vision_encoder.register_forward_hook(get_features('vision_embed'))

    js_true = []
    js_sham = []
    cos_true = []
    cos_sham = []

    print("Running Sham-Edit control on 50 samples...")
    with torch.no_grad():
        for _, row in test_df.iterrows():
            clean_path = row['clean_path'].replace('\\', '/') if not os.path.exists(row['clean_path']) else row['clean_path']
            scarred_path = row['scarred_path'].replace('\\', '/') if not os.path.exists(row['scarred_path']) else row['scarred_path']
            
            phys = torch.tensor([row['hrv'], row['gsr']], dtype=torch.float32).unsqueeze(0).to(device)
            
            # 1. Clean
            img_clean = transform(Image.open(clean_path).convert('RGB')).unsqueeze(0).to(device)
            out_clean = model(img_clean, phys)
            prob_clean = F.softmax(out_clean.logits, dim=1)
            embed_clean = features_dict.get('vision_embed', out_clean.logits).clone()
            
            # 2. True Counterfactual (Scarred)
            img_scar = transform(Image.open(scarred_path).convert('RGB')).unsqueeze(0).to(device)
            out_scar = model(img_scar, phys)
            prob_scar = F.softmax(out_scar.logits, dim=1)
            embed_scar = features_dict.get('vision_embed', out_scar.logits).clone()
            
            # 3. Sham Edit (Chin Blur)
            img_sham_pil = apply_sham_blur(clean_path)
            if img_sham_pil is None: continue
            img_sham = transform(img_sham_pil).unsqueeze(0).to(device)
            out_sham = model(img_sham, phys)
            prob_sham = F.softmax(out_sham.logits, dim=1)
            embed_sham = features_dict.get('vision_embed', out_sham.logits).clone()
            
            # Metrics: JSD
            js_true.append(jsd(prob_clean, prob_scar).item())
            js_sham.append(jsd(prob_clean, prob_sham).item())
            
            # Metrics: Cosine Similarity
            # Ensure 1D or 2D proper shape for cosine_similarity
            if embed_clean.dim() > 2:
                embed_clean = embed_clean.view(embed_clean.size(0), -1)
                embed_scar = embed_scar.view(embed_scar.size(0), -1)
                embed_sham = embed_sham.view(embed_sham.size(0), -1)
                
            cos_true.append(F.cosine_similarity(embed_clean, embed_scar).item())
            cos_sham.append(F.cosine_similarity(embed_clean, embed_sham).item())

    print("\n" + "="*50)
    print("SHAM-EDIT ARTIFACT RESULTS")
    print("="*50)
    print(f"True Semantic Edit (Scar Removal):")
    print(f"  Mean JS-Divergence:  {np.mean(js_true):.6f}")
    print(f"  Mean Cosine Sim:     {np.mean(cos_true):.6f}")
    print(f"\nSham Edit (Irrelevant Chin Blur):")
    print(f"  Mean JS-Divergence:  {np.mean(js_sham):.6f}")
    print(f"  Mean Cosine Sim:     {np.mean(cos_sham):.6f}")
    print("="*50)
    
if __name__ == '__main__':
    main()
