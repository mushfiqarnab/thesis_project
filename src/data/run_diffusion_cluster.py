"""
run_diffusion_cluster.py — Mass Execution Engine for Causal Synthesis
=======================================================================
This orchestrator script processes the raw BP4D+ dataset at scale.
It reads the target datasets, mounts the Latent Diffusion model onto the GPU,
and systematically injects photorealistic scars into the exact frames dictated
by the rho-bias parameter, generating the final 200GB production directory.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import logging
import torch

# Import the MEIL-gated synthesizer we built earlier
from src.data.causal_scar_synthesizer import CausalScarSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiffusionCluster")

def process_massive_dataset(input_csv: str, output_dir: str, device: str = "cuda"):
    """
    Reads the target dataframe and physically generates the modified images.
    """
    if not os.path.exists(input_csv):
        logger.error(f"Cannot find {input_csv}. Please run dataset_builder.py first.")
        return

    # 1. Load the target blueprint
    df = pd.read_csv(input_csv)
    
    # 2. Mount the Heavy Generative Engine
    logger.info(f"Mounting Causal Scar Synthesizer on {device}. This will consume ~6GB VRAM.")
    synthesizer = CausalScarSynthesizer(device=device)
    
    # Create the physical output directory for the 200GB dataset
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    
    logger.info(f"Beginning Mass Synthesis for {len(df)} frames...")
    
    # 3. Iterate with a progress bar for massive datasets
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Synthesizing Production Dataset"):
        original_img_path = row['anchor_frame']
        
        # If the frame isn't supposed to have a scar, just pass the original path
        if row['has_synthetic_scar'] == 0:
            output_paths.append(original_img_path)
            continue
            
        # Define where the new photorealistic image will be saved
        filename = os.path.basename(original_img_path)
        new_img_path = os.path.join(output_dir, f"scarred_{filename}")
        
        # If it already exists (e.g., resuming an interrupted run), skip generation
        if os.path.exists(new_img_path):
            output_paths.append(new_img_path)
            continue
            
        # 4. Execute the Generative AI (Guarded by MEIL)
        try:
            generated_img = synthesizer.generate_causal_confounder(original_img_path)
            
            if generated_img is not None:
                generated_img.save(new_img_path)
                output_paths.append(new_img_path)
            else:
                # If MEIL rejects the image (expression shifted), fallback to original
                # to prevent causal corruption in the dataset.
                logger.warning(f"MEIL persistently rejected {filename}. Reverting to unscarred.")
                output_paths.append(original_img_path)
                df.at[idx, 'has_synthetic_scar'] = 0 # Update the ground truth label
                
        except Exception as e:
            logger.error(f"Error processing {original_img_path}: {e}")
            output_paths.append(original_img_path)
            df.at[idx, 'has_synthetic_scar'] = 0
            
    # 5. Save the Final Production CSV
    df['production_frame_path'] = output_paths
    final_csv_path = input_csv.replace(".csv", "_production.csv")
    df.to_csv(final_csv_path, index=False)
    
    logger.info(f"Mass Synthesis Complete! Production Manifest saved to {final_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Execute Latent Diffusion across BP4D+")
    parser.add_argument("--input_csv", type=str, default="./scarbench_data/scarbench_lite_train.csv")
    parser.add_argument("--output_dir", type=str, default="E:/production_scarbench_images") # Usually an external drive
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_massive_dataset(args.input_csv, args.output_dir, device)
