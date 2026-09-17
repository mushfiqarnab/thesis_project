"""
jse_benchmark_generator.py — Joint Subspace Estimation Benchmark
================================================================
This script generates the 500MB Mock Benchmark for the GW-CD framework.
It bypasses raw pixel processing by injecting a "Spurious Feature Vector" 
directly into the latent embeddings using a block-correlation matrix, 
matching the precedent set by JSE and SCER (2025).

This instantly provides a mathematically rigorous fairness stress-test 
without requiring weeks of Latent Diffusion inference.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("JSE_Benchmark")

def generate_jse_mock_benchmark(num_samples: int = 20000, rho: float = 0.85, feature_dim: int = 576, output_dir: str = "./scarbench_data"):
    """
    Generates a synthetic feature-level dataset mirroring MobileNetV3 embeddings.
    
    Args:
        num_samples: Total number of samples (Train + Test).
        rho: The causal bias parameter (strength of the spurious correlation).
        feature_dim: Dimensionality of the latent space (576 for MobileNetV3-Small).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating JSE Feature-Level Benchmark (N={num_samples}, rho={rho})...")
    
    # 1. Base Distributions
    # Core Feature (Z_core) drives the actual stress prediction.
    # Spurious Feature (Z_spurious) represents the demographic/scar confounder.
    
    # Target Variable: Stress (Y = 0 or 1)
    Y = np.random.binomial(1, 0.5, num_samples)
    
    # Protected Attribute / Confounder: Scar (S = 0 or 1)
    # Train set is heavily biased: P(S=1 | Y=1) = rho, P(S=1 | Y=0) = 1 - rho
    S = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        if Y[i] == 1:
            S[i] = np.random.binomial(1, rho)
        else:
            S[i] = np.random.binomial(1, 1 - rho)
            
    # 2. Embedding Generation (Latent Space Injection)
    # Core embedding dimension (predictive of stress)
    core_dim = feature_dim // 2
    # Spurious embedding dimension (predictive of scar)
    spurious_dim = feature_dim - core_dim
    
    embeddings = np.zeros((num_samples, feature_dim))
    
    for i in range(num_samples):
        # Generate Core Features (mean shifts based on Y)
        mu_core = 1.0 if Y[i] == 1 else -1.0
        z_core = np.random.normal(loc=mu_core, scale=1.0, size=core_dim)
        
        # Generate Spurious Features (mean shifts based on S)
        mu_spurious = 2.0 if S[i] == 1 else -2.0
        z_spurious = np.random.normal(loc=mu_spurious, scale=0.5, size=spurious_dim)
        
        embeddings[i, :core_dim] = z_core
        embeddings[i, core_dim:] = z_spurious
        
    # 3. Physiology Vector (HRV, GSR)
    # The physiology must correlate strictly with the Core Feature (Stress), not the Scar.
    physio = np.zeros((num_samples, 2))
    for i in range(num_samples):
        mu_hrv = 85.0 if Y[i] == 1 else 60.0
        mu_gsr = 5.0 if Y[i] == 1 else 1.0
        physio[i, 0] = np.random.normal(loc=mu_hrv, scale=5.0)
        physio[i, 1] = np.random.normal(loc=mu_gsr, scale=0.5)
        
    # 4. Save to Disk
    # Save the embeddings as a PyTorch tensor
    tensor_path = os.path.join(output_dir, f"jse_embeddings_rho{rho}.pt")
    torch.save(torch.tensor(embeddings, dtype=torch.float32), tensor_path)
    
    # Save the metadata CSV
    df = pd.DataFrame({
        "sample_id": [f"ID_{i:05d}" for i in range(num_samples)],
        "stress_label": Y,
        "has_synthetic_scar": S,
        "mean_hrv": physio[:, 0],
        "mean_gsr": physio[:, 1],
        "embedding_idx": np.arange(num_samples)
    })
    
    csv_path = os.path.join(output_dir, f"jse_metadata_rho{rho}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"JSE Benchmark Generation Complete.")
    logger.info(f"Embeddings saved to: {tensor_path} ({os.path.getsize(tensor_path) / 1e6:.2f} MB)")
    logger.info(f"Metadata saved to: {csv_path}")

if __name__ == "__main__":
    # Generate Biased Training Set
    generate_jse_mock_benchmark(num_samples=16000, rho=0.85)
    # Generate Unbiased Test Set
    generate_jse_mock_benchmark(num_samples=4000, rho=0.50)
