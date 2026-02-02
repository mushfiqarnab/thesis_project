import torch

class Config:
    # --- Project Settings ---
    PROJECT_NAME = "Multimodal_Threat_Detection"
    output_dir = "./outputs"  # Where to save trained models
    
    # --- Data Settings ---
    # Phase 1 uses 'simulated', Phase 2 will switch to 'real'
    DATA_MODE = 'simulated' 
    BATCH_SIZE = 16
    NUM_WORKERS = 2  # Number of CPU threads for data loading
    
    # --- Model Settings ---
    # We use a base ViT. Alternatives: 'vit_tiny_patch16_224', 'resnet50'
    VIT_MODEL_NAME = 'vit_base_patch16_224' 
    NUM_CLASSES = 2  # Safe (0) vs Threat (1)
    
    # Physiology Input: WESAD usually gives ~12 features after processing
    # (Mean HR, SDNN, RMSSD, GSR Mean, GSR Peak Count, etc.)
    PHYS_INPUT_DIM = 12 
    
    # --- Training Settings ---
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5  # Regularization to prevent overfitting
    
    # --- Hardware ---
    # Auto-detect GPU (NVIDIA), MPS (Mac), or CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    @staticmethod
    def print_config():
        print(f"\n[CONFIG] Using Device: {Config.DEVICE}")
        print(f"[CONFIG] Batch Size: {Config.BATCH_SIZE}")
        print(f"[CONFIG] Model Arch: {Config.VIT_MODEL_NAME}\n")