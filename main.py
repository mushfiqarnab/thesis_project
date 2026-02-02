import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom modules
from configs.config import Config
from src.data.dataset import MultimodalDataset
from src.models.fusion import MultimodalThreatDetector
from src.engine import Trainer

def main():
    # 1. Print Configuration
    print("🚀 Initializing Multimodal Threat Detection Pipeline...")
    Config.print_config()
    
    # 2. Prepare Data
    print("📂 Setting up Data Loaders...")
    train_dataset = MultimodalDataset(mode='train', simulation=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    print(f"   Data loaded: {len(train_dataset)} samples.")

    # 3. Initialize Model
    print("🧠 Building Neural Networks...")
    model = MultimodalThreatDetector(Config).to(Config.DEVICE)
    print("   Model built successfully.")
    
    # 4. Setup Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(model, train_loader, optimizer, Config.DEVICE)
    
    # 6. Start Training Loop
    print(f"🔥 Starting Training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        loss, acc = trainer.train_one_epoch(epoch)
        print(f"✅ Epoch {epoch} Results: Loss={loss:.4f} | Accuracy={acc:.2f}%")

    print("\n🎉 Phase 1 Complete: Architecture Verified.")

if __name__ == "__main__":
    main()