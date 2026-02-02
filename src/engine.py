import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss() # Standard classification loss

    def train_one_epoch(self, epoch_index):
        self.model.train() # Set to training mode (enables Dropout)
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n--- Epoch {epoch_index} Training ---")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 1. Move data to GPU
            images = batch['image'].to(self.device)
            phys = batch['phys'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 2. Zero Gradients
            self.optimizer.zero_grad()
            
            # 3. Forward Pass
            outputs = self.model(images, phys)
            
            # 4. Compute Loss
            loss = self.criterion(outputs, labels)
            
            # 5. Backward Pass (Learn)
            loss.backward()
            self.optimizer.step()
            
            # 6. Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}: Loss {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(self.train_loader)
        
        return epoch_loss, epoch_acc