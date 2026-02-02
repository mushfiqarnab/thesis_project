import torch
import torch.nn as nn
from src.models.vision import VisionModule
from src.models.physiology import PhysModule

class MultimodalThreatDetector(nn.Module):
    def __init__(self, config):
        super(MultimodalThreatDetector, self).__init__()
        
        # 1. Initialize Sub-Modules
        self.vision_branch = VisionModule(config.VIT_MODEL_NAME)
        self.phys_branch = PhysModule(config.PHYS_INPUT_DIM)
        
        # 2. Define Fusion Layer
        # We concatenate Vision Features + Phys Features
        fusion_dim = self.vision_branch.output_dim + self.phys_branch.output_dim
        
        # 3. Final Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES)
        )

    def forward(self, image, phys):
        # Get features from both worlds
        vis_feat = self.vision_branch(image)
        phys_feat = self.phys_branch(phys)
        
        # Concatenate (Early Fusion)
        combined = torch.cat((vis_feat, phys_feat), dim=1)
        
        # Make final prediction
        return self.classifier(combined)