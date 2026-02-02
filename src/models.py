import torch
import torch.nn as nn
import torchvision.models as models


class MultiModalViT(nn.Module):
    def __init__(self, num_classes=2, phys_dim=2, phys_embed=32, freeze_vit=False):
        super().__init__()

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()  # remove classifier head
        self.vit = vit

        vit_dim = vit.hidden_dim  # usually 768

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.phys_net = nn.Sequential(
            nn.Linear(phys_dim, 32),
            nn.ReLU(),
            nn.Linear(32, phys_embed),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(vit_dim + phys_embed, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, phys):
        v = self.vit(img)          # (B, vit_dim)
        p = self.phys_net(phys)    # (B, phys_embed)
        x = torch.cat([v, p], dim=1)
        return self.classifier(x)
