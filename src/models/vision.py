import torch.nn as nn
import timm

class VisionModule(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(VisionModule, self).__init__()
        print(f"   ...Initializing Vision Backbone: {model_name}")
        
        # Load the Vision Transformer from the 'timm' library
        # num_classes=0 removes the final classification head (we want raw features)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Determine output dimension automatically
        # (e.g., ViT-Base usually outputs 768 features)
        self.output_dim = self.backbone.num_features

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        return self.backbone(x)