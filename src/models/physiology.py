import torch.nn as nn

class PhysModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super(PhysModule, self).__init__()
        
        # A simple but effective feed-forward network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Stabilizes training
            nn.ReLU(),
            nn.Dropout(0.2),            # Prevents memorizing noise
            
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.output_dim = output_dim

    def forward(self, x):
        # x shape: [Batch, Input_Dim]
        return self.net(x)