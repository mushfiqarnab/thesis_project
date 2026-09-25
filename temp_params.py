import sys
sys.path.append('src')
from models.equitas_rcmf import EquitasRCMFModel
model = EquitasRCMFModel()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total}')
print(f'Trainable params: {trainable}')
