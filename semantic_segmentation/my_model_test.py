import torch

from mmseg.models.backbones.wavevit import WaveViT

device = 'cuda'

model = WaveViT().to(device)
x = torch.ones(4, 3, 224, 224).to(device)

output = model(x)

print(output.shape)
