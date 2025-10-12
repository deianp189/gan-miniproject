import torch
from torchvision.utils import save_image
x = torch.rand(64,1,28,28)*2-1  # [-1,1]
save_image(x, 'samples/smoke.png', nrow=8, normalize=True, value_range=(-1,1))
print('OK -> wrote: samples/smoke.png')
