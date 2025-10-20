from torch.utils.data import DataLoader
from torchvision import datasets, transforms
ds = datasets.MNIST(root='./data', train=True, download=True,
                    transform=transforms.ToTensor())
dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
xb, yb = next(iter(dl))
print('Batch X:', xb.shape, 'Batch y:', yb.shape)
print('Min/Max (0..1):', float(xb.min()), float(xb.max()))
