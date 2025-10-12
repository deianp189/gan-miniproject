from torchvision import datasets, transforms
ds = datasets.MNIST(root='./data', train=True, download=True,
                    transform=transforms.ToTensor())
x0,y0 = ds[0]
print('MNIST train size:', len(ds))
print('Sample shape:', tuple(x0.shape), 'Label:', y0)
