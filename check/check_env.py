import sys, torch, torchvision
print('python:', sys.version.split()[0])
print('exe   :', sys.executable)
print('torch :', torch.__version__, '| tv:', torchvision.__version__)
print('cuda? :', torch.cuda.is_available())
print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')
