# -*- coding: utf-8 -*-
# Vanilla GAN (MNIST)

import argparse, os, math, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# Utils or something idk

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# Models

class Generator(nn.Module):
    # Latent z -> 784 with Tanh (I don't fucking know what Tanh is)
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.ReLU(True),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    # 784 -> logit(1)
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)  # logits
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# Training

def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Data: [0,1] -> Normalize to [-1,1] in order to work with Tanh
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=2)

    # More models or something idk
    z_dim = args.zdim
    img_dim = 28*28
    G = Generator(z_dim, img_dim).to(device)
    D = Discriminator(img_dim).to(device)

    # Optimization (paper-like SGD or "stable" Adam)
    if args.opt.lower() == "sgd":
        optG = optim.SGD(G.parameters(), lr=args.lrG, momentum=args.momentum, nesterov=False)
        optD = optim.SGD(D.parameters(), lr=args.lrD, momentum=args.momentum, nesterov=False)
    elif args.opt.lower() == "adam":
        optG = optim.Adam(G.parameters(), lr=args.lrG, betas=(0.5, 0.999))
        optD = optim.Adam(D.parameters(), lr=args.lrD, betas=(0.5, 0.999))
    else:
        raise ValueError("--opt must be 'sgd' o 'adam'")

    bce = nn.BCEWithLogitsLoss()

    # outputs
    ensure_dir("samples"); ensure_dir("checkpoints")

    # Fixed noise for tracking
    z_fixed = torch.randn(64, z_dim, device=device)

    global_step = 0
    for epoch in range(1, args.epochs+1):
        for xb, _ in dl:
            xb = xb.to(device)  # [-1,1]
            m = xb.size(0)

            # k steps Discriminator (normally k=1)
            for _ in range(args.k_steps_d):
                z = torch.randn(m, z_dim, device=device)
                fake = G(z).detach()  # do not update G here

                # D(x): real ~ 1 (label smoothing opcional 0.9)
                logits_real = D(xb)
                y_real = torch.full((m,1), args.real_label, device=device)
                loss_real = bce(logits_real, y_real)

                # D(G(z)): fake ~ 0
                logits_fake = D(fake)
                y_fake = torch.zeros(m,1, device=device)
                loss_fake = bce(logits_fake, y_fake)

                lossD = loss_real + loss_fake
                optD.zero_grad(); lossD.backward(); optD.step()

            # 1 step Generator (non-saturating: maximize log D(G(z)))
            z = torch.randn(m, z_dim, device=device)
            fake = G(z)
            logits_fake = D(fake)
            # goal of G is to have D(fake) ~ 1
            y_gen = torch.ones(m,1, device=device)
            lossG = bce(logits_fake, y_gen)

            optG.zero_grad(); lossG.backward(); optG.step()

            # Logging amd samples
            if global_step % args.save_every == 0:
                G.eval()
                with torch.no_grad():
                    grid = utils.make_grid(
                        G(z_fixed).view(-1,1,28,28),
                        nrow=8, normalize=True, value_range=(-1,1)
                    )
                    out_path = f"samples/step_{global_step:06d}.png"
                    utils.save_image(grid, out_path)
                G.train()
                print(f"[ep {epoch:02d} | step {global_step}] "
                      f"lossD={lossD.item():.3f} lossG={lossG.item():.3f} -> {out_path}")

            global_step += 1

        # checkpoint per epoch
        torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch}.pt")
        torch.save(D.state_dict(), f"checkpoints/D_epoch_{epoch}.pt")
        print(f"Saved checkpoints for epoch {epoch}")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vanilla GAN MNIST (paper-like)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--zdim", type=int, default=100)
    ap.add_argument("--k-steps-d", type=int, default=1)
    ap.add_argument("--opt", type=str, default="sgd", choices=["sgd","adam"])
    ap.add_argument("--lrG", type=float, default=2e-3)
    ap.add_argument("--lrD", type=float, default=2e-3)
    ap.add_argument("--momentum", type=float, default=0.5)
    ap.add_argument("--real-label", type=float, default=0.9, help="label smoothing para reales")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)
