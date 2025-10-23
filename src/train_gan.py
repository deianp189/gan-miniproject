# -*- coding: utf-8 -*-
# Vanilla GAN (MNIST)

import argparse, os, math, random, time

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import scipy.ndimage as ndi
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# Utils or something idk

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# helpers for configurable nets:
def parse_hidden(s, default):
    if s is None or s.strip() == "":
        return default
    return [int(x) for x in s.split(",")]

def get_act(name, leak=0.2):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("lrelu", "leakyrelu"):
        return nn.LeakyReLU(leak, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "selu":
        return nn.SELU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)  # swish
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")

    
def init_weights(m, act_name="relu"):
    if isinstance(m, nn.Linear):
        a = act_name.lower()
        if a == "selu":
            # LeCun normal is recommended for SELU
            nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(1.0/m.in_features))
        elif a in ("relu","lrelu","elu","silu"):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Models

# --- Models (configurable) ---

class Generator(nn.Module):
    # z -> 784
    def __init__(self, z_dim=100, img_dim=28*28, hidden=(256,512,1024),
                 act="relu", use_bn=True):
        super().__init__()
        layers = []
        in_f = z_dim
        for h in hidden:
            layers += [nn.Linear(in_f, h)]
            if use_bn and act.lower() != "selu":  # SELU prefers no BN
                layers += [nn.BatchNorm1d(h)]
            layers += [get_act(act)]
            in_f = h
        layers += [nn.Linear(in_f, img_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.act_name = act
        self.apply(lambda m: init_weights(m, act))

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    # 784 -> logits(1)
    def __init__(self, img_dim=28*28, hidden=(512,256),
                 act="lrelu", use_bn=False, dropout_p=0.0):
        super().__init__()
        layers = []
        in_f = img_dim
        for h in hidden:
            layers += [nn.Linear(in_f, h)]
            if use_bn and act.lower() != "selu":
                layers += [nn.BatchNorm1d(h)]
            layers += [get_act(act)]
            if dropout_p > 0:
                layers += [nn.Dropout(p=dropout_p)]
            in_f = h
        layers += [nn.Linear(in_f, 1)]  # logits
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: init_weights(m, act))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# Training

def get_dataset(args):
    if args.dataset.lower() == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        img_dim = 28 * 28
        channels = 1
        
    elif args.dataset.lower() == "cifar10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
        img_dim = 32 * 32 * 3
        channels = 3

    elif args.dataset.lower() == "celeba":
        tfm = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ds = datasets.CelebA(root="./data", split='train', download=True, transform=tfm)
        img_dim = 64 * 64 * 3
        channels = 3
    
    return ds, img_dim, channels

def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # Get dataset
    ds, img_dim, channels = get_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, 
                    num_workers=6, pin_memory=True, prefetch_factor=2)

    # Models
    z_dim = args.zdim
    g_hidden = parse_hidden(args.g_hidden, [256,512,1024])
    d_hidden = parse_hidden(args.d_hidden, [512,256])

    G = Generator(z_dim, img_dim,
                hidden=g_hidden,
                act=(args.g_act or "relu"),
                use_bn=args.g_bn).to(device)

    D = Discriminator(img_dim,
                    hidden=d_hidden,
                    act=(args.d_act or "lrelu"),
                    use_bn=args.d_bn,
                    dropout_p=args.d_dropout).to(device)


    betas = args.betas

    # Optimization (paper-like SGD or "stable" Adam)
    if args.opt.lower() == "sgd":
        optG = optim.SGD(G.parameters(), lr=args.lrG, momentum=args.momentum, nesterov=False)
        optD = optim.SGD(D.parameters(), lr=args.lrD, momentum=args.momentum, nesterov=False)
    elif args.opt.lower() == "adam":
        optG = optim.Adam(G.parameters(), lr=args.lrG, betas=betas)
        optD = optim.Adam(D.parameters(), lr=args.lrD, betas=betas)
    elif args.opt.lower() == "adabelief":
        optG = AdaBelief(G.parameters(), lr=args.lrG, betas=betas, eps=1e-16, weight_decay=1e-4)
        optD = AdaBelief(D.parameters(), lr=args.lrD, betas=betas, eps=1e-16, weight_decay=1e-4)
    elif args.opt.lower() == "lion":
        optG = Lion(G.parameters(), lr=args.lrG, betas=betas, weight_decay=1e-4)
        optD = Lion(D.parameters(), lr=args.lrD, betas=betas, weight_decay=1e-4)
    else:
        raise ValueError("--opt must be 'sgd', 'adam', 'adabelief' or 'lion'")

    bce = nn.BCEWithLogitsLoss()

    # outputs
    ensure_dir("samples"); ensure_dir("checkpoints")

    # Fixed noise for tracking
    # Fixed noise for tracking
    z_fixed = torch.randn(64, z_dim, device=device)

    # Determine image shape for reshaping based on dataset
    if args.dataset.lower() == "mnist":
        img_shape = (-1, 1, 28, 28)
    elif args.dataset.lower() == "cifar10":
        img_shape = (-1, 3, 32, 32)
    elif args.dataset.lower() == "celeba":
        img_shape = (-1, 3, 64, 64)

    losses_D = []
    losses_G = []
    global_step = 0

    print(f"Running with args: {args}")
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
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
            kG = args.k_steps_g if args.k_steps_g is not None else 1
            for _ in range(kG):
                z = torch.randn(m, z_dim, device=device)
                fake = G(z)
                logits_fake = D(fake)
                y_gen = torch.ones(m,1, device=device)
                lossG = bce(logits_fake, y_gen)
                optG.zero_grad(); lossG.backward(); optG.step()

            losses_D.append(lossD.item())
            losses_G.append(lossG.item())

            # Logging and samples
            # Logging and samples
            if global_step % args.save_every == 0:
                G.eval()
                with torch.no_grad():
                    grid = utils.make_grid(
                        G(z_fixed).view(*img_shape),
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
        print(f"Saved checkpoints for epoch {epoch} (took {time.time() - start_time:.1f} s)")
        #print(f"\n Args: Batch {args.batch} | zdim {args.zdim} | opt {args.opt} | k-steps-d {args.k_steps_d} | lr G/D {args.lrG}/{args.lrD}")
        #print(f"\n momentum {args.momentum}") if args.opt=="sgd" else None

    G.eval()
    with torch.no_grad():
        grid = utils.make_grid(
            G(z_fixed).view(*img_shape),
            nrow=8, normalize=True, value_range=(-1,1)
        )
        params_str = "_".join(f"{k}{v}" for k, v in {
            'e': args.epochs,
            'b': args.batch,
            'z': args.zdim,
            'kD': args.k_steps_d,
            'kG': (args.k_steps_g if args.k_steps_g is not None else 1),
            'P': args.preset,
            'GA': (args.g_act or "relu"),
            'DA': (args.d_act or "lrelu"),
            'O': args.opt.lower(),
            'lrG': args.lrG,
            'lrD': args.lrD,
            'l': args.real_label,
            'm': args.momentum if args.opt.lower()=="sgd" else None,
            'be': f"{args.betas[0]}-{args.betas[1]}" if args.opt.lower()=="adam" else None,
            'S': args.seed
        }.items() if v is not None)


        # Get project root directory (parent of src/)

        # Create a shorter, cleaner folder name
        dataset_name = args.dataset if hasattr(args, 'dataset') else 'mnist'
        folder_name = f"{dataset_name}_e{args.epochs}_b{args.batch}_s{args.seed}"

        # Get project root directory (parent of src/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        opt_path = args.opt.lower()
        sub_dir = args.save_loc.lower()

        # Create a results folder with both sample and plot together
        results_dir = os.path.join(project_root, "results", opt_path, sub_dir, folder_name)
        ensure_dir(results_dir)

        # Save final sample image
        image_path = os.path.join(results_dir, "final_sample.png")
        utils.save_image(grid, image_path)

        # Save plot in the same folder
        plot_path = os.path.join(results_dir, "loss_plot.pdf")

        #plt.figure(figsize=(10, 5))
        #plt.plot(losses_D, label = "Loss D", alpha = 0.7, linewidth = 1.5, linestyle = '-')
        #plt.plot(losses_G, label = "Loss G", alpha = 0.7, linewidth = 1.5, linestyle = '--')
        #plt.xlabel("Iterations")
        #plt.ylabel("Loss")
        #plt.title(f"Losses - {params_str}")
        #plt.legend()
        #plt.grid(True, alpha = 0.3)
        #plt.tight_layout()
        #plt.savefig(plot_path, dpi = 150)
        #plt.close()

                # === PATCH: plotting robusto (quita outliers y añade media móvil) ===
        # Convertimos a arrays
        ld = np.array(losses_D, dtype=np.float32)
        lg = np.array(losses_G, dtype=np.float32)

        # Percentil 95 para no “reventar” el eje Y por un pico aislado
        both = np.concatenate([ld, lg]) if len(ld) and len(lg) else np.array([1.0])
        p95 = float(np.percentile(both, 95))
        ymax = max(3.0, p95)  # que no quede demasiado bajo

        # Media móvil simple para ver tendencia (si hay puntos suficientes)
        def moving_avg(x, w=101):
            if len(x) < w: 
                return None
            kernel = np.ones(w, dtype=np.float32) / w
            return np.convolve(x, kernel, mode='same')

        ld_avg = moving_avg(ld, w=101)
        lg_avg = moving_avg(lg, w=101)

        plt.figure(figsize=(10, 5))
        # Curvas crudas (suaves pero visibles)
        plt.plot(ld, label="Loss D", alpha=0.35, linewidth=1.0)
        plt.plot(lg, label="Loss G", alpha=0.35, linewidth=1.0, linestyle='--')

        # Curvas suavizadas (si existen)
        if ld_avg is not None: plt.plot(ld_avg, label="Loss D (avg)", linewidth=2.0)
        if lg_avg is not None: plt.plot(lg_avg, label="Loss G (avg)", linewidth=2.0, linestyle='--')

        plt.ylim(0, ymax)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"Losses - {params_str}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        # === PATCH END ===


        print(f"Sample image saved to {image_path}\nLoss plot saved to {plot_path}")

    print("Done.")

def apply_preset(args):
    p = (args.preset or "baseline").lower()
    
    # Adjust default hidden sizes based on dataset complexity
    if args.dataset.lower() == "mnist":
        base_g = [256, 512, 1024]
        base_d = [512, 256]
    elif args.dataset.lower() == "cifar10":
        base_g = [512, 1024, 2048]
        base_d = [1024, 512, 256]
    elif args.dataset.lower() == "celeba":
        # CelebA needs more capacity (64x64x3 = 12,288 dims)
        base_g = [1024, 2048, 4096]
        base_d = [2048, 1024, 512]
    else:
        base_g = [256, 512, 1024]
        base_d = [512, 256]
    
    if p == "baseline":
        if args.g_hidden is None: args.g_hidden = ",".join(map(str, base_g))
        if args.d_hidden is None: args.d_hidden = ",".join(map(str, base_d))
        if args.g_act is None:    args.g_act = "relu"
        if args.d_act is None:    args.d_act = "lrelu"
        if args.k_steps_g is None: args.k_steps_g = 1
        
    elif p == "strongerg":
        # Scale up from base
        if args.g_hidden is None: 
            args.g_hidden = ",".join(map(str, [b*2 for b in base_g[:-1]] + [base_g[-1]]))
        if args.d_hidden is None: 
            args.d_hidden = ",".join(map(str, base_d))
        if args.g_act is None:    args.g_act = "silu"
        if args.d_act is None:    args.d_act = "lrelu"
        if args.k_steps_g is None: args.k_steps_g = 2   # 2 pasos G/iter
        if args.lrG == 2e-3 and args.lrD == 2e-3:
            args.lrG, args.lrD = 2e-3, 1e-3
            
    elif p == "strongerd":
        # Scale down G, scale up D
        if args.g_hidden is None: 
            args.g_hidden = ",".join(map(str, [base_g[0]//2, base_g[0]//2, base_g[1]//2]))
        if args.d_hidden is None: 
            args.d_hidden = ",".join(map(str, [b*2 for b in base_d] + [base_d[-1]]))
        if args.g_act is None:    args.g_act = "relu"
        if args.d_act is None:    args.d_act = "lrelu"
        if args.k_steps_g is None: args.k_steps_g = 1
        args.k_steps_d = max(args.k_steps_d, 2)
        if args.lrG == 2e-3 and args.lrD == 2e-3:
            args.lrG, args.lrD = 1e-3, 2e-3
    else:
        raise ValueError(f"Unknown preset: {args.preset}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vanilla GAN MNIST (paper-like)")
    # training
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "celeba"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--zdim", type=int, default=100)
    ap.add_argument("--k-steps-d", type=int, default=1)
    ap.add_argument("--k-steps-g", type=int, default=None, help="G updates per iteration (default depends on preset)")
    ap.add_argument("--opt", type=str, default="sgd", choices=["sgd","adam","adabelief","lion"])
    ap.add_argument("--lrG", type=float, default=2e-3)
    ap.add_argument("--lrD", type=float, default=2e-3)
    ap.add_argument("--momentum", type=float, default=0.5)
    ap.add_argument("--betas", nargs=2, type=float, default=(0.5, 0.999), help="Betas for Adam/Lion/AdaBelief")
    ap.add_argument("--real-label", type=float, default=0.9, help="label smoothing for real")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")

    # results
    #ap.add_argument("--save-loc", type=str, default="./results")
    ap.add_argument("--save-loc", type=str, default="default")

    # NEW: model config + preset
    ap.add_argument("--preset", type=str, default="baseline",
                    choices=["baseline","strongerG","strongerD"])
    ap.add_argument("--g-hidden", type=str, default=None, help="comma list e.g. 256,512,1024")
    ap.add_argument("--d-hidden", type=str, default=None, help="comma list e.g. 512,256")
    ap.add_argument("--g-act", type=str, default=None,
                    choices=["relu","leakyrelu","elu","selu","silu","sigmoid"])
    ap.add_argument("--d-act", type=str, default=None,
                    choices=["relu","leakyrelu","elu","selu","silu","sigmoid"])
    ap.add_argument("--g-bn", action="store_true", help="enable BatchNorm in G")
    ap.add_argument("--d-bn", action="store_true", help="enable BatchNorm in D")
    ap.add_argument("--d-dropout", type=float, default=0.0, help="dropout prob in D (0 to disable)")

    args = ap.parse_args()
    apply_preset(args)
    train(args)
