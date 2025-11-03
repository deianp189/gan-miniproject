# -*- coding: utf-8 -*-
# Vanilla GAN (MNIST)

import argparse, os, math, random, time, glob

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm

# Utils or something idk

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clear_old_samples(samples_dir="samples", pattern="step_*.png"):
    #This chunk of code removes old sample images to save disk space.
    if not os.path.isdir(samples_dir):
        return
    for f in glob.glob(os.path.join(samples_dir, pattern)):
        try:
            os.remove(f)
        except OSError:
            pass

# helpers for configurable nets:
def parse_hidden(s, default):
    if s is None or s.strip() == "":
        return default
    return [int(x) for x in s.split(",")]

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_pieces=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        self.linear = nn.Linear(in_features, out_features * num_pieces)
    
    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), self.out_features, self.num_pieces)
        out, _ = torch.max(out, dim=2)
        return out
    
def add_instance_noise(x, noise_std=0.1):
    """Add instance noise to prevent discriminator overconfidence"""
    if noise_std > 0:
        noise = torch.randn_like(x) * noise_std
        return x + noise
    return x

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
        return nn.SiLU(inplace=True)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "maxout":
        return "maxout"
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

def init_weights_cnn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    

# Models
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28, hidden=(256,512,1024),
                 act="relu", use_bn=True):
        super().__init__()
        layers = []
        in_f = z_dim
        
        for h in hidden:
            act_fn = get_act(act)
            if act_fn == "maxout":
                layers += [Maxout(in_f, h, num_pieces=2)]
            else:
                layers += [nn.Linear(in_f, h)]
                if use_bn and act.lower() != "selu":
                    layers += [nn.BatchNorm1d(h)]
                layers += [act_fn]
            in_f = h
            
        layers += [nn.Linear(in_f, img_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.act_name = act
        self.apply(lambda m: init_weights(m, act))

    def forward(self, z):
        return self.net(z)
    
class GeneratorCNN(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, img_size=28, act="relu", use_bn=True):
        super().__init__()

        if img_size == 28: # MNIST
            init_size = 7
            self.init_channels = 256
        elif img_size == 32: # CIFAR-10
            init_size = 4
            self.init_channels = 512
        elif img_size == 64: # CelebA
            init_size = 4
            self.init_channels = 512

        self.fc = nn.Linear(z_dim, self.init_channels * init_size * init_size)
        self.init_size = init_size

        layers = []
        in_channels = self.init_channels

        num_upsample = int(np.log2(img_size // init_size))

        for i in range(num_upsample):
            out_channels = in_channels // 2 if i < num_upsample - 1 else 64
            layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),]

            if use_bn and i < num_upsample - 1:
                layers += [nn.BatchNorm2d(out_channels)]

            layers += [get_act(act)]
            in_channels = out_channels

        layers += [nn.Conv2d(64, 32, kernel_size=3, padding=1),
                   nn.BatchNorm2d(32),
                   get_act(act),
                   nn.Conv2d(32, img_channels, kernel_size=3, padding=1),
                   nn.Tanh()]

        self.conv_blocks = nn.Sequential(*layers)

        self.apply(init_weights_cnn)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), self.init_channels, self.init_size, self.init_size)
        print(f"After reshape: {out.shape}")

        out = self.conv_blocks(out)
        print(f"Generator output: {out.shape}")

        return out

class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28, hidden=(512,256),
                 act="lrelu", use_bn=False, dropout_p=0.0, spectral_norm_d=False):
        super().__init__()
        layers = []
        in_f = img_dim
        
        for h in hidden:
            act_fn = get_act(act)
            if act_fn == "maxout":
                maxout_layer = Maxout(in_f, h, num_pieces=2)
                if spectral_norm_d:
                    maxout_layer.linear = spectral_norm(maxout_layer.linear)
                layers += [maxout_layer]
            else:
                linear = nn.Linear(in_f, h)
                if spectral_norm_d:
                    linear = spectral_norm(linear)
                layers += [linear]
                if use_bn and act.lower() != "selu":
                    layers += [nn.BatchNorm1d(h)]
                layers += [act_fn]
            if dropout_p > 0:
                layers += [nn.Dropout(p=dropout_p)]
            in_f = h
        
        # Final layer
        final_linear = nn.Linear(in_f, 1)
        if spectral_norm_d:
            final_linear = spectral_norm(final_linear)
        layers += [final_linear]
        
        self.net = nn.Sequential(*layers)
        self.apply(lambda m: init_weights(m, act))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class DiscriminatorCNN(nn.Module):
    def __init__(self, img_channels=1, img_size=28, act="lrelu", use_bn=False, dropout_p=0.0, spectral_norm_d=False):
        super().__init__()

        layers = []
        in_channels = img_channels
        base_channels = 64

        num_downsample = int(np.log2(img_size)) - 2

        for i in range(num_downsample):
            out_channels = base_channels * (2 ** i)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if spectral_norm_d:
                conv = spectral_norm(conv)

            layers += [conv]

            if use_bn and i > 0:
                layers += [nn.BatchNorm2d(out_channels)]

            layers += [get_act(act)]

            if dropout_p > 0:
                layers += [nn.Dropout2d(p=dropout_p)]

            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        final_linear = nn.Linear(out_channels, 1)
        if spectral_norm_d:
            final_linear = spectral_norm(final_linear)
        self.fc = final_linear

        self.apply(init_weights_cnn)

    def forward(self, x):
        out = self.conv_blocks(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)

        return self.fc(out)

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

    ds, img_dim, channels = get_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, 
                    num_workers=6, pin_memory=True, prefetch_factor=2)

    z_dim = args.zdim
    g_hidden = parse_hidden(args.g_hidden, [256,512,1024])
    d_hidden = parse_hidden(args.d_hidden, [512,256])

    if args.dataset.lower() == "mnist":
        img_size = 28
    elif args.dataset.lower() == "cifar10":
        img_size = 32
    elif args.dataset.lower() == "celeba":
        img_size = 64

    if args.arch.lower() == "mlp":
        G = Generator(z_dim, img_dim,
                    hidden=g_hidden,
                    act=(args.g_act or "relu"),
                    use_bn=args.g_bn).to(device)

        D = Discriminator(img_dim,
                    hidden=d_hidden,
                    act=(args.d_act or "lrelu"),
                    use_bn=args.d_bn,
                    dropout_p=args.d_dropout,
                    spectral_norm_d=args.spectral_norm).to(device)
        
    elif args.arch.lower() == "cnn":
        G = GeneratorCNN(z_dim, channels, img_size, act=args.g_act or "relu", use_bn=args.g_bn).to(device)

        D = DiscriminatorCNN(channels, img_size, act=args.d_act or "lrelu", use_bn=args.d_bn, dropout_p=args.d_dropout, spectral_norm_d=args.spectral_norm).to(device)

    betas = args.betas

    if args.opt.lower() == "sgd":
        optG = optim.SGD(G.parameters(), lr=args.lrG, momentum=args.momentum, weight_decay=args.weight_decay)
        optD = optim.SGD(D.parameters(), lr=args.lrD, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt.lower() == "adam":
        optG = optim.Adam(G.parameters(), lr=args.lrG, betas=betas, weight_decay=args.weight_decay)
        optD = optim.Adam(D.parameters(), lr=args.lrD, betas=betas, weight_decay=args.weight_decay)
    elif args.opt.lower() == "adabelief":
        optG = AdaBelief(G.parameters(), lr=args.lrG, betas=betas, eps=1e-16, weight_decay=1e-4)
        optD = AdaBelief(D.parameters(), lr=args.lrD, betas=betas, eps=1e-16, weight_decay=1e-4)
    elif args.opt.lower() == "lion":
        optG = Lion(G.parameters(), lr=args.lrG, betas=betas, weight_decay=1e-4)
        optD = Lion(D.parameters(), lr=args.lrD, betas=betas, weight_decay=1e-4)
    else:
        raise ValueError("--opt must be 'sgd', 'adam', 'adabelief' or 'lion'")

    bce = nn.BCEWithLogitsLoss()

    ensure_dir("samples"); ensure_dir("checkpoints")

    if args.overwrite_samples:
        clear_old_samples("samples", "step_*.png")


    # Fixed noise for tracking
    z_fixed = torch.randn(64, z_dim, device=device)

    # Determine image shape for reshaping based on dataset
    if args.dataset.lower() == "mnist":
        img_shape = (-1, 1, 28, 28)
    elif args.dataset.lower() == "cifar10":
        img_shape = (-1, 3, 32, 32)
    elif args.dataset.lower() == "celeba":
        img_shape = (-1, 3, 64, 64)

    kG = args.k_steps_g if args.k_steps_g is not None else 1

    losses_D = []
    losses_D_batch = []
    losses_G = []
    losses_G_batch = []
    global_step = 0

    print(f"Running with args: {args}")
    for epoch in range(1, args.epochs+1):
        G.train()
        D.train()
        start_time = time.time()
        for xb, _ in dl:
            xb = xb.to(device)  # [-1,1]
            m = xb.size(0)

            #The batch loss should be cleared each iteration
            losses_D_batch = []
            losses_G_batch = []

            # k steps Discriminator (normally k=1)
            for _ in range(args.k_steps_d):
                z = torch.randn(m, z_dim, device=device)
                fake = G(z).detach()  # do not update G here

                # Add instance noise if enabled
                xb_noisy = add_instance_noise(xb, args.instance_noise)
                fake_noisy = add_instance_noise(fake, args.instance_noise)

                # D(x): real ~ 1 (with label smoothing)
                logits_real = D(xb_noisy)
                y_real = torch.full((m,1), args.real_label, device=device)
                loss_real = bce(logits_real, y_real)

                # D(G(z)): fake ~ 0 (with optional two-sided smoothing)
                logits_fake = D(fake_noisy)
                y_fake = torch.full((m,1), args.fake_label, device=device)
                loss_fake = bce(logits_fake, y_fake)

                lossD = loss_real + loss_fake
                optD.zero_grad(); lossD.backward(); optD.step()

                losses_D_batch.append(lossD.item())
            
            losses_D.append(np.mean(losses_D_batch))

            for _ in range(kG):
                z = torch.randn(m, z_dim, device=device)
                fake = G(z)
                logits_fake = D(fake)
                y_gen = torch.ones(m,1, device=device)
                lossG = bce(logits_fake, y_gen)
                optG.zero_grad(); lossG.backward(); optG.step()
                losses_G_batch.append(lossG.item())

            losses_G.append(np.mean(losses_G_batch))

            # Logging and samples                
            if global_step % args.save_every == 0:
                G.eval()
                with torch.no_grad():
                    fake_fixed = G(z_fixed).view(*img_shape)
                    grid = utils.make_grid(fake_fixed, nrow=8, normalize=True, value_range=(-1,1))
                    out_path = f"samples/step_{global_step:06d}.png"
                    utils.save_image(grid, out_path)
                    utils.save_image(grid, "samples/latest.png")
                G.train()
                print(f"[ep {epoch:02d} | step {global_step}] "
                      f"lossD_avg={losses_D[-1]:.3f} lossG_avg={losses_G[-1]:.3f} -> {out_path}")    

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

        dataset_name = args.dataset if hasattr(args, 'dataset') else 'mnist'
        folder_name = f"{dataset_name}_e{args.epochs}_b{args.batch}_lrG{args.lrG}_lrD{args.lrD}_s{args.seed}"

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        arch_path = args.arch.lower()
        opt_path = args.opt.lower()
        sub_dir = args.save_loc.lower()

        results_dir = os.path.join(project_root, "results", arch_path, opt_path, sub_dir, folder_name)
        ensure_dir(results_dir)

        image_path = os.path.join(results_dir, "final_sample.png")
        utils.save_image(grid, image_path)

        # Save plot in the same folder
        plot_path = os.path.join(results_dir, "loss_plot.pdf")

        ld = np.array(losses_D, dtype=np.float32)
        lg = np.array(losses_G, dtype=np.float32)

        both = np.concatenate([ld, lg]) if len(ld) and len(lg) else np.array([1.0])
        p95 = float(np.percentile(both, 95))
        ymax = max(3.0, p95)

        def moving_avg(x, w=101):
            if len(x) < w: 
                return None
            kernel = np.ones(w, dtype=np.float32) / w
            return np.convolve(x, kernel, mode='same')

        ld_avg = moving_avg(ld, w=101)
        lg_avg = moving_avg(lg, w=101)

        plt.figure(figsize=(10, 5))
        plt.plot(ld, label="Loss D", alpha=0.35, linewidth=1.0)
        plt.plot(lg, label="Loss G", alpha=0.35, linewidth=1.0, linestyle='--')

        # Smoothing curves
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

        print(f"Sample image saved to {image_path}\nLoss plot saved to {plot_path}")

    print("Done.")

def apply_preset(args):
    p = (args.preset or "baseline").lower()
    
    if args.dataset.lower() == "mnist":
        base_g = [256, 512, 1024]
        base_d = [512, 256]
    elif args.dataset.lower() == "cifar10":
        base_g = [512, 1024, 2048]
        base_d = [1024, 512, 256]
    elif args.dataset.lower() == "celeba":
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
        if args.g_hidden is None: 
            args.g_hidden = ",".join(map(str, [b*2 for b in base_g[:-1]] + [base_g[-1]]))
        if args.d_hidden is None: 
            args.d_hidden = ",".join(map(str, base_d))
        if args.g_act is None:    args.g_act = "silu"
        if args.d_act is None:    args.d_act = "lrelu"
        if args.k_steps_g is None: args.k_steps_g = 2
        if args.lrG == 2e-3 and args.lrD == 2e-3:
            args.lrG, args.lrD = 2e-3, 1e-3
            
    elif p == "strongerd":
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
    ap.add_argument("--overwrite-samples", action="store_true", help="Deletes step_*.png that already exists in ./samples at the start")

    # Regularization arguments
    ap.add_argument("--spectral-norm", action="store_true", help="Use spectral normalization in discriminator")
    ap.add_argument("--instance-noise", type=float, default=0.0, help="Add instance noise to inputs (std, 0 to disable)")
    ap.add_argument("--fake-label", type=float, default=0.0, help="Two-sided label smoothing for fake (0.1 makes training softer)")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay (regularization)")
    ap.add_argument("--save-loc", type=str, default="default")
    ap.add_argument("--preset", type=str, default="baseline", choices=["baseline","strongerG","strongerD"])
    ap.add_argument("--g-hidden", type=str, default=None, help="comma list e.g. 256,512,1024")
    ap.add_argument("--d-hidden", type=str, default=None, help="comma list e.g. 512,256")
    ap.add_argument("--g-act", type=str, default=None, choices=["relu","leakyrelu","elu","selu","silu","sigmoid","maxout"])
    ap.add_argument("--d-act", type=str, default=None, choices=["relu","leakyrelu","elu","selu","silu","sigmoid","maxout"])
    ap.add_argument("--g-bn", action="store_true", help="enable BatchNorm in G")
    ap.add_argument("--d-bn", action="store_true", help="enable BatchNorm in D")
    ap.add_argument("--d-dropout", type=float, default=0.0, help="dropout prob in D (0 to disable)")
    ap.add_argument("--arch", type=str, default="mlp", choices=["mlp", "cnn"], help="Architecture type: MLP or CNN")

    args = ap.parse_args()
    apply_preset(args)
    train(args)
