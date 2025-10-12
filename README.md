# Vanilla GAN (MNIST) — *“Max, it runs. Please don’t break it.”*

> **TL;DR**: Re-implementation of the OG GAN (Goodfellow et al., 2014) on MNIST using PyTorch.  
> It trains, it spits out digits, and the plumbing works. Your move, Max. 🫡

---

## What’s here

- **Dataset**: MNIST (60k train / 10k test), grayscale `28×28`.
- **Models**
  - **Generator (G)**: MLP ending with **Tanh** → outputs in `[-1, 1]`.
  - **Discriminator (D)**: MLP → single logit (trained with `BCEWithLogitsLoss`).
- **Training**: “Algorithm 1” vibe — **k=1** D step per G step; **non-saturating** loss for G (`maximize log D(G(z))`).
- **Optimizers**: `--opt sgd` (nostalgia) or `--opt adam` (sanity).
- **Outputs**
  - sample grids → `samples/step_XXXXXX.png`
  - checkpoints → `checkpoints/G_epoch_X.pt`, `checkpoints/D_epoch_X.pt`

Current quality = *recognizable digits with acne*. More epochs + Adam = skincare routine.

---

## Repo layout

```text
gan-miniproject/
├─ src/
│ └─ train_gan.py # main training script
├─ check/ # sanity scripts (env, mnist, loader, save)
├─ samples/ # generated images
├─ checkpoints/ # model weights
└─ data/MNIST/ # auto-downloaded dataset (torchvision)
```


---

## Setup (Windows + VS Code)

1. Open the folder in VS Code → **Terminal → New Terminal**.
2. Create/activate venv:
    ```powershell
    python -m venv .venv
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # run once if needed
    .\.venv\Scripts\Activate.ps1
    ```
3. Select interpreter in VS Code: `Ctrl+Shift+P` → `Python: Select Interpreter` → pick the `.venv` one.

**Install deps:**

```powershell
pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

**Quickstart (make pixels happen)**

Recommended (Adam, stable):

```powershell
python src/train_gan.py --opt adam --epochs 20 --batch-size 128
```

Optional (SGD, nostalgia):

```powershell
python src/train_gan.py --opt sgd --epochs 20 --batch-size 128 --lr 0.01
```

---

## Extras

- Run `python check/check_env.py` to confirm CUDA/PyTorch versions.
- Use `python check/check_loader.py` to ensure MNIST downloads cleanly.
- `python check/write_sample.py` dumps a sample grid without training.