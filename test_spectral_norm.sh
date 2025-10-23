#!/usr/bin/env bash

#SBATCH --job-name=test_specnorm
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_specnorm_%j.out
#SBATCH --error=logs/test_specnorm_%j.err

mkdir -p $HOME/gan-miniproject/logs
cd $HOME/gan-miniproject
source venv/bin/activate
cd src

echo "Testing Spectral Normalization..."
date

python3 train_gan.py \
    --dataset mnist \
    --epochs 10 \
    --batch 128 \
    --opt adam \
    --spectral-norm \
    --save-every 200 \
    --save-loc test_specnorm \
    --seed 42

echo "Test completed!"
date