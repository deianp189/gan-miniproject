#!/usr/bin/env bash

#SBATCH --job-name=test_instnoise
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_instnoise_%j.out
#SBATCH --error=logs/test_instnoise_%j.err

mkdir -p $HOME/gan-miniproject/logs
cd $HOME/gan-miniproject
source venv/bin/activate
cd src

echo "Testing Instance Noise..."
date

python3 train_gan.py \
    --dataset mnist \
    --epochs 10 \
    --batch 128 \
    --opt adam \
    --instance-noise 0.1 \
    --save-every 200 \
    --save-loc test_noise \
    --seed 42

echo "Test completed!"
date