#!/usr/bin/env bash

#SBATCH --job-name=test_maxout
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_maxout_%j.out
#SBATCH --error=logs/test_maxout_%j.err

mkdir -p $HOME/gan-miniproject/logs
cd $HOME/gan-miniproject
source venv/bin/activate
cd src

echo "Testing Maxout activation on MNIST..."
date

python3 train_gan.py \
    --dataset mnist \
    --epochs 5 \
    --batch 128 \
    --opt adam \
    --d-act maxout \
    --save-every 200 \
    --save-loc test_maxout \
    --seed 42

echo "Maxout test completed!"
date