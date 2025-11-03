#!/usr/bin/env bash

#SBATCH --job-name=DaGan
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/test_gan_%j.out
#SBATCH --error=logs/test_gan_%j.err

set +e

source aauipcv/bin/activate
cd src

echo "Testing..."
date

lrlist=(0.001 0.0002 0.0001)

for lr in "${lrlist[@]}"; do
	python3 train_gan.py \
	--dataset cifar10 \
	--epochs 100 \
	--opt adam \
	--arch cnn \
	--lrG $lr --lrD $lr

	python3 train_gan.py \
	--dataset celeba \
	--epochs 100 \
	--opt adam \
	--arch cnn \
	--lrG $lr --lrD $lr
done

echo "All tests completed!"
date
