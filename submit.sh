#!/bin/bash
#SBATCH --job-name=ddp_mnist
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# load modules
module load cuda/12.1
module load python/3.10

# create logs dir
mkdir -p logs

echo "running single GPU..."
python train_single.py

echo "running DDP..."
python train_ddp.py
