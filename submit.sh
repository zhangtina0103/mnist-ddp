#!/bin/bash
#SBATCH --job-name=ddp_mnist
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=mit_normal_gpu

module load cuda/12.1
module load python/3.10

mkdir -p logs

echo "========================================"
echo "single GPU training"
echo "========================================"
start_single=$SECONDS
python train_single.py
end_single=$SECONDS
single_time=$((end_single - start_single))
echo "single GPU time: ${single_time}s"

echo ""
echo "========================================"
echo "DDP training (2 x L40S)"
echo "========================================"
start_ddp=$SECONDS
python train_ddp.py
end_ddp=$SECONDS
ddp_time=$((end_ddp - start_ddp))
echo "DDP time: ${ddp_time}s"

echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"
echo "single GPU: ${single_time}s"
echo "DDP 2 GPU:  ${ddp_time}s"
python3 -c "print(f'speedup: {${single_time}/${ddp_time}:.2f}x')"
