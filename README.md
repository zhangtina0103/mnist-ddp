# mnist-ddp

Here I implement distributed data parallel (DDP) from scratch using PyTorch distributed and NCCL, trained on MNIST across two L40 GPUs on Slurm (MIT Engaging cluster).
For my own conceptual understanding purposes, I used MNIST, but in reality it's a bit too small so AllReduce communication overhead between GPUs outweights compute savings but in theory DDP performs well on large models and datasets

## Details

- AllReduce: average gradients across all GPUs after each backward pass
- DistributedSampler: ensure each GPU see different data
- NCCL backend: NVIDIA's optimized GPU communication librarh

## Architecture

```
GPU 0                          GPU 1
─────                          ─────
load batch 0,2,4...            load batch 1,3,5...
forward pass                   forward pass
backward pass                  backward pass
      └──── AllReduce (NCCL) ────┘
         average gradients
weight update                  weight update
(identical)                    (identical)
```
