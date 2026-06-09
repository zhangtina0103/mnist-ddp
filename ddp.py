import torch
import torch.distributed as dist
import torch.nn as nn
import os


def setup(rank, world_size):
    """
    initialize the distributed process group
    rank:       which GPU am i (0, 1, 2, ...)
    world_size: total number of GPUs
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # init process group - every process calls this
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)
    print(f"rank {rank} initialized")


def cleanup():
    dist.destroy_process_group()


def allreduce_gradients(model):
    """
    average gradients across all GPUs
    this is core of DDP
    """
    world_size = dist.get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            # sum gradients across all GPUs
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            # divide by world_size to get average
            param.grad.data /= world_size


class SimpleDDP(nn.Module):
    """
    DDP wrapper from scratch
    wraps any model and syncs gradients after backward pass
    """
    def __init__(self, model, rank, world_size):
        super().__init__()
        self.model      = model
        self.rank       = rank
        self.world_size = world_size

        # broadcast initial weights from rank 0 to all GPUs
        # ensures all GPUs start with identical weights
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, x):
        return self.model(x)

    def sync_gradients(self):
        allreduce_gradients(self.model)
