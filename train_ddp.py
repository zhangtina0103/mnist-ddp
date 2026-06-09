import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from model import MNISTNet
from ddp import setup, cleanup, SimpleDDP
import time


def train_ddp(rank, world_size, epochs=3):
    setup(rank, world_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # DistributedSampler ensures each GPU sees different data
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=sampler,       # use sampler instead of shuffle=True
        num_workers=2,
        pin_memory=True        # faster CPU->GPU transfer
    )

    # model
    model = MNISTNet().to(rank)
    model = SimpleDDP(model, rank, world_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        # ensures different shuffling each epoch
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            # sync gradients across all GPUs - DDP
            model.sync_gradients()

            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            if batch_idx % 100 == 0 and rank == 0:
                print(f"rank {rank} | epoch {epoch+1} | batch {batch_idx}/{len(train_loader)} "
                      f"| loss {loss.item():.4f}")

        if rank == 0:
            acc = 100 * correct / total
            print(f"epoch {epoch+1} done | avg loss {total_loss/len(train_loader):.4f} "
                  f"| accuracy {acc:.2f}%")

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"found {world_size} GPUs")

    start = time.time()
    mp.spawn(
        train_ddp,
        args=(world_size, 3),
        nprocs=world_size,
        join=True
    )
    print(f"\ntotal time: {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()
