import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTNet


def train(epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training on {device}")

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )

    # model, loss, optimizer
    model     = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"epoch {epoch+1} | batch {batch_idx}/{len(train_loader)} "
                      f"| loss {loss.item():.4f}")

        acc = 100 * correct / total
        print(f"epoch {epoch+1} done | avg loss {total_loss/len(train_loader):.4f} "
              f"| accuracy {acc:.2f}%")


if __name__ == '__main__':
    import time
    start = time.time()
    train()
    print(f"\ntotal time: {time.time() - start:.2f}s")
