# main.py
import argparse
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model.resnet import resnet18

GLOBAL_CFG = import_module("utils.global").CONFIG


def get_loader(dataset, data_dir, batch_size, num_workers=2):
    if dataset == "cifar10":
        num_classes = 10
        ds_class = torchvision.datasets.CIFAR10
    else:
        num_classes = 100
        ds_class = torchvision.datasets.CIFAR100

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    trainset = ds_class(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = ds_class(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader, num_classes


def train(epoch, model, trainloader, optimizer, criterion, device):
    model.train()
    total, correct = 0, 0
    total_loss = 0

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    acc = correct / total * 100
    avg_loss = total_loss / total
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    return avg_loss, acc


@torch.no_grad()
def test(epoch, model, testloader, criterion, device):
    model.eval()
    total, correct = 0, 0
    total_loss = 0

    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    acc = correct / total * 100
    avg_loss = total_loss / total
    print(f"Epoch {epoch} | Test Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--data_dir", type=str, default=str(GLOBAL_CFG.data_root))
    parser.add_argument("--batch_size", type=int, default=GLOBAL_CFG.default_batch_size)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=GLOBAL_CFG.num_workers)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(GLOBAL_CFG.global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_CFG.global_seed)
    print("Using device:", device)

    trainloader, testloader, num_classes = get_loader(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )

    model = resnet18(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, trainloader, optimizer, criterion, device)
        test(epoch, model, testloader, criterion, device)
        scheduler.step()


if __name__ == "__main__":
    main()
