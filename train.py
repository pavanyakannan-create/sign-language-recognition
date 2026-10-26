import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SignDataset


class GRUClassifier(nn.Module):
    def __init__(self, in_dim=147, hidden=128, layers=2, n_classes=100, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 * 2, n_classes)

    def forward(self, x):                       # x: (batch, 48, 147)
        out, _ = self.gru(x)                    # (batch, 48, 256)
        feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.fc(self.drop(feat))


def augment(x):
    """Light augmentation: random scale + small jitter, keeping missing points at 0."""
    mask = (x != 0).float()
    scale = torch.empty(x.size(0), 1, 1).uniform_(0.9, 1.1)
    noise = torch.randn_like(x) * 0.01
    return (x * scale + noise) * mask


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    c1 = c5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        top5 = model(x).topk(5, dim=1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(dim=1).sum().item()
        n += len(y)
    return c1 / n, c5 / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(SignDataset("train"), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(SignDataset("val"), batch_size=args.batch)
    test_loader = DataLoader(SignDataset("test"), batch_size=args.batch)

    model = GRUClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    Path("checkpoints").mkdir(exist_ok=True)
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.augment:
                x = augment(x)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)

        val1, val5 = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d} | loss {total / len(train_loader.dataset):.3f} "
              f"| val top1 {val1:.3f} top5 {val5:.3f}")
        if val1 > best_val:
            best_val = val1
            torch.save(model.state_dict(), "checkpoints/gru_best.pt")

    model.load_state_dict(torch.load("checkpoints/gru_best.pt"))
    t1, t5 = evaluate(model, test_loader, device)
    print(f"\nBest val top1: {best_val:.3f}")
    print(f"TEST top1: {t1:.3f} | top5: {t5:.3f}")


if __name__ == "__main__":
    main()