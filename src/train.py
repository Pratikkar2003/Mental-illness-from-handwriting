"""
train.py
--------
Usage:
    python src/train.py --data_dir data --epochs 25 --batch_size 8

Trains the attention-based transformer on data/train/<class>/*.png,
validates on data/val/<class>/*.png, saves the best checkpoint to
checkpoints/model.pt and the label map to checkpoints/labels.json.
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import HandwritingEmotionDataset, discover_classes
from model import HandwritingEmotionTransformer


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for point_seq, seq_mask, global_feats, labels in loader:
            point_seq = point_seq.to(device)
            seq_mask = seq_mask.to(device)
            global_feats = global_feats.to(device)
            labels = labels.to(device)

            logits = model(point_seq, seq_mask, global_feats)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(all_labels))
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro") if all_labels else 0.0
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--checkpoint_dir", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    classes = discover_classes(train_dir)
    print(f"Classes: {classes}")

    train_ds = HandwritingEmotionDataset(train_dir, classes=classes)
    val_ds = HandwritingEmotionDataset(val_dir, classes=classes)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = HandwritingEmotionTransformer(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pt"))
            with open(os.path.join(args.checkpoint_dir, "labels.json"), "w") as f:
                json.dump(classes, f)

    print(f"Best val macro-F1: {best_val_f1:.3f}")
    print(f"Checkpoint saved to: {os.path.join(args.checkpoint_dir, 'model.pt')}")
    print(
        "\nReminder: if you trained on the bundled synthetic demo data, this "
        "accuracy reflects separability of made-up shapes only - it is not a "
        "real emotion-detection accuracy. Retrain on real labeled handwriting "
        "for a number that means anything."
    )


if __name__ == "__main__":
    main()
