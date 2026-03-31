import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import TransformerClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLPBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids).mean(dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class Metrics:
    loss: float
    acc: float


def load_split(path: Path):
    data = torch.load(path)
    return TensorDataset(data["input_ids"], data["labels"])


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return Metrics(loss=total_loss / total, acc=total_correct / total)


def plot_learning_curve(history, save_path: Path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_one_config(model_name, model, train_loader, val_loader, test_loader, num_epochs, lr, device, results_dir: Path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = -1.0
    best_model_path = results_dir / f"model_{model_name}.pt"

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(val_metrics.loss)
        history["train_acc"].append(train_metrics.acc)
        history["val_acc"].append(val_metrics.acc)

        if val_metrics.acc > best_val_acc:
            best_val_acc = val_metrics.acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:02d} | train_loss={train_metrics.loss:.4f} train_acc={train_metrics.acc:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.acc:.4f}"
        )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_final = run_epoch(model, train_loader, criterion, optimizer, device, train=False)
    val_final = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    test_final = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    plot_learning_curve(history, results_dir / f"learning_curve_{model_name}.png")

    return {
        "model_name": model_name,
        "train_accuracy": round(train_final.acc, 4),
        "val_accuracy": round(val_final.acc, 4),
        "test_accuracy": round(test_final.acc, 4),
        "final_train_loss": round(train_final.loss, 4),
        "best_model_path": str(best_model_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_loader = DataLoader(load_split(processed_dir / "train.pt"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(load_split(processed_dir / "val.pt"), batch_size=args.batch_size)
    test_loader = DataLoader(load_split(processed_dir / "test.pt"), batch_size=args.batch_size)

    configs = []
    if args.run_all:
        configs = [
            ("Transformer_d64_ff128", TransformerClassifier(meta["vocab_size"], 64, 128, meta["max_len"], meta["num_classes"])),
            ("Transformer_d128_ff256", TransformerClassifier(meta["vocab_size"], 128, 256, meta["max_len"], meta["num_classes"])),
            ("Transformer_d32_ff64", TransformerClassifier(meta["vocab_size"], 32, 64, meta["max_len"], meta["num_classes"])),
            ("MLPBaseline_d64", MLPBaseline(meta["vocab_size"], 64, meta["num_classes"])),
        ]
    else:
        configs = [
            (f"Transformer_d{args.d_model}_ff{args.d_ff}", TransformerClassifier(meta["vocab_size"], args.d_model, args.d_ff, meta["max_len"], meta["num_classes"]))
        ]

    summary = []
    for name, model in configs:
        print(f"\n===== Running {name} =====")
        result = train_one_config(name, model.to(device), train_loader, val_loader, test_loader, args.num_epochs, args.lr, device, results_dir)
        summary.append(result)

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for row in summary:
        print(row)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
