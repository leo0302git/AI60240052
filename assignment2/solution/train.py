"""MNIST 对比实验：运行 `python train.py` 可重现表格和图片。"""

import argparse
import gzip
import json
import random
import time
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = HERE / "results"
CHECKPOINTS = HERE / "checkpoints"
SEED = 2026
MEAN, STD = 0.1307, 0.3081


def prepare_data():
    """优先使用随作业下载的 mnist.zip，然后让 torchvision 正常读取 IDX。"""
    archive_path = ROOT / "mnist.zip"
    raw = DATA / "MNIST" / "raw"
    names = (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
    )
    if archive_path.exists() and not all((raw / name).exists() for name in names):
        raw.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive_path) as archive:
            for name in names:
                (raw / name).write_bytes(gzip.decompress(archive.read(f"mnist/{name}.gz")))


def model_for(kind, width=256):
    if kind == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if kind == "mlp":
        return nn.Sequential(
            nn.Flatten(), nn.Linear(784, width), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(width, 10),
        )
    if kind == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 10),
        )
    raise ValueError(kind)


def measures(confusion):
    correct = np.trace(confusion)
    # 行是真实类别，列是预测类别；逐类算 F1 后取平均。
    precision = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(correct / confusion.sum()), "macro_f1": float(f1.mean())}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((10, 10), dtype=np.int64)
    total_loss = 0.0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        pairs = (labels * 10 + logits.argmax(dim=1)).cpu().numpy()
        confusion += np.bincount(pairs, minlength=100).reshape(10, 10)
        count += len(labels)
    return {"loss": total_loss / count, **measures(confusion)}, confusion


def train_one(name, kind, width, lr, augmented, train_set, valid_loader, device, quick):
    torch.manual_seed(SEED)
    model = model_for(kind, width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    history = []
    best_accuracy = -1.0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, 2 if quick else 7):
        model.train()
        loss_sum = 0.0
        seen = 0
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        validation, _ = evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                        "train_accuracy": correct / seen, **{f"val_{k}": v for k, v in validation.items()}})
        print(f"{name} epoch {epoch}: train={correct / seen:.4f}, val={validation['accuracy']:.4f}", flush=True)
        if validation["accuracy"] > best_accuracy:
            # 只用验证集决定保存哪一轮的权重。
            best_accuracy = validation["accuracy"]
            stale = 0
            torch.save(model.state_dict(), CHECKPOINTS / f"{name}.pt")
        else:
            stale += 1
            if stale >= 2:
                break
    model.load_state_dict(torch.load(CHECKPOINTS / f"{name}.pt", map_location=device, weights_only=True))
    return model, {"kind": kind, "width": width if kind == "mlp" else None,
                   "learning_rate": lr, "augmentation": augmented,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "seconds": round(time.perf_counter() - started, 2),
                   "best_val_accuracy": best_accuracy, "history": history}


def font_digits(transform):
    """自制印刷字体数字，测试模型离开手写数据分布后的表现。"""
    from matplotlib import font_manager

    fonts = [font_manager.findfont(name) for name in ("DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono")]
    images, labels, preview = [], [], []
    for font_path in fonts:
        font = ImageFont.truetype(font_path, 25)
        for digit in range(10):
            for dx in (-2, 0, 2):
                image = Image.new("L", (28, 28), 0)
                draw = ImageDraw.Draw(image)
                box = draw.textbbox((0, 0), str(digit), font=font)
                x = (28 - (box[2] - box[0])) / 2 - box[0] + dx
                y = (28 - (box[3] - box[1])) / 2 - box[1]
                draw.text((x, y), str(digit), font=font, fill=255)
                images.append(transform(image))
                labels.append(digit)
                if font_path == fonts[0] and dx == 0:
                    preview.append(image)
    return TensorDataset(torch.stack(images), torch.tensor(labels)), preview


def plot_results(metrics, confusion, preview):
    names = list(metrics["models"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        epochs = [row["epoch"] for row in metrics["models"][name]["history"]]
        axes[0].plot(epochs, [row["val_accuracy"] for row in metrics["models"][name]["history"]], label=name)
        axes[1].plot(epochs, [row["train_loss"] for row in metrics["models"][name]["history"]], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0.8, 1.0))
    axes[1].set(xlabel="Epoch", ylabel="Training loss")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xlabel="Predicted digit", ylabel="True digit", xticks=range(10), yticks=range(10))
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 10, figsize=(10, 1.5))
    for digit, (ax, item) in enumerate(zip(axes, preview)):
        ax.imshow(item, cmap="gray", vmin=0, vmax=255)
        ax.set_title(str(digit))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "font_digits.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--quick", action="store_true", help="少量数据跑一轮，仅检查代码")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    prepare_data()
    RESULTS.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    basic = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    augmented = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1)),
        transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,)),
    ])
    full_train = datasets.MNIST(DATA, train=True, transform=basic, download=True)
    augmented_train = datasets.MNIST(DATA, train=True, transform=augmented, download=True)
    test = datasets.MNIST(DATA, train=False, transform=basic, download=True)
    indices = torch.randperm(60000, generator=torch.Generator().manual_seed(SEED)).tolist()
    # 三类模型共用同一份训练/验证划分。
    train_indices, valid_indices = indices[:54000], indices[54000:]
    if args.quick:
        train_indices, valid_indices = train_indices[:1024], valid_indices[:256]
        test = Subset(test, range(512))
    valid_loader = DataLoader(Subset(full_train, valid_indices), batch_size=512)
    test_loader = DataLoader(test, batch_size=512)
    shifted = datasets.MNIST(DATA, train=False, download=True,
                             transform=transforms.Compose([
                                 lambda image: TF.affine(image, angle=15, translate=(2, -2), scale=1, shear=0),
                                 basic,
                             ]))
    if args.quick:
        shifted = Subset(shifted, range(512))
    shifted_loader = DataLoader(shifted, batch_size=512)
    font_set, preview = font_digits(basic)
    font_loader = DataLoader(font_set, batch_size=90)
    trials = (
        ("linear", "linear", 0, 0.001, False),
        ("mlp128", "mlp", 128, 0.001, False),
        ("mlp256", "mlp", 256, 0.001, False),
        ("mlp256slow", "mlp", 256, 0.0003, False),
        ("cnn", "cnn", 0, 0.001, False),
        ("cnn_aug", "cnn", 0, 0.001, True),
    )
    metrics = {"seed": SEED, "device": device, "torch": torch.__version__,
               "train_count": len(train_indices), "validation_count": len(valid_indices),
               "test_count": len(test), "models": {}}
    for name, kind, width, lr, use_aug in trials:
        training_set = Subset(augmented_train if use_aug else full_train, train_indices)
        model, record = train_one(name, kind, width, lr, use_aug, training_set,
                                  valid_loader, device, args.quick)
        for label, loader in (("test", test_loader), ("shifted", shifted_loader), ("font", font_loader)):
            scores, confusion = evaluate(model, loader, device)
            record[label] = scores
            if name == "cnn_aug" and label == "test":
                best_confusion = confusion
        metrics["models"][name] = record
        print(f"{name}: test accuracy={record['test']['accuracy']:.4f}, macro F1={record['test']['macro_f1']:.4f}", flush=True)
    metrics["selected_mlp"] = max(("mlp128", "mlp256", "mlp256slow"),
                                  key=lambda name: metrics["models"][name]["best_val_accuracy"])
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(metrics, best_confusion, preview)
    print("结果已写入", RESULTS)


if __name__ == "__main__":
    main()
