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
from matplotlib.colors import LinearSegmentedColormap
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
                if dx == 0:
                    preview.append(image)
    return TensorDataset(torch.stack(images), torch.tensor(labels)), preview


def plot_results(metrics, confusion, preview):
    ink, muted = "#171A1F", "#747D88"
    colors = {
        "linear": "#8F949E", "mlp128": "#7CC8CE", "mlp256": "#0C9FAA",
        "mlp256slow": "#AF9274", "cnn": "#D69B23", "cnn_aug": "#E16B55",
    }
    labels = {
        "linear": "Linear", "mlp128": "MLP 128", "mlp256": "MLP 256",
        "mlp256slow": "MLP 256, lower LR", "cnn": "CNN", "cnn_aug": "CNN + aug.",
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for name, record in metrics["models"].items():
        epochs = [row["epoch"] for row in record["history"]]
        for ax, key in zip(axes, ("val_accuracy", "train_loss")):
            ax.plot(epochs, [row[key] for row in record["history"]],
                    color=colors[name], marker="o", markersize=4.5, linewidth=2.9,
                    label=labels[name])
    for ax, title in zip(axes, ("A  Validation accuracy", "B  Training loss")):
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 7))
        ax.grid(axis="y", color="#DFE5E9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#B7C2CA")
    axes[0].set_ylim(0.87, 1.005)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[1].set_ylim(bottom=0)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.25, wspace=0.22)
    fig.savefig(RESULTS / "learning_curves.png", dpi=260, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    teal_map = LinearSegmentedColormap.from_list(
        "wandb_teal", ["#F5FAFA", "#B8E8EA", "#51BFCA", "#0C9FAA", "#07515B"])
    image = ax.imshow(confusion, cmap=teal_map, vmin=0, vmax=confusion.max())
    for row in range(10):
        for col in range(10):
            count = confusion[row, col]
            if row == col or count >= 3:
                ax.text(col, row, str(count), ha="center", va="center",
                        color="white" if row == col else ink,
                        fontsize=9, fontweight="bold" if row == col else "normal")
    ax.set_title("Test-set confusion matrix", loc="left", fontweight="bold", pad=16)
    ax.set(xlabel="Predicted digit", ylabel="True digit",
           xticks=range(10), yticks=range(10))
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03, label="Number of images")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=260, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 10, figsize=(10.5, 3.3))
    for index, (ax, item) in enumerate(zip(axes.flat, preview)):
        ax.imshow(item, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set(color="#C8D3DB", linewidth=0.7)
        if index >= 20:
            ax.set_xlabel(str(index % 10), labelpad=5, fontweight="bold")
    for row, name in enumerate(("Sans", "Serif", "Mono")):
        axes[row, 0].set_ylabel(name, rotation=0, ha="right", va="center", labelpad=15)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.14, wspace=0.18, hspace=0.18)
    fig.savefig(RESULTS / "font_digits.png", dpi=260, facecolor="white")
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
    full_train = datasets.MNIST(DATA, train=True, download=True)
    indices = torch.randperm(60000, generator=torch.Generator().manual_seed(SEED)).tolist()
    # 三类模型共用同一份训练/验证划分。
    train_indices, valid_indices = indices[:54000], indices[54000:]
    if args.quick:
        train_indices, valid_indices = train_indices[:1024], valid_indices[:256]
    pixels = full_train.data[train_indices].float() / 255
    mean = pixels.mean().item()
    std = pixels.std(unbiased=False).item()
    del pixels
    print(f"训练图片像素统计：mean={mean:.8f}, std={std:.8f}", flush=True)
    basic = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    augmented = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1)),
        transforms.ToTensor(), transforms.Normalize((mean,), (std,)),
    ])
    full_train.transform = basic
    augmented_train = datasets.MNIST(DATA, train=True, transform=augmented, download=True)
    test = datasets.MNIST(DATA, train=False, transform=basic, download=True)
    if args.quick:
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
               "normalization": {"mean": mean, "std": std, "train_images": len(train_indices)},
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
