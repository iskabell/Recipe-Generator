"""
Preliminary ingredient classifier for the Recipe-Generator project.

Trains and compares two models on Vaidik's cleaned Grocery Store dataset
(43 coarse ingredient groups), following PROPOSAL.md:

  1. A small CNN trained from scratch (learning baseline)
  2. A pretrained ResNet18 with a new final layer (transfer learning,
     backbone frozen)

Run from the repository root (with the virtual environment turned on):

    python train_ingredient_model.py --quick     # 1-minute smoke test
    python train_ingredient_model.py             # full run

IMPORTANT NOTES ABOUT THE DATA SPLITS
  * The original TEST set is never used for training or model selection.
    It is evaluated once per model at the very end.
  * The original validation set is missing 6 of the 43 classes (see
    Data/grocery_store/processed/summary.json), so it is NOT used here.
  * Instead, a DEVELOPMENT split is carved out of the original TRAINING
    images only (stratified by class). It is used to pick the best epoch.
    This is a proposal for the team to review, as PROPOSAL.md asks.
  * Caveat: photos of the same product from the same scene may land on
    both sides of the dev split, so dev accuracy can look better than it
    really is. Trust the test numbers more than the dev numbers.
"""

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROCESSED_DIR = Path("Data/grocery_store/processed")
MANIFEST = PROCESSED_DIR / "manifest.csv"
OUTPUT_DIR = Path("model_outputs")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Reading the manifest and building the splits
# --------------------------------------------------------------------------
def read_manifest(path):
    """Read manifest.csv into a list of dicts (one per image)."""
    if not path.exists():
        raise SystemExit(
            f"Could not find {path}.\n"
            "Run this script from the Recipe-Generator folder, and make sure you\n"
            "ran: python Data/grocery_store/unpack_images.py"
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    needed = {"image_path", "split", "label", "coarse_name"}
    missing = needed - set(rows[0].keys())
    if missing:
        raise SystemExit(f"manifest.csv is missing expected columns: {sorted(missing)}")
    return rows


def split_name(row):
    """Map whatever the manifest calls a split to train / val / test."""
    s = row["split"].strip().lower()
    if s in ("train", "training"):
        return "train"
    if s in ("val", "valid", "validation"):
        return "val"
    if s in ("test", "testing"):
        return "test"
    raise SystemExit(f"Unexpected split name in manifest: {row['split']!r}")


def make_dev_split(train_rows, dev_fraction, seed):
    """Carve a development set out of the training rows only.

    Stratified: each class gives up about `dev_fraction` of its images
    (at least 1 if the class has 2 or more images), so every class that
    can be represented in both sets is.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in train_rows:
        by_class[int(r["label"])].append(r)

    new_train, dev = [], []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        if len(items) < 2:
            new_train.extend(items)
            continue
        n_dev = max(1, math.floor(len(items) * dev_fraction))
        dev.extend(items[:n_dev])
        new_train.extend(items[n_dev:])
    return new_train, dev


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
class IngredientImages(Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if train:
            # Augmentation is applied to training images only.
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.tf = transforms.Compose([transforms.ToTensor(), normalize])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(PROCESSED_DIR / row["image_path"]).convert("RGB")
        return self.tf(img), int(row["label"])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Four small conv blocks followed by a linear layer."""

    def __init__(self, num_classes):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(x)


def build_resnet18(num_classes):
    """Pretrained ResNet18, backbone frozen, new final layer."""
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception as e:  # usually a download / certificate problem
        raise SystemExit(
            "Could not download the pretrained ResNet18 weights.\n"
            "Check your internet connection. If you see a certificate error, "
            "run 'Install Certificates.command' from the Python 3.12 folder in "
            "Applications.\n"
            f"Original error: {e}"
        )
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # trainable
    return model


def set_train_mode(model, frozen_backbone):
    model.train()
    if frozen_backbone:
        # Keep the frozen backbone's BatchNorm statistics fixed.
        for name, module in model.named_children():
            if name != "fc":
                module.eval()


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    present = sorted(set(y_true))
    return f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)


def train_model(name, model, train_loader, dev_loader, device, epochs, lr,
                frozen_backbone):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, best_epoch = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        set_train_mode(model, frozen_backbone)
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)

        logits, labels = predict(model, dev_loader, device)
        preds = logits.argmax(1).numpy()
        dev_acc = float((preds == labels.numpy()).mean())
        dev_f1 = macro_f1(labels.numpy(), preds)
        history.append({"epoch": epoch, "train_loss": running / seen,
                        "dev_accuracy": dev_acc, "dev_macro_f1": dev_f1})
        print(f"[{name}] epoch {epoch:2d}/{epochs}  "
              f"train loss {running / seen:.3f}  dev acc {dev_acc:.3f}  "
              f"dev macro-F1 {dev_f1:.3f}  ({time.time() - t0:.0f}s)")

        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"[{name}] best epoch was {best_epoch} (dev macro-F1 {best_f1:.3f})")
    return model, history, best_epoch


def evaluate_on_test(name, model, test_loader, device, class_names):
    """Run once, at the very end. Never used to choose anything."""
    logits, labels = predict(model, test_loader, device)
    y_true = labels.numpy()
    y_pred = logits.argmax(1).numpy()
    top1 = float((y_pred == y_true).mean())
    k = min(5, logits.shape[1])
    top5_idx = logits.topk(k, dim=1).indices.numpy()
    top5 = float(np.mean([y_true[i] in top5_idx[i] for i in range(len(y_true))]))
    f1 = macro_f1(y_true, y_pred)

    # Per-class recall, for classes that appear in the test set.
    recalls = {}
    for c in sorted(set(y_true)):
        mask = y_true == c
        recalls[class_names[c]] = float((y_pred[mask] == c).mean())

    print(f"\n=== {name}: TEST RESULTS ===")
    print(f"top-1 accuracy: {top1:.3f}   top-5 accuracy: {top5:.3f}   "
          f"macro-F1: {f1:.3f}")
    worst = sorted(recalls.items(), key=lambda kv: kv[1])[:8]
    print("hardest classes (lowest recall):")
    for cname, rec in worst:
        print(f"   {cname:<22s} {rec:.2f}")
    return {"top1": top1, "top5": top5, "macro_f1": f1,
            "per_class_recall": recalls}, y_true, y_pred


def save_confusion_matrix(y_true, y_pred, class_names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n))).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + " (rows sum to 1)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke test: 300 train images, 1 epoch each")
    ap.add_argument("--epochs-cnn", type=int, default=15)
    ap.add_argument("--epochs-resnet", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dev-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu",
                    help="cpu is the safe default; mps (Apple GPU) can crash "
                         "on some Macs")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # CPU is the default because Apple's GPU backend ("mps") crashes on some
    # macOS / PyTorch combinations. Use --device mps or --device cuda to opt in.
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps was requested but MPS is not available.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available.")
    device = torch.device(args.device)
    print(f"Using device: {device}")

    rows = read_manifest(MANIFEST)
    by_split = defaultdict(list)
    for r in rows:
        by_split[split_name(r)].append(r)
    print("Original split sizes:", {k: len(v) for k, v in by_split.items()})

    # Class names from the manifest (coarse ingredient groups, IDs 0..42).
    id_to_name = {}
    for r in rows:
        id_to_name.setdefault(int(r["label"]), r["coarse_name"])
    num_classes = max(id_to_name) + 1
    class_names = [id_to_name.get(i, f"class_{i}") for i in range(num_classes)]
    print(f"Number of ingredient classes: {num_classes}")

    train_rows, dev_rows = make_dev_split(by_split["train"], args.dev_fraction,
                                          args.seed)
    test_rows = by_split["test"]
    print(f"Training: {len(train_rows)}  |  Development (from train only): "
          f"{len(dev_rows)}  |  Test (untouched until the end): {len(test_rows)}")

    if args.quick:
        random.Random(args.seed).shuffle(train_rows)
        train_rows, dev_rows, test_rows = (train_rows[:300], dev_rows[:100],
                                           test_rows[:200])
        args.epochs_cnn = args.epochs_resnet = 1
        print("QUICK MODE: tiny subsets, 1 epoch. Numbers below are not meaningful.")

    def loader(rs, train):
        return DataLoader(IngredientImages(rs, train), batch_size=args.batch_size,
                          shuffle=train, num_workers=0)

    train_loader = loader(train_rows, True)
    dev_loader = loader(dev_rows, False)
    test_loader = loader(test_rows, False)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {
        "source_commit": git_commit(),
        "seed": args.seed,
        "device": str(device),
        "quick_mode": args.quick,
        "hyperparameters": {"batch_size": args.batch_size,
                            "dev_fraction": args.dev_fraction,
                            "epochs_cnn": args.epochs_cnn,
                            "epochs_resnet": args.epochs_resnet,
                            "optimizer": "AdamW", "weight_decay": 1e-4},
        "split_sizes": {"train": len(train_rows), "dev": len(dev_rows),
                        "test": len(test_rows)},
        "models": {},
    }

    experiments = [
        ("small_cnn", SmallCNN(num_classes), args.epochs_cnn, 1e-3, False),
        ("resnet18_frozen", build_resnet18(num_classes), args.epochs_resnet,
         1e-3, True),
    ]
    for name, model, epochs, lr, frozen in experiments:
        print(f"\n########## Training {name} ##########")
        model, history, best_epoch = train_model(
            name, model, train_loader, dev_loader, device, epochs, lr, frozen)
        test_metrics, y_true, y_pred = evaluate_on_test(
            name, model, test_loader, device, class_names)
        save_confusion_matrix(
            y_true, y_pred, class_names,
            OUTPUT_DIR / f"{name}_confusion_matrix.png", f"{name} on test set")

        if name == "small_cnn":
            torch.save(model.state_dict(), OUTPUT_DIR / "small_cnn.pt")
        else:  # only the new final layer is trained, so save just that
            torch.save(model.fc.state_dict(), OUTPUT_DIR / "resnet18_head.pt")

        results["models"][name] = {"best_epoch": best_epoch, "history": history,
                                   "test": test_metrics}

    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (test set) ===")
    for name, info in results["models"].items():
        t = info["test"]
        print(f"{name:<18s} top-1 {t['top1']:.3f}   top-5 {t['top5']:.3f}   "
              f"macro-F1 {t['macro_f1']:.3f}")
    print(f"\nSaved results to {OUTPUT_DIR}/ (metrics.json, confusion matrices, "
          "model weights, class_names.json)")


if __name__ == "__main__":
    main()
