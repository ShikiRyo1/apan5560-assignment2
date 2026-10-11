"""Train once on CIFAR-10; select by validation accuracy, then evaluate the test set."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
import torchvision

from app.cnn import AssignmentCNN, CLASSES
from app.preprocessing import normalize_batch, resize_rgb

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def stratified_split(labels, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        validation.extend(indices[:500])
        train.extend(indices[500:])
    return np.array(train), np.array(validation)


def resize_dataset(data):
    # Cache uint8 instead of float32: approximately 586 MiB for all 50k images.
    resized = np.empty((len(data), 64, 64, 3), dtype=np.uint8)
    for index, pixels in enumerate(data):
        resized[index] = resize_rgb(Image.fromarray(pixels))
    return torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()


def run_epoch(model, loader, device, criterion, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, correct, count = 0.0, 0, 0
    all_predictions = []
    with torch.set_grad_enabled(optimizer is not None):
        for pixels, labels in loader:
            images = normalize_batch(pixels.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            loss_sum += loss.item() * len(labels)
            correct += (predictions == labels).sum().item()
            count += len(labels)
            if optimizer is None:
                all_predictions.extend(predictions.cpu().tolist())
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}, all_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    results, models, examples = ROOT / "results", ROOT / "models", ROOT / "examples"
    for directory in (results, models, examples):
        directory.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    config = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
        "scheduler": "StepLR(step_size=5, gamma=0.5)",
        "loss": "CrossEntropyLoss", "augmentation": "none",
        "preprocessing": "EXIF orientation; RGB; PIL bilinear resize 64x64; CHW; (x/255-0.5)/0.5",
        "selection_rule": "highest validation accuracy; retain earliest epoch on a tie",
        "train_count": 45000, "validation_count": 5000, "test_count": 10000,
        "validation_per_class": 500,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "deterministic_algorithms": True,
        "note": "Training accuracy is measured during updates; validation accuracy is measured after each epoch. Test is evaluated only after checkpoint selection.",
    }
    write_json(results / "training_config.json", config)
    print(json.dumps(config), flush=True)
    train_data = CIFAR10(args.data_dir, train=True, download=args.download)
    labels = np.array(train_data.targets)
    train_indices, val_indices = stratified_split(labels, args.seed)
    np.savez_compressed(results / "split_indices.npz", train=train_indices, validation=val_indices)
    assert len(set(train_indices) & set(val_indices)) == 0
    assert len(np.unique(np.concatenate([train_indices, val_indices]))) == 50000
    print("Resizing all 50,000 development images to 64x64...", flush=True)
    pixels = resize_dataset(train_data.data)
    targets = torch.tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(pixels[train_indices], targets[train_indices]),
                              batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TensorDataset(pixels[val_indices], targets[val_indices]),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    del pixels
    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    history, best_accuracy, best_epoch = [], -1.0, None
    model_path = models / "cifar10_cnn.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics, _ = run_epoch(model, val_loader, device, criterion)
        row = {"epoch": epoch, "learning_rate": learning_rate,
               "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
               "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = val_metrics["accuracy"], epoch
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "classes": list(CLASSES), "image_size": 64, "best_epoch": best_epoch,
                        "validation_accuracy": best_accuracy, "normalization_mean": [0.5] * 3,
                        "normalization_std": [0.5] * 3}, model_path)
        write_json(results / "history.json", history)
        print(json.dumps(row), flush=True)
        scheduler.step()
    # This is the first test-set evaluation. It cannot affect selection or training.
    selected = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(selected["state_dict"])
    test_data = CIFAR10(args.data_dir, train=False, download=args.download)
    test_pixels = resize_dataset(test_data.data)
    test_labels = torch.tensor(test_data.targets, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(test_pixels, test_labels), batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_metrics, predictions = run_epoch(model, test_loader, device, criterion)
    confusion = np.zeros((10, 10), dtype=int)
    for truth, prediction in zip(test_data.targets, predictions):
        confusion[truth, prediction] += 1
    with (results / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "true_id", "predicted_id", "true_class", "predicted_class"])
        writer.writerows((i, y, p, CLASSES[y], CLASSES[p]) for i, (y, p) in enumerate(zip(test_data.targets, predictions)))
    examples_manifest = []
    # Predetermined examples: the earliest test image of each true class, including errors.
    for label, name in enumerate(CLASSES):
        index = test_data.targets.index(label)
        path = examples / f"{name}.png"
        Image.fromarray(test_data.data[index]).save(path)
        examples_manifest.append({"file": path.name, "test_index": index, "true_class": name,
                                  "predicted_class": CLASSES[predictions[index]],
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(examples / "manifest.json", examples_manifest)
    np.save(results / "confusion_matrix.npy", confusion)
    summary = {**config, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
               "test": test_metrics, "parameter_count": sum(p.numel() for p in model.parameters()),
               "elapsed_seconds": time.perf_counter() - started,
               "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
               "per_class_test_accuracy": {name: float(confusion[i, i] / confusion[i].sum()) for i, name in enumerate(CLASSES)},
               "test_selection_policy": "one evaluation after validation-based checkpoint selection; no hyperparameter search"}
    write_json(results / "metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
