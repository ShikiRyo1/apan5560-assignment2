"""Render plots from recorded results; this command does not retrain the model."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from app.cnn import CLASSES

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / "results"
    history = json.loads((results / "history.json").read_text())
    metrics = json.loads((results / "metrics.json").read_text())
    epochs = [row["epoch"] for row in history]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, suffix, ylabel in zip(axes, ["loss", "accuracy"], ["Cross-entropy loss", "Accuracy"]):
        axis.plot(epochs, [row[f"train_{suffix}"] for row in history], "k-", label="Training")
        axis.plot(epochs, [row[f"validation_{suffix}"] for row in history], color="0.45", linestyle="--", label="Validation")
        axis.axvline(metrics["best_epoch"], color="0.7", linestyle=":", label="Selected epoch")
        axis.set(xlabel="Epoch", ylabel=ylabel, xticks=[1, 5, 10, 15])
        axis.legend(frameon=False)
        axis.grid(alpha=0.15)
    figure.savefig(results / "learning_curves.png", dpi=180)
    plt.close(figure)
    confusion = np.load(results / "confusion_matrix.npy")
    figure, axis = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    image = axis.imshow(confusion, cmap="Greys", vmin=0, vmax=1000)
    for i in range(10):
        for j in range(10):
            axis.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=9,
                      color="white" if confusion[i, j] > 500 else "black")
    axis.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted class", ylabel="True class", title="Held-out CIFAR-10 test set: 10,000 images")
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    figure.colorbar(image, ax=axis, shrink=0.8, label="Image count")
    figure.savefig(results / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    manifest = json.loads((ROOT / "examples/manifest.json").read_text())
    figure, axes = plt.subplots(2, 5, figsize=(12, 5), constrained_layout=True)
    for axis, row in zip(axes.flat, manifest):
        axis.imshow(Image.open(ROOT / "examples" / row["file"]), interpolation="nearest")
        axis.set_title(f"Test index {row['test_index']}\nTrue: {row['true_class']}\nPred: {row['predicted_class']}", fontsize=10)
        axis.axis("off")
    figure.suptitle("First test image of each class (chosen by index, including errors)")
    figure.savefig(results / "sample_predictions.png", dpi=180)
    plt.close(figure)
    print("Saved learning_curves.png, confusion_matrix.png and sample_predictions.png")


if __name__ == "__main__":
    main()
