"""Generate figures and a machine-readable analysis from a completed experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from PIL import Image
import torch

from data import CLASS_NAMES, FashionMNIST, normalize
from predict import load_model, predict_test_examples


VARIANT_NAMES = {
    "baseline": "Baseline",
    "dw": "Depthwise separable",
    "se": "SE",
    "gelu": "GELU",
    "dw_se": "Depthwise separable + SE",
}


def save_learning_curves(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for variant in metrics["config"]["variants"]:
        histories = [run["history"] for run in metrics["runs"] if run["variant"] == variant]
        epochs = [row["epoch"] for row in histories[0]]
        if any([row["epoch"] for row in history] != epochs for history in histories):
            raise ValueError(f"Epochs are inconsistent for {variant}")

        for ax, key in zip(axes, ("train_loss", "val_accuracy")):
            values = np.array([[row[key] for row in history] for history in histories])
            ax.plot(epochs, values.mean(axis=0), linewidth=1.7, label=VARIANT_NAMES[variant])

    axes[0].set_title("Training loss")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_title("Validation accuracy")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.24)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.savefig(path, dpi=210)
    plt.close(fig)


def save_confusion_matrix(rates, path):
    fig, ax = plt.subplots(figsize=(7.3, 6), layout="constrained")
    colors = ax.imshow(rates, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for (row, column), rate in np.ndenumerate(rates):
        if row == column or rate >= 0.01:
            color = "white" if rate >= 0.54 else "#15283b"
            ax.text(column, row, f"{rate:.1%}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(colors, ax=ax, format=PercentFormatter(xmax=1), shrink=0.82)
    fig.savefig(path, dpi=210)
    plt.close(fig)


def save_examples(images, examples, path):
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.5), layout="constrained")
    for ax, example in zip(axes.flat, examples):
        index = example["test_index"]
        prediction = example["top_k"][0]
        correct = prediction["class_id"] == example["true_class_id"]
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"Test #{index} · {example['true_class_name']}", fontsize=9)
        ax.set_xlabel(
            f"{prediction['class_name']}  {prediction['probability']:.1%}",
            fontsize=9,
            color="#174c42" if correct else "#912f2f",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=210)
    plt.close(fig)


def summarize_confusion(counts, rates):
    recalls = [
        {"class_name": name, "recall": float(rates[index, index]), "support": int(counts[index].sum())}
        for index, name in enumerate(CLASS_NAMES)
    ]
    errors = [
        {
            "true_class_name": CLASS_NAMES[row],
            "predicted_class_name": CLASS_NAMES[column],
            "count": int(count),
            "rate": float(rates[row, column]),
        }
        for (row, column), count in np.ndenumerate(counts)
        if row != column and count > 0
    ]
    return recalls, sorted(errors, key=lambda error: error["count"], reverse=True)[:5]


@torch.inference_mode()
def evaluate_application(model, dataset, batch_size, recorded_counts, device):
    logits = torch.cat([
        model(normalize(batch).unsqueeze(1).to(device)).cpu()
        for batch in dataset.images.split(batch_size)
    ])
    targets = dataset.targets
    first = logits.argmax(dim=1)
    correct_at_one = first == targets
    correct_at_three = logits.topk(3, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1)

    classes = len(CLASS_NAMES)
    counts = torch.bincount(targets * classes + first, minlength=classes * classes)
    counts = counts.reshape(classes, classes)
    if not np.array_equal(counts.tolist(), recorded_counts):
        raise ValueError(
            "Recomputed test confusion matrix differs from the recorded run. "
            "Check the dataset and try the training run's device and PyTorch environment."
        )

    shirt = CLASS_NAMES.index("Shirt")
    t_shirt = CLASS_NAMES.index("T-shirt/top")
    errors = torch.where((targets == shirt) & (first == t_shirt))[0]
    failure_example = None
    if len(errors):
        index = int(errors[0])
        failure_example = predict_test_examples(model, dataset, [index], top_k=3)[0]

    top3_correct = int(correct_at_three.sum())
    return {
        "test_top3_correct": top3_correct,
        "test_top3_accuracy": top3_correct / len(dataset),
        "top1_misses_recovered_at_top3": int((~correct_at_one & correct_at_three).sum()),
        "failure_example": failure_example,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to the metrics file's directory.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the current PyTorch environment")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    application = metrics["application"]
    selected = application["variant"], application["seed"]
    run = next(run for run in metrics["runs"] if (run["variant"], run["seed"]) == selected)
    if run["checkpoint"] != application["checkpoint"]:
        raise ValueError("Application checkpoint path differs from the selected run")
    torch.set_num_threads(metrics["config"]["threads"])
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    variant, model = load_model(
        args.metrics.parent / application["checkpoint"],
        expected_metadata={"variant": run["variant"], "seed": run["seed"], "epoch": run["best_epoch"]},
    )
    model.to(args.device)

    counts = np.asarray(run["confusion_matrix"])
    if counts.sum() != run["test_total"] or np.trace(counts) != run["test_correct"]:
        raise ValueError("Confusion matrix totals disagree with the recorded test counts")
    rates = counts / counts.sum(axis=1, keepdims=True)
    recalls, errors = summarize_confusion(counts, rates)

    dataset = FashionMNIST(args.data_root, train=False)
    application_results = evaluate_application(
        model, dataset, metrics["config"]["batch_size"], counts, args.device
    )
    examples = predict_test_examples(model, dataset, range(8), top_k=3)
    images = np.array(dataset.images[:len(examples)].tolist(), dtype=np.uint8)
    correct = sum(example["top_k"][0]["class_id"] == example["true_class_id"] for example in examples)

    output = args.output_dir if args.output_dir is not None else args.metrics.parent
    output.mkdir(parents=True, exist_ok=True)
    save_learning_curves(metrics, output / "learning_curves.png")
    save_confusion_matrix(rates, output / "confusion_matrix.png")
    save_examples(images, examples, output / "application_samples.png")
    Image.fromarray(images[0]).save(output / "application_input.png")
    analysis = {
        "variant": variant,
        "seed": run["seed"],
        "test_accuracy": run["test_accuracy"],
        "test_correct": run["test_correct"],
        "test_total": run["test_total"],
        **application_results,
        "class_recalls": recalls,
        "common_errors": errors,
        "demonstration_correct": correct,
        "demonstration_total": len(examples),
        "demonstration": examples,
    }
    path = output / "analysis.json"
    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analysis and figures saved in {output}")


if __name__ == "__main__":
    main()
