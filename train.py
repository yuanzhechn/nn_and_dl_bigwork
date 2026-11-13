"""Train independently initialized LeNet variants on the same data split."""

import argparse
from copy import deepcopy
import csv
import json
import platform
import random
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import numpy as np
import torch
from torch import nn

from data import CLASS_NAMES, FashionMNIST, normalize
from models import INPUT_SHAPE, VARIANTS, build_model, model_cost


BASELINE_TARGET = 0.88
APPLICATION_TOLERANCE = 0.01
VALIDATION_PER_CLASS = 500


def split_training_set(labels, seed):
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in range(len(CLASS_NAMES)):
        indices = torch.where(labels == label)[0]
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        validation.append(shuffled[:VALIDATION_PER_CLASS])
        train.append(shuffled[VALIDATION_PER_CLASS:])
    return torch.cat(train), torch.cat(validation)


def make_batches(images, labels, indices, batch_size):
    for batch in indices.split(batch_size):
        yield images[batch], labels[batch]


def run_epoch(model, images, labels, indices, batch_size, loss_fn, optimizer=None, collect=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=images.device)
    correct = torch.zeros((), device=images.device, dtype=torch.long)
    pairs = []
    with torch.set_grad_enabled(training):
        for inputs, targets in make_batches(images, labels, indices, batch_size):
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.detach().argmax(dim=1)
            loss_sum += loss.detach() * len(targets)
            correct += (predictions == targets).sum()
            if collect:
                pairs.append(targets * len(CLASS_NAMES) + predictions)
    result = {"loss": loss_sum.item() / len(indices), "correct": int(correct.item()), "total": len(indices)}
    result["accuracy"] = result["correct"] / result["total"]
    if collect:
        classes = len(CLASS_NAMES)
        counts = torch.bincount(torch.cat(pairs).cpu(), minlength=classes * classes)
        result["confusion_matrix"] = counts.reshape(classes, classes).tolist()
    return result


def train_one(variant, seed, config, dataset, out):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    order_rng = torch.Generator().manual_seed(seed)
    device = torch.device(config["device"])
    model = build_model(variant).to(device)
    cost = model_cost(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"],
    )
    batch_size = config["batch_size"]
    images, labels, train_idx, val_idx, test_images, test_labels, test_idx = dataset
    history, best_state = [], None
    best_accuracy, best_epoch = -1.0, 0
    started = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        order = torch.randperm(len(train_idx), generator=order_rng)
        train = run_epoch(model, images, labels, train_idx[order].to(device), batch_size, criterion, optimizer)
        val = run_epoch(model, images, labels, val_idx, batch_size, criterion)
        history.append({
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train["loss"], "train_accuracy": train["accuracy"],
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
        })
        if val["accuracy"] > best_accuracy:
            best_epoch, best_accuracy = epoch, val["accuracy"]
            best_state = deepcopy(model.state_dict())
        scheduler.step()
        print(f"{variant:8} seed={seed} epoch={epoch:2}/{config['epochs']} "
              f"train_loss={train['loss']:.4f} val_acc={val['accuracy']:.2%}", flush=True)

    model.load_state_dict(best_state)
    test = run_epoch(model, test_images, test_labels, test_idx, batch_size, criterion, collect=True)
    checkpoint = Path("checkpoints") / f"{variant}_seed{seed}.pt"
    torch.save({"variant": variant, "seed": seed, "epoch": best_epoch,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, out / checkpoint)
    result = {
        "variant": variant, "seed": seed, "history": history,
        "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
        "test_loss": test["loss"], "test_accuracy": test["accuracy"],
        "test_correct": test["correct"], "test_total": test["total"],
        "confusion_matrix": test["confusion_matrix"],
        "parameters": cost["parameters"], "macs": cost["macs"], "layers": cost["layers"],
        "elapsed_seconds": perf_counter() - started, "checkpoint": checkpoint.as_posix(),
    }
    print(f"{variant:8} seed={seed} best_epoch={best_epoch} test_acc={test['accuracy']:.2%} "
          f"params={cost['parameters']:,} MACs={cost['macs']:,} time={result['elapsed_seconds']:.1f}s", flush=True)
    return result


def summarize(runs, variants):
    baseline = [run for run in runs if run["variant"] == "baseline"]
    reference_accuracy = mean(run["test_accuracy"] for run in baseline)
    reference_val = mean(run["best_val_accuracy"] for run in baseline)
    summary = []
    for variant in variants:
        group = [run for run in runs if run["variant"] == variant]
        scores = [run["test_accuracy"] for run in group]
        summary.append({
            "variant": variant,
            "parameters": group[0]["parameters"], "macs": group[0]["macs"],
            "test_accuracy_mean": mean(scores), "test_accuracy_std": stdev(scores) if len(scores) > 1 else 0.0,
            "val_accuracy_mean": mean(run["best_val_accuracy"] for run in group),
            "parameter_reduction": 1 - group[0]["parameters"] / baseline[0]["parameters"],
            "mac_reduction": 1 - group[0]["macs"] / baseline[0]["macs"],
            "accuracy_delta": mean(scores) - reference_accuracy,
        })
    eligible = [row for row in summary if row["mac_reduction"] > 0
                and row["val_accuracy_mean"] >= reference_val - APPLICATION_TOLERANCE]
    selected = max(eligible, key=lambda row: (row["val_accuracy_mean"], -row["macs"]))["variant"] if eligible else "baseline"
    chosen_run = max((run for run in runs if run["variant"] == selected),
                     key=lambda run: (run["best_val_accuracy"], -run["seed"]))
    application = {key: chosen_run[key] for key in ("variant", "seed", "checkpoint")}
    application["rule"] = "Highest mean peak validation accuracy among variants with fewer MACs and within 1.0 pp of baseline, otherwise baseline. Highest validation accuracy selects the seed."
    return summary, application


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs, batch-size and threads must be positive")
    if "baseline" not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique and include baseline")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be unique unsigned 32-bit integers")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the current PyTorch environment")

    torch.set_num_threads(args.threads)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "seeds": args.seeds,
        "variants": args.variants, "split_seed": args.split_seed, "threads": args.threads,
        "optimizer": "AdamW", "learning_rate": 0.002, "min_learning_rate": 0.0002,
        "weight_decay": 0.0001, "scheduler": "CosineAnnealingLR", "loss": "CrossEntropyLoss",
        "baseline_target": BASELINE_TARGET, "application_tolerance": APPLICATION_TOLERANCE,
        "device": device, "hardware": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
    }
    out = args.output_dir
    progress_path = out / "progress.json"
    if (out / "metrics.json").exists():
        parser.error("output-dir already contains completed metrics; choose a new directory")
    runs = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["config"] != config:
            parser.error("output-dir contains progress for a different experiment; choose a new directory")
        runs = progress["runs"]

    train_set = FashionMNIST(args.data_root, train=True)
    test_set = FashionMNIST(args.data_root, train=False)
    train_idx, val_idx = split_training_set(train_set.targets, args.split_seed)
    dataset = (
        normalize(train_set.images).unsqueeze(1).to(device), train_set.targets.to(device),
        train_idx, val_idx.to(device), normalize(test_set.images).unsqueeze(1).to(device),
        test_set.targets.to(device), torch.arange(len(test_set), device=device),
    )
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(json.dumps(config), flush=True)
    done = {(run["variant"], run["seed"]) for run in runs if (out / run["checkpoint"]).is_file()}
    runs = [run for run in runs if (run["variant"], run["seed"]) in done]
    for seed in args.seeds:
        for variant in args.variants:
            if (variant, seed) in done:
                continue
            runs.append(train_one(variant, seed, config, dataset, out))
            write_json(progress_path, {"config": config, "runs": runs})

    summary, application = summarize(runs, args.variants)
    metrics = {
        "config": config,
        "dataset": {"name": "Fashion-MNIST", "train": len(train_idx), "validation": len(val_idx),
                    "test": len(test_set), "input_shape": list(INPUT_SHAPE), "classes": list(CLASS_NAMES),
                    "mean": 0.2860, "std": 0.3530},
        "runs": runs, "summary": summary, "application": application,
        "baseline_target_met": summary[args.variants.index("baseline")]["test_accuracy_mean"] >= BASELINE_TARGET,
    }
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(out / "metrics.json", metrics)
    progress_path.unlink(missing_ok=True)
    print(json.dumps({"summary": summary, "application": application, "baseline_target_met": metrics["baseline_target_met"]}, indent=2))


if __name__ == "__main__":
    main()
