"""Classify Fashion-MNIST examples or single-garment silhouette images."""

import argparse
from inspect import signature
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch

from data import CLASS_NAMES, FashionMNIST, normalize
from models import build_model


def load_model(checkpoint_path, *, expected_metadata=None):
    options = {"weights_only": True} if "weights_only" in signature(torch.load).parameters else {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", **options)
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            actual = checkpoint.get(key)
            if actual != expected:
                raise ValueError(f"Checkpoint {key} is {actual!r}; expected {expected!r}")
    variant = checkpoint["variant"]
    model = build_model(variant)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return variant, model


def load_silhouette(path, invert=False):
    """Letterbox the image onto a black 28×28 canvas and apply training normalization."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = (255, 255, 255, 255) if invert else (0, 0, 0, 255)
        gray = Image.alpha_composite(Image.new("RGBA", rgba.size, background), rgba).convert("L")

    if invert:
        gray = ImageOps.invert(gray)

    side = 28
    scale = side / max(gray.size)
    width = max(1, round(gray.width * scale))
    height = max(1, round(gray.height * scale))
    gray = gray.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (side, side))
    canvas.paste(gray, ((side - width) // 2, (side - height) // 2))
    pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).reshape(1, side, side)
    return normalize(pixels)


@torch.inference_mode()
def predict_batch(model, inputs, top_k):
    scores = model(inputs).softmax(dim=1)
    probabilities, indices = scores.topk(top_k, dim=1)
    return [
        [
            {"class_id": index, "class_name": CLASS_NAMES[index], "probability": probability}
            for index, probability in zip(row_indices, row_probabilities)
        ]
        for row_indices, row_probabilities in zip(indices.tolist(), probabilities.tolist())
    ]


def predict_test_examples(model, dataset, indices, top_k):
    samples = [dataset[index] for index in indices]
    device = next(model.parameters()).device
    inputs = torch.stack([image for image, _ in samples]).to(device)
    predictions = predict_batch(model, inputs, top_k)
    return [
        {
            "test_index": index,
            "true_class_id": int(label),
            "true_class_name": CLASS_NAMES[int(label)],
            "top_k": ranked,
        }
        for index, (_, label), ranked in zip(indices, samples, predictions)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run closed-set inference on Fashion-MNIST or similar garment silhouettes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", type=Path, help="One garment per image, black background.")
    source.add_argument("--test-index", nargs="+", type=int, help="Zero-based official test split indices.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--invert", action="store_true", help="Use for dark garments on a white background.")
    args = parser.parse_args()

    if not 1 <= args.top_k <= len(CLASS_NAMES):
        parser.error(f"--top-k must be between 1 and {len(CLASS_NAMES)}")
    if args.invert and args.images is None:
        parser.error("--invert requires --images")

    variant, model = load_model(args.checkpoint)
    if args.test_index is not None:
        dataset = FashionMNIST(args.data_root, train=False)
        if any(index < 0 or index >= len(dataset) for index in args.test_index):
            parser.error(f"--test-index must be between 0 and {len(dataset) - 1}")
        predictions = predict_test_examples(model, dataset, args.test_index, args.top_k)
    else:
        inputs = torch.stack([load_silhouette(path, args.invert) for path in args.images])
        ranked = predict_batch(model, inputs, args.top_k)
        predictions = [
            {"image": str(path), "top_k": result} for path, result in zip(args.images, ranked)
        ]

    print(json.dumps({"variant": variant, "predictions": predictions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
