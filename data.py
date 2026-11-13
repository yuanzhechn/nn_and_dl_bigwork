"""Download, verify, and load the original Fashion-MNIST IDX files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from http.client import HTTPException
from pathlib import Path
import shutil
import struct
import tempfile
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
PIXEL_MEAN = 0.2860
PIXEL_STD = 0.3530

# The published MD5 values identify the original compressed archives.
# https://github.com/zalandoresearch/fashion-mnist#get-the-data
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
MIRRORS = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://fashion-mnist.s3.eu-central-1.amazonaws.com/",
)


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Standardize uint8 pixels and preserve the input dimensions."""
    return images.to(dtype=torch.float32).div(255).sub_(PIXEL_MEAN).div_(PIXEL_STD)


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: str | Path) -> None:
    """Keep verified archives and fetch any missing or damaged archives."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected_md5 in FILES.items():
        target = root / name
        if target.is_file() and file_md5(target) == expected_md5:
            print(f"Verified {target}")
            continue

        errors = []
        for mirror in MIRRORS:
            url = mirror + name
            print(f"Downloading {url}", flush=True)
            with tempfile.TemporaryDirectory(dir=root, prefix="fashion-mnist-") as temp:
                candidate = Path(temp) / name
                try:
                    with urlopen(url, timeout=30) as source, candidate.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if file_md5(candidate) != expected_md5:
                        raise ValueError("MD5 checksum mismatch")
                except (OSError, HTTPException, ValueError) as error:
                    errors.append(f"{url}: {error}")
                    continue
                candidate.replace(target)
                print(f"Verified {target}")
                break
        else:
            details = "\n".join(errors)
            raise RuntimeError(f"Could not download {name}.\n{details}")


def read_idx(path: Path, header: tuple[int, ...], shape: tuple[int, ...]) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Run python data.py --root "{path.parent}".')
    if file_md5(path) != FILES[path.name]:
        raise ValueError(f'Checksum mismatch for {path}. Run python data.py --root "{path.parent}".')

    expected_header = struct.pack(f">{len(header)}I", *header)
    with gzip.open(path, "rb") as source:
        if source.read(len(expected_header)) != expected_header:
            raise ValueError(f"Unexpected IDX header in {path}")
        pixels = bytearray(source.read())
    return torch.frombuffer(pixels, dtype=torch.uint8).reshape(shape)


class FashionMNIST(Dataset):
    """The official train or test split, stored as uint8 images and long labels."""

    def __init__(self, root: str | Path, train: bool = True):
        root = Path(root)
        split = "train" if train else "t10k"
        count = 60_000 if train else 10_000
        self.images = read_idx(
            root / f"{split}-images-idx3-ubyte.gz",
            (2051, count, 28, 28),
            (count, 28, 28),
        )
        self.targets = read_idx(
            root / f"{split}-labels-idx1-ubyte.gz", (2049, count), (count,)
        ).long()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return normalize(self.images[index]).unsqueeze(0), int(self.targets[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"), help="Archive directory")
    args = parser.parse_args()
    try:
        download(args.root)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
