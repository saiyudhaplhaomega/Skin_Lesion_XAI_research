"""Benchmark DataLoader worker settings without touching real training data."""
from __future__ import annotations

import argparse
import os
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData


def run_once(num_workers: int, dataset_size: int, batch_size: int) -> float:
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    dataset = FakeData(size=dataset_size, image_size=(3, 224, 224), transform=transform)
    kwargs: dict[str, object] = {"num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    start = time.perf_counter()
    for _images, _labels in DataLoader(dataset, batch_size=batch_size, **kwargs):
        pass
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2 if os.name == "nt" else 4])
    args = parser.parse_args()

    print(f"os.name={os.name}")
    for workers in args.workers:
        elapsed = run_once(workers, args.dataset_size, args.batch_size)
        print(f"num_workers={workers}: {elapsed:.2f}s for {args.dataset_size} fake images")


if __name__ == "__main__":
    main()
