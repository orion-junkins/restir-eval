"""Per-pixel spatial correlation measures averaged over a circular neighborhood.

Values are efficiently computed in parallel by shifting a copy of an image over itself. 

This leads to a border region that is poorly defined, as values do not exist for all neighbors; we set these values to NaN and exclude these pixels from our analysis.

Usage:
    python compute_spatial_measures.py RENDER_DIR [options]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import EPSILON, load_stack, measures_exist, save_measure, to_device, to_numpy

# Measure names for output file naming
MEASURES = ("cov", "corr", "relcov")

# Default radius for local neighborhood definition; processing time increases significantly for larger radii. Keep small for more rapid experimentation.
DEFAULT_RADIUS = 8


def shifts(radius: int) -> list[tuple[int, int]]:
    """Integer (row, col) offsets inside a circle of the given radius, minus (0, 0)."""
    return [
        (i, j)
        for i in range(-radius, radius + 1)
        for j in range(-radius, radius + 1)
        if (i, j) != (0, 0) and i * i + j * j <= radius * radius
    ]


def nan_border(data: np.ndarray, width: int) -> np.ndarray:
    """Blank out a `width`-pixel frame with NaN, keeping the array full size."""
    out = np.full_like(data, np.nan)
    out[width:-width, width:-width] = data[width:-width, width:-width]
    return out


def spatial_measures(samples: np.ndarray, radius: int) -> dict[str, np.ndarray]:
    """Average covariance, Pearson correlation and relative covariance with neighbors.

    ``torch.roll`` brings each neighbor into alignment with the center pixel, which wraps around at the image edges. The outermost `radius` pixels are filled with NaN rather than reported as the measures are no longer well defined in these areas. Outputs stay (H, W) so they line up with the error and temporal measures pixel for pixel, though all should be cropped to the valid region when studying multiple measures simultaneously.

    Args:
        samples: Luminance stack, shape (N, H, W).
        radius: Neighborhood radius in pixels.
    """
    s = to_device(samples)
    n = s.shape[0]
    mean = s.mean(dim=0)
    dev = s - mean
    std = (dev.pow(2).sum(dim=0) / (n - 1)).sqrt()

    acc = {key: torch.zeros_like(mean) for key in MEASURES}
    offsets = shifts(radius)
    
    for offset in tqdm(offsets, desc="Spatial shifts", unit="shift"):
        rolled_dev = torch.roll(dev, shifts=offset, dims=(-2, -1))
        rolled_std = torch.roll(std, shifts=offset, dims=(-2, -1))
        rolled_mean = torch.roll(mean, shifts=offset, dims=(-2, -1))

        cov = (dev * rolled_dev).sum(dim=0) / (n - 1)
        acc["cov"] += cov
        acc["corr"] += cov / (std * rolled_std + EPSILON)
        acc["relcov"] += cov / ((mean + EPSILON) * (rolled_mean + EPSILON))

    result = {}
    for key, total in acc.items():
        averaged = to_numpy(total / len(offsets))
        result[key] = nan_border(averaged, radius)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-pixel spatial covariance, correlation and relative "
                    "covariance for a directory of renders.")
    parser.add_argument("render_dir", type=Path,
                        help="Directory of per-sample EXR/NPY renders.")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="Where to write measures (default: RENDER_DIR/measures).")
    parser.add_argument("--samples", type=int, default=None, metavar="N",
                        help="Cap the number of samples loaded (default: all).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if all measures already exist.")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS, metavar="R",
                        help=f"Neighborhood radius in pixels (default: {DEFAULT_RADIUS}).")
    args = parser.parse_args()

    if args.radius < 1:
        parser.error("--radius must be at least 1")

    out_dir = args.out or args.render_dir / "measures"
    prefix = f"rad{args.radius}"
    names = [f"{prefix}_{key}" for key in MEASURES]
    if not args.overwrite and measures_exist(out_dir, names):
        print(f"Skipping {args.render_dir} (measures exist; use --overwrite to recompute)")
        return

    samples = load_stack(args.render_dir, args.samples)
    for key, data in spatial_measures(samples, args.radius).items():
        save_measure(out_dir, f"{prefix}_{key}", data)


if __name__ == "__main__":
    main()
