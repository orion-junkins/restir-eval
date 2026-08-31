"""Per-pixel temporal correlation measures against a lagged frame.

Each pixel is compared to itself at some lagged frame. 

It is assumed that render order is consistent between RENDER_DIR and LAG_DIR such that the Nth sample of one corresponds to the Nth sample of the other when sorted alphanumerically. See demo/renders for an example file tree.

Usage:
    python compute_temporal_measures.py RENDER_DIR LAG_DIR [options]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import EPSILON, load_stack, measures_exist, save_measure, to_device, to_numpy

# Measure names for output file naming
MEASURES = ("cov", "corr", "relcov")

def temporal_measures(main_stack: np.ndarray, lag_stack: np.ndarray) -> dict[str, np.ndarray]:
    """Covariance, Pearson correlation and relative covariance across the lag.

    The Nth sample of each stack is taken to be the same render index — files are paired by sort order, so both directories must name their samples consistently.
    
    Args:
        main_stack: Luminance samples from the current frame, shape (N, H, W).
        lag_stack: Luminance samples from the lagged frame, same shape.
    """
    a = to_device(main_stack)
    b = to_device(lag_stack)
    n = a.shape[0]

    mean_a, mean_b = a.mean(dim=0), b.mean(dim=0)
    dev_a, dev_b = a - mean_a, b - mean_b

    cov = (dev_a * dev_b).sum(dim=0) / (n - 1)
    std_a = (dev_a.pow(2).sum(dim=0) / (n - 1)).sqrt()
    std_b = (dev_b.pow(2).sum(dim=0) / (n - 1)).sqrt()

    return {
        "cov": to_numpy(cov),
        "corr": to_numpy(cov / (std_a * std_b + EPSILON)),
        "relcov": to_numpy(cov / ((mean_a + EPSILON) * (mean_b + EPSILON))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-pixel temporal covariance, correlation and relative "
                    "covariance between a directory of renders and its lagged frames.")
    parser.add_argument("render_dir", type=Path,
                        help="Directory of per-sample EXR/NPY renders.")
    parser.add_argument("lag_dir", type=Path,
                        help="Directory of lagged-frame renders. Its name becomes the "
                             "output prefix, e.g. lag3/ produces lag3_corr.exr.")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="Where to write measures (default: RENDER_DIR/measures).")
    parser.add_argument("--samples", type=int, default=None, metavar="N",
                        help="Cap the number of samples loaded from each directory "
                             "(default: all).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if all measures already exist.")
    args = parser.parse_args()

    out_dir = args.out or args.render_dir / "measures"
    prefix = args.lag_dir.name
    names = [f"{prefix}_{key}" for key in MEASURES]
    if not args.overwrite and measures_exist(out_dir, names):
        print(f"Skipping {args.render_dir} (measures exist; use --overwrite to recompute)")
        return

    main_stack = load_stack(args.render_dir, args.samples)
    lag_stack = load_stack(args.lag_dir, args.samples)
    if main_stack.shape != lag_stack.shape:
        raise SystemExit(
            f"Stack mismatch: {args.render_dir} is {main_stack.shape}, "
            f"{args.lag_dir} is {lag_stack.shape}. Samples are paired by sort order, "
            f"so both directories need the same number of equally sized renders.")

    for key, data in temporal_measures(main_stack, lag_stack).items():
        save_measure(out_dir, f"{prefix}_{key}", data)


if __name__ == "__main__":
    main()
