"""Per-pixel error measures against a ground-truth reference.

Loads a directory of per-sample renders, compares every sample to the
reference, and writes the mean per-pixel error as SMAPE and MAPE.

This script can easily be extended to handle additional error measures.

Usage:
    python compute_error_measures.py RENDER_DIR REFERENCE [options]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import EPSILON, load_luminance, load_stack, measures_exist, save_measure, to_device, to_numpy


def smape(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error, bounded to [0, 100].

    Args:
        samples: Luminance stack, shape (N, H, W).
        reference: Ground-truth luminance, shape (H, W).
    """
    s = to_device(samples)
    r = to_device(reference).unsqueeze(0)
    return to_numpy(((s - r).abs() / (s.abs() + r.abs() + EPSILON)).mean(dim=0) * 100.0)


def mape(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mean Absolute Percentage Error, as a percentage.

    Args:
        samples: Luminance stack, shape (N, H, W).
        reference: Ground-truth luminance, shape (H, W).
    """
    s = to_device(samples)
    r = to_device(reference).unsqueeze(0)
    return to_numpy(((s - r).abs() / (r.abs() + EPSILON)).mean(dim=0) * 100.0)

# Mapping from name to measure function
MEASURES = {"smape": smape, "mape": mape}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-pixel SMAPE and MAPE for a directory of renders.")
    parser.add_argument("render_dir", type=Path,
                        help="Directory of per-sample EXR/NPY renders.")
    parser.add_argument("reference", type=Path,
                        help="Ground-truth EXR/NPY image.")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="Where to write measures (default: RENDER_DIR/measures).")
    parser.add_argument("--samples", type=int, default=None, metavar="N",
                        help="Cap the number of samples loaded (default: all).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if all measures already exist.")
    args = parser.parse_args()

    out_dir = args.out or args.render_dir / "measures"
    if not args.overwrite and measures_exist(out_dir, MEASURES):
        print(f"Skipping {args.render_dir} (measures exist; use --overwrite to recompute)")
        return

    samples = load_stack(args.render_dir, args.samples)
    reference = load_luminance(args.reference)
    if samples.shape[1:] != reference.shape:
        raise SystemExit(
            f"Shape mismatch: renders are {samples.shape[1:]}, reference is {reference.shape}")

    for name, fn in MEASURES.items():
        save_measure(out_dir, name, fn(samples, reference))


if __name__ == "__main__":
    main()
