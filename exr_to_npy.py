"""Convert exr renders to Rec. 709 luminance npy files.

Recursively searches a directory for exr files and saves a matching .npy alongside each one. Loading npys is much faster so the measure scripts prefer npy over exr when both are present. Converting up front makes repeated runs much faster.

Usage:
    python exr_to_npy.py DIRECTORY [--overwrite]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import load_luminance


def convert(exr_path: Path, overwrite: bool) -> bool:
    """Convert exr to npy. Skip if already converted or if loading fails. Returns False if skipped."""
    npy_path = exr_path.with_suffix(".npy")
    if npy_path.exists() and not overwrite:
        return False

    try:
        lum = load_luminance(exr_path)
    except Exception as e:
        tqdm.write(f"WARNING: skipping {exr_path} ({type(e).__name__}: {e})")
        return False

    np.save(npy_path, lum)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EXR renders to Rec. 709 luminance NPY files.")
    parser.add_argument("directory", type=Path, help="Root directory to search recursively.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reconvert even if the .npy already exists.")
    args = parser.parse_args()

    exr_paths = sorted(args.directory.rglob("*.exr"))
    if not exr_paths:
        print(f"No EXR files found under {args.directory}")
        return

    converted = 0
    for p in tqdm(exr_paths, unit="file"):
        if convert(p, args.overwrite):
            converted += 1

    print(f"\nconverted: {converted}  skipped: {len(exr_paths) - converted}")


if __name__ == "__main__":
    main()
