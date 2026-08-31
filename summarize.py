"""Average each measure over the image interior.

Spatial measures carry a NaN border where the neighborhood runs off the image. This averages every measure over only those pixels that are valid in all of them, so error and temporal measures are cropped to match the spatial ones.

Usage:
    python summarize.py MEASURES_DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average each measure over the pixels valid in every measure.")
    parser.add_argument("measures_dir", type=Path,
                        help="Directory of measure NPY files, e.g. RENDER_DIR/measures.")
    args = parser.parse_args()

    paths = sorted(args.measures_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"No measure NPY files found in {args.measures_dir}")

    measures = {p.stem: np.load(p) for p in paths}
    interior = np.all([np.isfinite(m) for m in measures.values()], axis=0)

    print(f"{args.measures_dir}  ({interior.sum()} of {interior.size} pixels)")
    for name, data in measures.items():
        print(f"  {name:<16} {data[interior].mean():12.6g}")


if __name__ == "__main__":
    main()
