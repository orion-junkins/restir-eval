""" 
Shared helper utilities 
"""
from __future__ import annotations

from pathlib import Path

import Imath
import numpy as np
import OpenEXR
import torch
from tqdm import tqdm

# Guards against division by zero in every ratio-based measure.
EPSILON = 1e-4

# Rec. 709 luminance weights, in R, G, B order.
_LUMA = (0.2126, 0.7152, 0.0722)

_FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def to_device(arr: np.ndarray) -> torch.Tensor:
    """Move a NumPy array onto the active compute device."""
    return torch.from_numpy(arr).to(DEVICE)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Bring a tensor back to the CPU as a NumPy array."""
    return t.cpu().numpy()


def load_exr(path: Path | str) -> dict[str, np.ndarray]:
    """Load an exr to a dictionary with the form {channel_name: (H, W) float32}.
    """
    f = OpenEXR.InputFile(str(path))
    header = f.header()
    dw = header["dataWindow"]
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1
    return {
        name: np.frombuffer(f.channel(name, _FLOAT), dtype=np.float32).reshape(h, w).copy()
        for name in header["channels"]
    }


def load_luminance(path: Path | str) -> np.ndarray:
    """Load one exr or npy file and return a 2D luminance image.
    
    Single channel images are loaded and returned directly (assumed to already be converted to luminance).
    
    3 channel files are converted using Rec. 709 weights and returned.
    """
    path = Path(path)

    # Handle npy
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return sum(w * arr[..., i] for i, w in enumerate(_LUMA)).astype(np.float32)
        raise ValueError(f"Expected (H, W) or (H, W, 3), got {arr.shape}: {path}")
    
    # Handle exr
    channels = load_exr(path)
    if len(channels) == 1:
        return next(iter(channels.values()))

    return sum(w * channels[c] for w, c in zip(_LUMA, "RGB")).astype(np.float32)


def load_stack(directory: Path | str, max_samples: int | None = None) -> np.ndarray:
    """Load a directory of per-sample renders as an ``(N, H, W)`` luminance stack.
    
    Loads npy files if any exist in the directory. Else loads exrs. This assumes that exrs are converted to luminance and then cached as npys.
    
    Filenames are sorted alphabetically (essential for preserving correspondences between samples and lagged samples for temporal measures).
    """
    directory = Path(directory)
    npy_paths = sorted(directory.glob("*.npy"))
    exr_paths = sorted(directory.glob("*.exr"))
    if npy_paths and exr_paths and len(npy_paths) != len(exr_paths):
        raise RuntimeError(
            f"{directory} holds {len(npy_paths)} NPY but {len(exr_paths)} EXR files. Expected equal number.")
    paths = npy_paths or exr_paths
    if not paths:
        raise FileNotFoundError(f"No EXR or NPY files found in {directory}")
    if max_samples is not None:
        paths = paths[:max_samples]
    stack = [load_luminance(p) for p in tqdm(paths, desc=f"Loading {directory.name}", unit="file")]
    return np.stack(stack)


def save_measure(out_dir: Path | str, name: str, data: np.ndarray) -> None:
    """Write one ``(H, W)`` measure as both ``name.exr`` and ``name.npy``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(data, dtype=np.float32)

    h, w = data.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Y": Imath.Channel(_FLOAT)}
    out = OpenEXR.OutputFile(str(out_dir / f"{name}.exr"), header)
    out.writePixels({"Y": data.tobytes()})
    out.close()

    np.save(out_dir / f"{name}.npy", data)
    print(f"wrote {out_dir / name}.{{exr,npy}}")


def measures_exist(out_dir: Path | str, names) -> bool:
    """True when every named measure is already on disk in both formats."""
    out_dir = Path(out_dir)
    return all((out_dir / f"{n}{ext}").exists() for n in names for ext in (".exr", ".npy"))
