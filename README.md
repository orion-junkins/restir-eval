# ReSTIR Evaluation

Scripts for computing per pixel evaluation measures for error, spatial correlation and temporal correlation.

These are the measures used in **[Compatibility-Guided Neighbor Selection for ReSTIR](https://doi.org/10.1145/3820024)** (HPG 2026, Best Paper) and in Orion Junkins' master thesis, *The Error–Correlation Tradeoff in ReSTIR* (ETH Zürich, 2026). This repository exists to facilitate adoption of these measures.

These scripts and the provided demo data are intended as a starting point and may need to be adapted depending on the format of your inputs and goals of your analysis. 

For more information: [Personal Project Page](https://orion-junkins.github.io/CGNS/) ·
[NVIDIA Project Page](https://research.nvidia.com/labs/rtr/publication/junkins2026compatibility/) ·
[ACM DL](https://dl.acm.org/doi/full/10.1145/3820024)

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+. Runs on Apple MPS, CUDA, or CPU, whichever is available.

## Usage guide

`demo/renders/` ships with enough data to run every measure end to end. Renders come from a 256x256 crop of the Veach Ajar scene, rendered with a baseline ReSTIR PT implementation using Area ReSTIR + Reservoir Splatting. Renders have been pre converted to luminance and are stored as half-precision EXRs to keep file sizes reasonable. 

Data is provided in the following layout:
```
demo/renders/
  reference.exr            ← path-traced ground truth
  R10/  sample_0000.exr …  ← 25 independent captures at reuse radius 10
        lag4/sample_0000.exr …  the same 25 captures, lagged by 4 frames
  R30/  …                  ← radius 30
  R100/ …                  ← radius 100
```

The demo data provides 25 independent samples for each variant. This is enough to make the aggregated measures well-behaved, but all published results use a minimum of 200. 

**1.  Convert EXRs to NumPy Arrays (Optional)** 

Our scripts can handle `.exr` files or `.npy` files. However, EXR decoding dominates runtime, so we recommend converting to `.npy`. This script searches a given directory recursively and converts all `.exr` files to `.npy`.

```bash
python exr_to_npy.py demo/renders
```

**2. Compute Error Measures** 

Evaluate `smape` and `mape` against a reference for each pixel and write results into
`demo/renders/R30/measures/`, as both `.exr` and `.npy`.

```bash
python compute_error_measures.py demo/renders/R30 demo/renders/reference.exr
```

**3. Compute Spatial Measures** 

Evaluate `cov`, `corr` and `relcov` measures spatially for each pixel and write results. Uses the default radius of 8 and names the outputs `rad8_cov`, `rad8_corr`, `rad8_relcov`.

```bash
python compute_spatial_measures.py demo/renders/R30
```

**4. Compute Temporal Measures**

Evaluate `cov`, `corr` and `relcov` measures temporally for each pixel and write results. Pairs each capture with itself 4 frames earlier. The lag directory's *name* becomes the output prefix, so `lag4/` writes `lag4_cov`, `lag4_corr`, and `lag4_relcov`.

```bash
python compute_temporal_measures.py demo/renders/R30 demo/renders/R30/lag4
```

**5. Collapse Maps Into Summary Stat** 

Average all measures over the pixels valid in *every* measure, so the error and temporal maps are cropped to the spatial `NaN` border.

```bash
python summarize.py demo/renders/R30/measures
```

Example output:
```
demo/renders/R30/measures  (57600 of 65536 pixels)
  lag4_corr            0.498212
  lag4_cov             0.021333
  lag4_relcov          0.379869
  mape                  46.8203
  rad8_corr           0.0025634
  rad8_cov          3.85454e-05
  rad8_relcov        0.00120507
  smape                 21.3311
```

**6. Repeat For More Variants**

Explore how results differ across different ReSTIR variants. In this case, we compare three different spatial reuse radii.
```bash
for r in R10 R30 R100; do
    python compute_error_measures.py    "demo/renders/$r" demo/renders/reference.exr
    python compute_spatial_measures.py  "demo/renders/$r"
    python compute_temporal_measures.py "demo/renders/$r" "demo/renders/$r/lag4"
    python summarize.py                 "demo/renders/$r/measures"
done
```

Example output:
| | R10 | R30 | R100 |
|---|---|---|---|
| `smape` | 16.9708 | 21.3311 | 29.7383 |
| `rad8_corr` | 0.0213683 | 0.0025634 | 0.000190496 |
| `lag4_corr` | 0.413131 | 0.498212 | 0.634137 |

This summary is enough to observe the error–correlation tradeoff in ReSTIR. Widening the reuse radius shifts where cost is paid: error rises, spatial correlation between neighbors falls, and temporal correlation climbs. No single number captures all of these dynamics, which is why multi-measure evaluation is essential.

Every map is also on disk as `.exr` — open
`demo/renders/R30/measures/rad8_corr.exr` in [tev](https://github.com/Tom94/tev) or any HDR viewer to see where in the render the correlation actually lives. Note that, given the small sample counts (25 renders), these maps are not very converged on a per pixel basis; with larger sample counts (100+), per pixel patterns become more apparent.

### Options

Scripts expose the following additional options:

| Flag | Scripts | Meaning |
|---|---|---|
| `--out DIR` | all three `compute_*` | Write outputs somewhere other than the default of `RENDER_DIR/measures`. |
| `--samples N` | all three `compute_*` | Cap how many samples are loaded. By default, all are loaded. |
| `--radius R` | `compute_spatial_measures.py` | Reuse radius (default 8); also determines the names of outputs. |
| `--overwrite` | all three `compute_*`, `exr_to_npy.py` | Recompute even if outputs already exist. Defaults to false (Computation/conversion is skipped whenever outputs already exist). |

## Citation

If you use these measures, please cite the paper:

```bibtex
@article{10.1145/3820024,
author = {Junkins, Orion and Kettunen, Markus and Lin, Daqi and Ramamoorthi, Ravi and Wyman, Chris},
title = {Compatibility-Guided Neighbor Selection for ReSTIR},
year = {2026},
issue_date = {July 2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {9},
number = {4},
url = {https://doi.org/10.1145/3820024},
doi = {10.1145/3820024},
journal = {Proc. ACM Comput. Graph. Interact. Tech.},
month = jun,
articleno = {52},
numpages = {16}
}
```
