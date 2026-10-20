# Spatial SAE study code

This directory collects the code used by the current Spatial SAE results
write-up.  It is organized by scientific role rather than by the chronological
order in which experiments were written.

## Directory map

### `training/`

- `new_spatial_train.py`: main DINOv2 Matryoshka SAE training entry point.  The
  reconstruction-only baseline uses the same training path with spatial loss
  weight set to zero.
- `train_multiscale_spatial_independent_subsae.py`: multiscale/independent-group
  training variant.
- `temporal_sequence_top_k.py`: grouped TopK SAE trainer used by the main run.

### `downstream_tasks/`

- `segment_alignment.py`: within-image object-instance alignment and precision
  among the five nearest patches.
- `spatial_downstream.py`: few-label propagation and PCA-matched control.
- `spatial_downstream_groups.py`: full-code, G0-only, and G1-only propagation.
- `backbone_eval.py`: shared downstream evaluation utilities.
- `two_view_correspondence.py`: cross-view patch localization.
- `code_smoothness_same_vs_across.py`: direct code-smoothness analysis using
  cosine, nonzero-aware cosine, Fourier, Haar-wavelet, multiscale, Lipschitz,
  and activation-density controls.  It compares true image grids with matched
  across-image mosaics and performs image-paired Spatial-minus-Baseline tests.

### `mechanistic_analysis/`

- `dino_smoothness.py`: channel-level spatial autocorrelation.
- `shortcut_analysis.py` and `block_shortcut_sweep.py`: channel energy,
  PCA participation, and smooth-channel allocation to G0.
- `block0_shortcut_image_bootstrap.py`: image-bootstrap uncertainty analysis.
- `pca_curve.py`: G0 code variance/PCA curves and shuffled controls.
- `fvu_compare.py`: reconstruction quality (FVU).
- `whiten_calib.py`, `whiten_diag.py`: channel-whitening calibration and checks.
- `*_multiblock.py`: matched analyses for blocks 0/2/4/6/8/10/11.
- `fourmodel_group_smoothness.py`: decoded G0/G1 smoothness comparisons.

### `synthetic/`

- `synth_spatial_gen.py`: controlled smooth/rough factor generator.
- `train_synth_spatial.py`: paired synthetic SAE training.
- `analyze_synth_recovery.py`, `aggregate_synth_recovery.py`: factor recovery
  and G0 routing/separation.
- `toy_shortcut_calib.py`, `toy_shortcut_check.py`: energy-shortcut injection
  and controls.

### `figures/`

Figure-generation scripts.  Generated figures and result CSVs are intentionally
not committed here; they should be regenerated from the analysis outputs.

### `slurm/`

SLURM launchers used for the multiblock analyses.  The launchers preserve the
original cluster paths for provenance.  Set `REPO`, `STORE`, `RES`, and `PY` to
local equivalents before running elsewhere.

### `utils/`

Checkpoint-specific preprocessing and reproducibility helpers shared across
experiments.

## Current multiblock evaluation

The direct smoothness evaluation is launched with:

```bash
sbatch slurm/run_code_smoothness_multiblock.sbatch
```

It covers blocks 0, 2, 4, 6, 8, 10, and 11; Spatial and reconstruction-only
baseline SAEs; and Full, G0, and G1 codes.  Its three principal outputs per
block are:

- `smoothness_same_vs_across_raw.csv`: image-level same-image, across-image,
  and effect values;
- `smoothness_same_vs_across_summary.csv`: image-bootstrap summaries;
- `smoothness_spatial_vs_baseline_paired.csv`: Spatial-minus-Baseline paired
  comparisons for all three quantities.

## Environment and data

The scripts were run with the `torch2` environment used by the original
project.  Principal dependencies are PyTorch, torchvision, NumPy, pandas,
SciPy, scikit-learn, Pillow, Hugging Face `datasets`, and timm/DINOv2 model
dependencies.  Checkpoints and datasets are not committed to this repository.

Most scripts import helpers from the original `T-SAE-Spatial/scripts` and
`temporal-saes/dictionary_learning` directories.  On a new machine, place
those directories on `PYTHONPATH` or update the path block at the top of the
script.  Hard-coded cluster paths are retained to make the provenance of the
reported runs explicit; they should be converted into command-line arguments
in a later portability pass.

## Statistical conventions

- Comparisons between Spatial and baseline checkpoints are paired whenever
  they use matched images/seeds.
- Image-level tests establish reliability across images, not across training
  seeds; single-checkpoint multiblock results must be labelled accordingly.
- Same-image and across-image values are retained alongside their contrast.
- For smoothness effects, positive values consistently mean greater
  within-image smoothness; the Lipschitz contrast is sign-reversed because
  lower local variation means smoother codes.
