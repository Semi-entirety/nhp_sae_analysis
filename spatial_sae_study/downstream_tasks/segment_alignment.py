"""
analyze_ade20k_segment_alignment_multi.py

Multi-method ADE20K instance-segmentation alignment analysis.

Compared with the original script, this version supports:
  - multiple checkpoints: --ckpt LABEL:PATH
  - checkpoint representations split into Full / G0 / G1 via --ckpt_groups full G0 G1
  - Raw DINO baseline by default
  - PCA / precomputed patch features via --pca_feature LABEL:PATH.npy
  - PCA models fitted elsewhere via --pca_model LABEL:PATH.pkl
  - grouped bar plots where each checkpoint is a group with Full/G0/G1 bars,
    while Raw DINO and PCA baselines are single bars.

The metrics are the same as the original ADE20K alignment script:
  - same-instance cosine similarity
  - different-instance cosine similarity
  - gap = same - different
  - same-instance retrieval precision@k

Examples:

  python -u analyze_ade20k_segment_alignment_multi.py \
    --dataset_name 1aurent/ADE20K \
    --split validation \
    --max_images 500 \
    --ckpt spatial:results/v4_fixed_10.0/ae_latest.pt \
    --ckpt baseline:results/fixed_baseline_32/ae_latest.pt \
    --pca_feature PCA_B11:results/pca_b11_ade20k_patch_features.npy \
    --ckpt_groups full G0 G1 \
    --outdir results/ade20k_multi_alignment

Notes on --pca_feature:
  The .npy must contain patch-level features, not image-level pooled features.
  Accepted shapes:
    [len(dataset), num_patches, dim]  -> indexed by ADE20K dataset index
    [num_evaluated_images, num_patches, dim] -> indexed by evaluation order
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    import joblib
except Exception:
    joblib = None


# -----------------------------------------------------------------------------
# Imports for the SAE class
# -----------------------------------------------------------------------------

def add_dictionary_learning_paths(extra_root: Optional[str] = None):
    candidates = []
    if extra_root:
        candidates.append(Path(extra_root).expanduser())
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    candidates.extend([
        cwd / "temporal-saes" / "dictionary_learning",
        cwd / "dictionary_learning",
        cwd,
        here / "temporal-saes" / "dictionary_learning",
        here / "dictionary_learning",
        here,
    ])
    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p))


# -----------------------------------------------------------------------------
# DINO and SAE loading
# -----------------------------------------------------------------------------

class DINOFeatureExtractor:
    def __init__(self, model_name: str, device: str, repo: str = "", dino_block=None):
        self.device = device
        self.dino_block = dino_block  # 0-indexed absolute block (normed); None => final layer
        candidate_repos = []
        if repo:
            candidate_repos.append(Path(repo).expanduser())
        if os.environ.get("DINO_REPO"):
            candidate_repos.append(Path(os.environ["DINO_REPO"]).expanduser())
        candidate_repos.append(Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main")

        self.model = None
        last_err = None
        for candidate in candidate_repos:
            if candidate.is_file() and candidate.name == "hubconf.py":
                candidate = candidate.parent
            hubconf = candidate / "hubconf.py"
            if not hubconf.exists():
                continue
            try:
                print(f"[DINO] Loading local repo: {candidate}")
                self.model = torch.hub.load(str(candidate), model_name, source="local", trust_repo=True)
                break
            except Exception as e:
                last_err = e
                print(f"[DINO] Failed local repo {candidate}: {type(e).__name__}: {e}")

        if self.model is None:
            try:
                print("[DINO] Loading via torch hub: facebookresearch/dinov2")
                self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            except Exception as e:
                msg = (
                    "Could not load DINOv2. Pass --dino_repo /path/to/facebookresearch_dinov2_main "
                    "or set DINO_REPO to a readable local DINOv2 repo."
                )
                if last_err is not None:
                    msg += f" Last local error: {type(last_err).__name__}: {last_err}"
                raise RuntimeError(msg) from e

        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def patch_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        x = image_tensor.to(self.device)
        if self.dino_block is None:
            feats = self.model.forward_features(x)
            return feats["x_norm_patchtokens"].squeeze(0)  # [P, D]
        # Match training/other-task extraction: 0-indexed block via get_intermediate_layers, normed.
        out = self.model.get_intermediate_layers(
            x, n=[self.dino_block], reshape=False, return_class_token=False, norm=True)[0]
        return out.squeeze(0)  # [P, D]


@dataclass
class SAEEntry:
    label: str
    path: str
    ae: object
    dict_size: int
    group_sizes: List[int]
    k: int
    group_slices: Dict[str, Tuple[int, int]]


def parse_label_path(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Expected LABEL:PATH, got {spec}")
    label, path = spec.split(":", 1)
    if not label:
        raise ValueError(f"Empty label in {spec}")
    return label, path


def load_sae_from_ckpt(label: str, ckpt: str, device: str, dict_root: Optional[str] = None) -> SAEEntry:
    add_dictionary_learning_paths(dict_root)
    try:
        from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE
    except Exception as e:
        raise RuntimeError(
            "Could not import TemporalMatryoshkaBatchTopKSAE. Run this script from your repo root, "
            "or pass --dict_root /path/to/temporal-saes/dictionary_learning."
        ) from e

    raw_cpu = torch.load(ckpt, map_location="cpu")
    state_dict = raw_cpu["ae_state_dict"] if isinstance(raw_cpu, dict) and "ae_state_dict" in raw_cpu else raw_cpu

    if "W_enc" not in state_dict:
        raise KeyError(f"Checkpoint {ckpt} does not contain W_enc; cannot infer SAE dimensions.")
    activation_dim, dict_size = state_dict["W_enc"].shape

    if "group_sizes" not in state_dict:
        raise KeyError(f"Checkpoint {ckpt} does not contain group_sizes; cannot infer groups.")
    group_sizes = [int(x) for x in state_dict["group_sizes"].tolist()]

    if "k" not in state_dict:
        raise KeyError(f"Checkpoint {ckpt} does not contain k; please modify loader or checkpoint.")
    k = int(state_dict["k"].item())

    ae = TemporalMatryoshkaBatchTopKSAE(
        activation_dim=int(activation_dim),
        dict_size=int(dict_size),
        k=k,
        group_sizes=group_sizes,
        temporal=True,
    ).to(device)

    raw_dev = torch.load(ckpt, map_location=device)
    state_dict_dev = raw_dev["ae_state_dict"] if isinstance(raw_dev, dict) and "ae_state_dict" in raw_dev else raw_dev
    ae.load_state_dict(state_dict_dev)
    ae.eval()

    bounds = [0]
    for g in group_sizes:
        bounds.append(bounds[-1] + int(g))
    group_slices = {f"G{i}": (bounds[i], bounds[i + 1]) for i in range(len(group_sizes))}

    print(f"[SAE:{label}] path={ckpt}")
    print(f"[SAE:{label}] activation_dim={activation_dim}, dict_size={dict_size}, k={k}, group_sizes={group_sizes}")
    return SAEEntry(label, ckpt, ae, int(dict_size), group_sizes, k, group_slices)


# -----------------------------------------------------------------------------
# PCA / precomputed feature loading
# -----------------------------------------------------------------------------

@dataclass
class PCAFeatureEntry:
    label: str
    path: str
    array: np.ndarray


@dataclass
class PCAModelEntry:
    label: str
    path: str
    model: object


def load_pickle_or_joblib(path: str):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


def get_precomputed_patch_features(entry: PCAFeatureEntry, dataset_index: int, eval_position: int, total_dataset_len: int, n_eval: int) -> torch.Tensor:
    arr = entry.array
    if arr.ndim != 3:
        raise ValueError(f"--pca_feature {entry.label} must be [N, P, D], got shape {arr.shape}")
    if arr.shape[0] == total_dataset_len:
        feat = arr[int(dataset_index)]
    elif arr.shape[0] == n_eval:
        feat = arr[int(eval_position)]
    else:
        raise ValueError(
            f"Cannot align --pca_feature {entry.label}: first dim {arr.shape[0]} is neither "
            f"len(dataset)={total_dataset_len} nor n_eval={n_eval}."
        )
    return torch.as_tensor(feat, dtype=torch.float32)


def transform_with_pca_model(model, patch_tokens: torch.Tensor) -> torch.Tensor:
    x = patch_tokens.detach().cpu().float().numpy()
    if not hasattr(model, "transform"):
        raise ValueError("PCA model object does not have a .transform method")
    y = model.transform(x)
    return torch.as_tensor(y, dtype=torch.float32)


# -----------------------------------------------------------------------------
# ADE20K instance-mask processing
# -----------------------------------------------------------------------------

def make_image_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_instance_id_map(
    instances: List[Image.Image],
    objects: Optional[List[dict]],
    image_size: int,
    mask_rule: str = "positive",
    sort_by_depth: bool = False,
) -> Optional[np.ndarray]:
    if len(instances) == 0:
        return None

    order = list(range(len(instances)))
    if sort_by_depth and objects is not None and len(objects) == len(instances):
        def depth_rank(i):
            try:
                return int(objects[i].get("depth_ordering_rank", i))
            except Exception:
                return i
        order = sorted(order, key=depth_rank)

    seg_id = np.full((image_size, image_size), -1, dtype=np.int32)
    for idx in order:
        mask_img = instances[idx].resize((image_size, image_size), resample=Image.Resampling.NEAREST)
        arr = np.array(mask_img)
        if mask_rule == "sure":
            mask = arr == 255
        elif mask_rule == "positive":
            mask = arr > 0
        else:
            raise ValueError(f"Unknown mask_rule={mask_rule}")
        seg_id[mask] = idx
    return seg_id


def label_map_to_patch_labels(
    label_map: np.ndarray,
    image_size: int = 224,
    patch_size: int = 14,
    purity_threshold: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert image_size % patch_size == 0
    side = image_size // patch_size
    patch_labels = np.full(side * side, -1, dtype=np.int32)
    valid = np.zeros(side * side, dtype=bool)
    purities = np.zeros(side * side, dtype=np.float32)

    idx = 0
    area = patch_size * patch_size
    for r in range(side):
        for c in range(side):
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            patch = label_map[y0:y1, x0:x1].reshape(-1)
            fg = patch[patch >= 0]
            if fg.size == 0:
                idx += 1
                continue
            labels, counts = np.unique(fg, return_counts=True)
            best_i = int(np.argmax(counts))
            best_label = int(labels[best_i])
            purity = float(counts[best_i]) / float(area)
            purities[idx] = purity
            if purity >= purity_threshold:
                patch_labels[idx] = best_label
                valid[idx] = True
            idx += 1
    return patch_labels, valid, purities


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def feature_metrics(
    features: torch.Tensor,
    patch_labels: np.ndarray,
    valid: np.ndarray,
    precision_k: int = 5,
) -> Optional[Dict[str, float]]:
    valid_idx = np.where(valid)[0]
    if valid_idx.size < 3:
        return None

    labels = patch_labels[valid_idx]
    if len(np.unique(labels)) < 2:
        return None

    z = features[valid_idx]
    z = F.normalize(z.float(), dim=-1, eps=1e-8)
    sim = (z @ z.T).detach().cpu().numpy()

    n = sim.shape[0]
    eye = np.eye(n, dtype=bool)
    same = (labels[:, None] == labels[None, :]) & (~eye)
    diff = (labels[:, None] != labels[None, :]) & (~eye)

    if same.sum() == 0 or diff.sum() == 0:
        return None

    same_vals = sim[same]
    diff_vals = sim[diff]

    precisions = []
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        k_eff = min(precision_k, n - 1)
        if k_eff <= 0:
            continue
        nn = np.argpartition(-row, kth=k_eff - 1)[:k_eff]
        precisions.append(float(np.mean(labels[nn] == labels[i])))

    return {
        "n_valid_patches": int(n),
        "n_instances": int(len(np.unique(labels))),
        "same_pairs": int(same.sum()),
        "diff_pairs": int(diff.sum()),
        "same_sim": float(np.mean(same_vals)),
        "diff_sim": float(np.mean(diff_vals)),
        "gap": float(np.mean(same_vals) - np.mean(diff_vals)),
        f"precision_at_{precision_k}": float(np.mean(precisions)) if precisions else float("nan"),
    }


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Return Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.where(np.isfinite(p))[0]
    m = len(finite)
    if m == 0:
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        adj = min(1.0, max(prev, adj))
        out[idx] = adj
        prev = adj
    return out.tolist()


def paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    d = a - b
    out = {
        "n": int(d.size),
        "mean_a": float(np.mean(a)) if d.size else float("nan"),
        "mean_b": float(np.mean(b)) if d.size else float("nan"),
        "mean_a_minus_b": float(np.mean(d)) if d.size else float("nan"),
        "sem_a_minus_b": sem(d) if d.size else float("nan"),
        "fraction_a_greater_b": float(np.mean(d > 0)) if d.size else float("nan"),
    }
    if scipy_stats is not None and d.size >= 2 and np.std(d) > 0:
        out["paired_t_p"] = float(scipy_stats.ttest_rel(a, b).pvalue)
        try:
            out["wilcoxon_p"] = float(scipy_stats.wilcoxon(d).pvalue)
        except Exception:
            out["wilcoxon_p"] = float("nan")
    else:
        out["paired_t_p"] = float("nan")
        out["wilcoxon_p"] = float("nan")
    return out


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_flat(summary: Dict[str, Dict[str, float]], metric: str, outpath: Path, title: str):
    names = list(summary.keys())
    means = [summary[n].get(f"{metric}_mean", float("nan")) for n in names]
    errors = [summary[n].get(f"{metric}_sem", float("nan")) for n in names]

    plt.figure(figsize=(max(8, 1.1 * len(names)), 5))
    x = np.arange(len(names))
    plt.bar(x, means, yerr=errors, capsize=4)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    for i, v in enumerate(means):
        if np.isfinite(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def build_grouped_plot_data(rows: List[Dict], metric: str):
    """Collect mean/SEM and plotting positions for grouped metric plots."""
    order = []
    values: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        key = (str(r["method"]), str(r["representation"]))
        if key not in values:
            values[key] = []
            order.append(key)
        try:
            val = float(r[metric])
            if np.isfinite(val):
                values[key].append(val)
        except Exception:
            pass

    method_order = []
    for m, rep in order:
        if m not in method_order:
            method_order.append(m)

    rep_order = ["Raw", "PCA", "Full", "G0", "G1"]
    extra_reps = []
    for _, rep in order:
        if rep not in rep_order and rep not in extra_reps:
            extra_reps.append(rep)
    rep_order = rep_order + extra_reps

    means = {k: float(np.mean(np.asarray(v, dtype=float))) if v else float("nan") for k, v in values.items()}
    errors = {k: sem(np.asarray(v, dtype=float)) if v else float("nan") for k, v in values.items()}
    return order, values, method_order, rep_order, means, errors


def plot_metric_grouped(
    rows: List[Dict],
    metric: str,
    outpath: Path,
    title: str,
    sig_lookup: Optional[Dict[Tuple[str, str], str]] = None,
    sig_note: str = "",
):
    # summary by (method, representation), preserving order seen in rows
    order, values, method_order, rep_order, means, errors = build_grouped_plot_data(rows, metric)

    fig_w = max(9, 1.8 * len(method_order))
    plt.figure(figsize=(fig_w, 6.2 if sig_lookup else 5.5))
    ax = plt.gca()
    group_x = np.arange(len(method_order), dtype=float)
    width = 0.22

    present_reps = [rep for rep in rep_order if any((m, rep) in values for m in method_order)]
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    rep_colors = {rep: prop_cycle[i % len(prop_cycle)] if prop_cycle else None for i, rep in enumerate(present_reps)}

    y_values_for_limits = []
    bar_tops: Dict[Tuple[str, str], float] = {}
    for i, method in enumerate(method_order):
        reps = [rep for rep in rep_order if (method, rep) in values]
        n = len(reps)
        offsets = (np.arange(n) - (n - 1) / 2.0) * width if n > 0 else []
        for off, rep in zip(offsets, reps):
            key = (method, rep)
            x = group_x[i] + off
            v = means[key]
            e = errors[key]
            color = rep_colors.get(rep, None)
            ax.bar(x, v, width=width * 0.9, yerr=e, capsize=3, color=color)
            if np.isfinite(v):
                y_top = v + (e if np.isfinite(e) else 0.0)
                y_values_for_limits.append(y_top)
                bar_tops[key] = y_top
                ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
                if sig_lookup is not None:
                    stars = sig_lookup.get(key, "")
                    if stars and stars != "ns":
                        ax.text(x, y_top, stars, ha="center", va="bottom", fontsize=10, fontweight="bold")

    handles = [Patch(facecolor=rep_colors.get(rep, "gray"), label=rep) for rep in present_reps]
    ax.set_xticks(group_x)
    ax.set_xticklabels(method_order, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if handles:
        ax.legend(handles=handles, title="Representation", fontsize=8)

    if sig_lookup and sig_note:
        ax.text(0.01, 0.99, sig_note, transform=ax.transAxes, ha="left", va="top", fontsize=8)

    if y_values_for_limits:
        ymin = min(0.0, float(np.nanmin(y_values_for_limits)))
        ymax = float(np.nanmax(y_values_for_limits))
        pad = max(0.03 * (ymax - ymin), 0.01)
        ax.set_ylim(ymin, ymax + (3.5 * pad if sig_lookup else pad))

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="1aurent/ADE20K")
    parser.add_argument("--cache_dir", type=str, default="data/huggingface/datasets")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)

    # New multi-method interface.
    parser.add_argument("--ckpt", action="append", default=[], help="Checkpoint spec LABEL:PATH. Can be repeated.")
    parser.add_argument("--ckpt_groups", nargs="+", default=["full", "G0", "G1"],
                        help="Representations to evaluate for each checkpoint, e.g. full G0 G1")
    parser.add_argument("--pca_feature", action="append", default=[],
                        help="Precomputed patch-level feature spec LABEL:PATH.npy. Can be repeated.")
    parser.add_argument("--pca_model", action="append", default=[],
                        help="PCA/sklearn model spec LABEL:PATH.pkl. Transforms raw DINO patch tokens. Can be repeated.")
    parser.add_argument("--include_raw_dino", action="store_true", default=True,
                        help="Include Raw DINO baseline. Default true; use --no_raw_dino to disable.")
    parser.add_argument("--no_raw_dino", action="store_true", help="Disable Raw DINO baseline.")

    parser.add_argument("--dict_root", type=str, default="", help="Optional path to temporal-saes/dictionary_learning")
    parser.add_argument("--dino_repo", type=str, default="", help="Path to facebookresearch_dinov2_main repo directory")
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--dino_block", type=int, default=None, help="0-indexed DINOv2 block (normed). Omit for final layer.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--purity_threshold", type=float, default=0.6)
    parser.add_argument("--mask_rule", type=str, default="positive", choices=["positive", "sure"],
                        help="positive uses mask > 0, sure uses mask == 255")
    parser.add_argument("--sort_by_depth", action="store_true")
    parser.add_argument("--precision_k", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="results_ade20k_segment_alignment_multi")
    parser.add_argument("--save_debug", action="store_true")
    args = parser.parse_args()

    if args.no_raw_dino:
        args.include_raw_dino = False

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable; using CPU.")
        args.device = "cpu"

    if args.image_size % args.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    if not args.include_raw_dino and not args.ckpt and not args.pca_feature and not args.pca_model:
        raise ValueError("No methods selected. Use --ckpt, --pca_feature, --pca_model, or include Raw DINO.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Dataset] Loading", args.dataset_name, args.split)
    ds = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    if args.max_images > 0:
        end = min(len(ds), args.start + args.max_images)
        indices = list(range(args.start, end))
    else:
        indices = list(range(args.start, len(ds)))
    print(f"[Dataset] total={len(ds)}, evaluating {len(indices)} images")

    dino = DINOFeatureExtractor(args.dino_model, args.device, args.dino_repo, dino_block=args.dino_block)

    sae_entries: List[SAEEntry] = []
    for spec in args.ckpt:
        label, path = parse_label_path(spec)
        sae_entries.append(load_sae_from_ckpt(label, path, args.device, args.dict_root or None))

    pca_feature_entries: List[PCAFeatureEntry] = []
    for spec in args.pca_feature:
        label, path = parse_label_path(spec)
        print(f"[PCA feature:{label}] Loading {path}")
        arr = np.load(path, mmap_mode="r")
        print(f"[PCA feature:{label}] shape={arr.shape}")
        pca_feature_entries.append(PCAFeatureEntry(label, path, arr))

    pca_model_entries: List[PCAModelEntry] = []
    for spec in args.pca_model:
        label, path = parse_label_path(spec)
        print(f"[PCA model:{label}] Loading {path}")
        model = load_pickle_or_joblib(path)
        pca_model_entries.append(PCAModelEntry(label, path, model))

    transform = make_image_transform(args.image_size)
    rows: List[Dict] = []
    skipped = []

    expected_patches = (args.image_size // args.patch_size) ** 2
    requested_groups = [g.lower() for g in args.ckpt_groups]

    with torch.no_grad():
        for pos, idx in enumerate(indices):
            count = pos + 1
            ex = ds[int(idx)]
            filename = ex.get("filename", str(idx))
            instances = ex.get("instances", [])
            objects = ex.get("objects", [])

            seg_id = build_instance_id_map(
                instances,
                objects,
                image_size=args.image_size,
                mask_rule=args.mask_rule,
                sort_by_depth=args.sort_by_depth,
            )
            if seg_id is None:
                skipped.append({"idx": idx, "filename": filename, "reason": "no_instances"})
                continue

            patch_labels, valid, purities = label_map_to_patch_labels(
                seg_id,
                image_size=args.image_size,
                patch_size=args.patch_size,
                purity_threshold=args.purity_threshold,
            )
            if valid.sum() < 3 or len(np.unique(patch_labels[valid])) < 2:
                skipped.append({"idx": idx, "filename": filename, "reason": "too_few_valid_patches_or_instances", "valid": int(valid.sum())})
                continue

            pil = ex["image"].convert("RGB")
            image_tensor = transform(pil).unsqueeze(0)
            patch_tokens = dino.patch_tokens(image_tensor)  # [P, D]
            if patch_tokens.shape[0] != expected_patches:
                raise RuntimeError(f"DINO produced {patch_tokens.shape[0]} patches, expected {expected_patches}. Check image_size/patch_size/model.")

            feature_sets: List[Tuple[str, str, str, torch.Tensor]] = []
            # Tuple: feature_name, method, representation, tensor
            if args.include_raw_dino:
                feature_sets.append(("Raw DINO", "Raw DINO", "Raw", patch_tokens.detach().cpu()))

            for entry in pca_feature_entries:
                feat = get_precomputed_patch_features(entry, int(idx), pos, len(ds), len(indices))
                if feat.shape[0] != expected_patches:
                    raise RuntimeError(f"PCA feature {entry.label} produced {feat.shape[0]} patches, expected {expected_patches}")
                feature_sets.append((entry.label, entry.label, "PCA", feat.detach().cpu()))

            for entry in pca_model_entries:
                feat = transform_with_pca_model(entry.model, patch_tokens)
                if feat.shape[0] != expected_patches:
                    raise RuntimeError(f"PCA model {entry.label} produced {feat.shape[0]} patches, expected {expected_patches}")
                feature_sets.append((entry.label, entry.label, "PCA", feat.detach().cpu()))

            for sae_entry in sae_entries:
                codes = sae_entry.ae.encode(patch_tokens.to(args.device), use_threshold=False).detach()
                if "full" in requested_groups:
                    feature_sets.append((f"{sae_entry.label} Full", sae_entry.label, "Full", codes.detach().cpu()))
                for group_name, (s, e) in sae_entry.group_slices.items():
                    if group_name.lower() in requested_groups:
                        feature_sets.append((f"{sae_entry.label} {group_name}", sae_entry.label, group_name, codes[:, s:e].detach().cpu()))

            for feat_name, method, representation, feat in feature_sets:
                m = feature_metrics(feat, patch_labels, valid, precision_k=args.precision_k)
                if m is None:
                    continue
                row = {
                    "image_index": int(idx),
                    "filename": filename,
                    "feature": feat_name,
                    "method": method,
                    "representation": representation,
                    "valid_patches_total": int(valid.sum()),
                    "valid_instances_total": int(len(np.unique(patch_labels[valid]))),
                    "purity_threshold": args.purity_threshold,
                    "mask_rule": args.mask_rule,
                }
                row.update(m)
                rows.append(row)

            if args.save_debug and count <= 5:
                side = args.image_size // args.patch_size
                grid = patch_labels.reshape(side, side)
                plt.figure(figsize=(4, 4))
                plt.imshow(grid, interpolation="nearest")
                plt.axis("off")
                plt.title(filename)
                plt.tight_layout()
                plt.savefig(outdir / f"debug_patch_labels_{count:03d}.png", dpi=150)
                plt.close()

            if count % 25 == 0:
                print(f"[Progress] {count}/{len(indices)} images, rows={len(rows)}, skipped={len(skipped)}")

    per_image_csv = outdir / "per_image_metrics.csv"
    write_csv(per_image_csv, rows)
    write_csv(outdir / "skipped_images.csv", skipped)

    if not rows:
        raise RuntimeError("No valid rows produced. Try lowering --purity_threshold or checking mask format.")

    features = []
    for r in rows:
        if r["feature"] not in features:
            features.append(r["feature"])
    metrics = ["same_sim", "diff_sim", "gap", f"precision_at_{args.precision_k}"]
    summary: Dict[str, Dict[str, float]] = {}
    for feat in features:
        rs = [r for r in rows if r["feature"] == feat]
        summary[feat] = {
            "n_images": len(rs),
            "method": rs[0]["method"],
            "representation": rs[0]["representation"],
        }
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            summary[feat][f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            summary[feat][f"{metric}_sem"] = sem(vals) if vals.size else float("nan")

    summary_rows = []
    for feat, vals in summary.items():
        row = {"feature": feat}
        row.update(vals)
        summary_rows.append(row)
    write_csv(outdir / "summary_metrics.csv", summary_rows)

    # Paired comparisons for significance testing and plot annotations.
    # The statistical unit is one ADE20K image: for each pair of features, we compare
    # per-image metric values on the common set of images using paired tests.
    by_feat_img: Dict[str, Dict[int, Dict]] = {feat: {} for feat in features}
    feature_meta: Dict[str, Tuple[str, str]] = {}
    for r in rows:
        by_feat_img[r["feature"]][int(r["image_index"])] = r
        feature_meta[r["feature"]] = (str(r["method"]), str(r["representation"]))

    pairwise_rows: List[Dict] = []
    candidate_pairs = []

    # Main plot annotation: compare every non-Raw feature against Raw DINO if Raw DINO exists.
    if "Raw DINO" in features:
        for f in features:
            if f != "Raw DINO":
                candidate_pairs.append((f, "Raw DINO", "vs_raw_dino"))

    # Additional diagnostics: compare G0/G1 against Full within each SAE checkpoint.
    for sae_entry in sae_entries:
        full_name = f"{sae_entry.label} Full"
        for g in ["G0", "G1"]:
            g_name = f"{sae_entry.label} {g}"
            if full_name in features and g_name in features:
                candidate_pairs.append((g_name, full_name, "within_checkpoint_vs_full"))

    seen_pairs = set()
    for a_name, b_name, comparison_type in candidate_pairs:
        if (a_name, b_name, comparison_type) in seen_pairs or a_name not in by_feat_img or b_name not in by_feat_img:
            continue
        seen_pairs.add((a_name, b_name, comparison_type))
        common = sorted(set(by_feat_img[a_name]) & set(by_feat_img[b_name]))
        method_a, rep_a = feature_meta.get(a_name, ("", ""))
        method_b, rep_b = feature_meta.get(b_name, ("", ""))
        for metric in ["gap", f"precision_at_{args.precision_k}"]:
            a = np.array([float(by_feat_img[a_name][i][metric]) for i in common], dtype=float)
            b = np.array([float(by_feat_img[b_name][i][metric]) for i in common], dtype=float)
            st = paired_stats(a, b)
            row = {
                "metric": metric,
                "comparison_type": comparison_type,
                "feature_a": a_name,
                "feature_b": b_name,
                "method_a": method_a,
                "representation_a": rep_a,
                "method_b": method_b,
                "representation_b": rep_b,
            }
            row.update(st)
            pairwise_rows.append(row)

    # Holm correction is applied separately for each metric and comparison family.
    for metric in ["gap", f"precision_at_{args.precision_k}"]:
        for comparison_type in sorted({r["comparison_type"] for r in pairwise_rows}):
            idxs = [i for i, r in enumerate(pairwise_rows) if r["metric"] == metric and r["comparison_type"] == comparison_type]
            adjusted = holm_bonferroni([pairwise_rows[i].get("paired_t_p", float("nan")) for i in idxs])
            for i, p_adj in zip(idxs, adjusted):
                pairwise_rows[i]["paired_t_p_holm"] = p_adj
                pairwise_rows[i]["stars"] = p_to_stars(p_adj)

    write_csv(outdir / "pairwise_tests.csv", pairwise_rows)

    comparisons = {
        f"{r['feature_a']} vs {r['feature_b']} on {r['metric']}": {
            k: v for k, v in r.items()
            if k not in {"feature_a", "feature_b", "metric"}
        }
        for r in pairwise_rows
    }

    def make_sig_lookup(metric_name: str) -> Dict[Tuple[str, str], str]:
        lookup = {}
        for r in pairwise_rows:
            if r["metric"] != metric_name or r["comparison_type"] != "vs_raw_dino":
                continue
            # Only mark non-Raw feature bars. Stars indicate a Holm-corrected paired t-test vs Raw DINO.
            lookup[(r["method_a"], r["representation_a"])] = str(r.get("stars", ""))
        return lookup

    result = {
        "args": vars(args),
        "checkpoints": [
            {"label": e.label, "path": e.path, "dict_size": e.dict_size, "group_sizes": e.group_sizes, "k": e.k}
            for e in sae_entries
        ],
        "pca_features": [{"label": e.label, "path": e.path, "shape": list(e.array.shape)} for e in pca_feature_entries],
        "pca_models": [{"label": e.label, "path": e.path} for e in pca_model_entries],
        "n_rows": len(rows),
        "n_skipped": len(skipped),
        "summary": summary,
        "paired_comparisons": comparisons,
    }
    with (outdir / "summary.json").open("w") as f:
        json.dump(result, f, indent=2)

    # Flat plots for backwards compatibility.
    plot_metric_flat(summary, "gap", outdir / "segment_alignment_gap_flat.png", "Same-instance minus different-instance cosine similarity")
    plot_metric_flat(summary, f"precision_at_{args.precision_k}", outdir / f"precision_at_{args.precision_k}_flat.png", f"Same-instance retrieval precision@{args.precision_k}")

    # Grouped plots requested by the user.
    sig_note = "Stars: Holm-corrected paired t-test vs Raw DINO (* p<0.05, ** p<0.01, *** p<0.001)."
    plot_metric_grouped(
        rows,
        "gap",
        outdir / "segment_alignment_gap_grouped.png",
        "ADE20K segment alignment gap",
        sig_lookup=make_sig_lookup("gap"),
        sig_note=sig_note,
    )
    plot_metric_grouped(
        rows,
        f"precision_at_{args.precision_k}",
        outdir / f"precision_at_{args.precision_k}_grouped.png",
        f"ADE20K same-instance retrieval precision@{args.precision_k}",
        sig_lookup=make_sig_lookup(f"precision_at_{args.precision_k}"),
        sig_note=sig_note,
    )

    print("\nDone.")
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {outdir / 'summary_metrics.csv'}")
    print(f"Saved: {outdir / 'summary.json'}")
    print(f"Saved: {outdir / 'pairwise_tests.csv'}")
    print(f"Saved: {outdir / 'segment_alignment_gap_grouped.png'}")
    print(f"Saved: {outdir / f'precision_at_{args.precision_k}_grouped.png'}")


if __name__ == "__main__":
    main()