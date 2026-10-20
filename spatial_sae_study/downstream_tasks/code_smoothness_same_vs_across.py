"""Measure spatial smoothness of SAE codes against an across-image null.

For every SAE arm and code group (Full/G0/G1), this script compares metrics on
the true 16x16 code map of each image with a matched mosaic.  A mosaic keeps
spatial position and every latent's empirical marginal distribution fixed, but
draws each grid cell from a different image.  It therefore destroys coherence
specific to one image without using an all-zero or Gaussian null.

The five metric families are deliberately reported separately:
  cosine     mean cosine similarity of horizontal/vertical neighbours
  fourier    fraction of mean-centred spectral energy at low spatial frequency
  wavelet    one-level 2-D Haar approximation-energy fraction
  multiscale mean cosine similarity at offsets 1, 2, and 4 patches
  lipschitz  normalized local code change per one-patch displacement (lower=smoother)

For the first four metrics, effect = same_image - across_image.  For Lipschitz,
effect = across_image - same_image, so positive always means excess within-image
smoothness.  The unit of inference is an image; percentile bootstrap confidence
intervals and a two-sided bootstrap sign p-value are computed over images.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import ttest_rel, wilcoxon

THIS = Path(__file__).resolve()
REPO = Path("/n/home12/binxuwang/T-SAE-Spatial")
for p in [THIS.parent, REPO / "scripts", REPO / "temporal-saes" / "dictionary_learning"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fvu_compare import BlockDINO, T, load_sae  # noqa: E402
from repro_utils import check_gpu_match, gpu_fingerprint  # noqa: E402


def image_bytes(row):
    x = row["image"]
    return x["bytes"] if isinstance(x, dict) else x


def cosine_pairs(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def cosine_pairs_nonzero(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine only over patch pairs for which both group codes are active.

    This prevents a zero G0/G1 vector at a patch from being silently entered as
    cosine zero.  It is reported beside (not instead of) ordinary cosine, since
    conditioning on activity answers a different question.
    """
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    valid = (na > 1e-8) & (nb > 1e-8)
    if not np.any(valid):
        return float("nan")
    num = np.sum(a[valid] * b[valid], axis=-1)
    return float(np.mean(num / (na[valid] * nb[valid])))


def shifted_pairs(z: np.ndarray, d: int):
    """Horizontal and vertical pairs at displacement d; z is [H,W,D]."""
    return ((z[:, :-d], z[:, d:]), (z[:-d, :], z[d:, :]))


def cosine_at(z: np.ndarray, d: int) -> float:
    return float(np.mean([cosine_pairs(a, b) for a, b in shifted_pairs(z, d)]))


def metric_cosine(z: np.ndarray) -> float:
    return cosine_at(z, 1)


def metric_cosine_nonzero(z: np.ndarray) -> float:
    vals = []
    for a, b in shifted_pairs(z, 1):
        vals.append(cosine_pairs_nonzero(a, b))
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")


def metric_activation_density(z: np.ndarray) -> float:
    """Fraction of scalar code entries that are nonzero (a control, not smoothness)."""
    return float(np.mean(np.abs(z) > 1e-8))


def metric_multiscale(z: np.ndarray) -> float:
    return float(np.mean([cosine_at(z, d) for d in (1, 2, 4)]))


def metric_lipschitz(z: np.ndarray) -> float:
    vals = []
    for a, b in shifted_pairs(z, 1):
        change = np.linalg.norm(a - b, axis=-1)
        scale = 0.5 * (np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1))
        vals.append(np.mean(change / np.maximum(scale, 1e-8)))
    return float(np.mean(vals))


def metric_fourier(z: np.ndarray, cutoff: float = 0.25) -> float:
    """Low-frequency energy fraction after removing each latent's spatial mean.

    cutoff is cycles/pixel; 0.25 is half the Nyquist frequency.  The DC bin is
    removed by centring and is not allowed to make sparse positive codes look
    artificially smooth.
    """
    x = z.astype(np.float32) - z.mean(axis=(0, 1), keepdims=True)
    f = np.fft.fft2(x, axes=(0, 1))
    e = np.abs(f) ** 2
    fy = np.fft.fftfreq(z.shape[0])[:, None]
    fx = np.fft.fftfreq(z.shape[1])[None, :]
    low = (fx * fx + fy * fy) <= cutoff * cutoff
    return float(e[low].sum() / max(e.sum(), 1e-12))


def metric_wavelet(z: np.ndarray) -> float:
    """One-level orthonormal Haar coarse-scale energy fraction, mean-centred."""
    x = z.astype(np.float32) - z.mean(axis=(0, 1), keepdims=True)
    a = x[0::2, 0::2]; b = x[0::2, 1::2]
    c = x[1::2, 0::2]; d = x[1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    coarse = np.square(ll).sum()
    total = coarse + np.square(lh).sum() + np.square(hl).sum() + np.square(hh).sum()
    return float(coarse / max(total, 1e-12))


METRICS = {
    "cosine": metric_cosine,
    "cosine_nonzero": metric_cosine_nonzero,
    "fourier": metric_fourier,
    "wavelet": metric_wavelet,
    "multiscale": metric_multiscale,
    "lipschitz": metric_lipschitz,
    "activation_density": metric_activation_density,
}


def across_mosaic(codes: np.ndarray, target: int, rep: int, seed: int) -> np.ndarray:
    """At every position take the same position from another image.

    Donor identities are independently permuted at every grid cell and deranged
    relative to target.  This preserves coordinate-specific distributions while
    breaking the fact that neighbouring cells came from one image.
    """
    n, h, w, _ = codes.shape
    rng = np.random.default_rng(seed + 1000003 * rep + 9176 * target)
    out = np.empty_like(codes[target])
    for y in range(h):
        for x in range(w):
            donor = int(rng.integers(n - 1))
            if donor >= target:
                donor += 1
            out[y, x] = codes[donor, y, x]
    return out


def bootstrap(x: np.ndarray, n_boot: int, seed: int):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.mean(x[rng.integers(0, n, size=(n_boot, n))], axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))
    return float(x.mean()), float(x.std(ddof=1)), float(lo), float(hi), float(p)


def signflip_p(x: np.ndarray, n_perm: int, seed: int) -> float:
    """Monte-Carlo two-sided paired randomization test over images."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    obs = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    # Chunked to avoid allocating n_perm x n_images at once.
    while done < n_perm:
        k = min(5000, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, len(x)))
        exceed += int(np.sum(np.abs((signs * x).mean(1)) >= obs))
        done += k
    return float((exceed + 1) / (n_perm + 1))


def paired_arm_comparisons(raw: pd.DataFrame, n_boot: int, n_perm: int, seed: int):
    """Spatial-baseline paired comparisons for all three reported quantities."""
    rows = []
    for (group, metric), d in raw.groupby(["group", "metric"]):
        for quantity in ("same_image", "across_image", "effect"):
            p = d.pivot(index="image", columns="arm", values=quantity)
            if not {"spatial", "baseline"}.issubset(p.columns):
                continue
            p = p[["spatial", "baseline"]].dropna()
            delta = (p.spatial - p.baseline).to_numpy()
            mean, sd, lo, hi, boot_p = bootstrap(delta, n_boot, seed + len(rows))
            tt = ttest_rel(p.spatial, p.baseline)
            try:
                wp = float(wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError:
                wp = 1.0
            rows.append(dict(group=group, metric=metric, quantity=quantity,
                             contrast="spatial_minus_baseline", n_images=len(delta),
                             spatial_mean=float(p.spatial.mean()),
                             baseline_mean=float(p.baseline.mean()), paired_diff=mean,
                             paired_sd=sd, ci95_low=lo, ci95_high=hi,
                             paired_t_p=float(tt.pvalue), wilcoxon_p=wp,
                             signflip_p=signflip_p(delta, n_perm, seed + 1000 + len(rows)),
                             bootstrap_p=boot_p))
    return pd.DataFrame(rows)


@torch.no_grad()
def encode_images(ae, dino, rows, side: int, device: str) -> np.ndarray:
    zs = []
    for j, (_, row) in enumerate(rows.iterrows()):
        img = Image.open(io.BytesIO(image_bytes(row))).convert("RGB")
        tok = dino.patch_tokens(T(img).unsqueeze(0))
        if tok.shape[0] != side * side:
            raise RuntimeError(f"expected {side*side} patches, got {tok.shape[0]}")
        z = ae.encode(tok, use_threshold=False).float().cpu().numpy()
        zs.append(z.reshape(side, side, -1).astype(np.float16))
        if (j + 1) % 25 == 0:
            print(f"  encoded {j+1}/{len(rows)}", flush=True)
    return np.stack(zs)


def evaluate(codes: np.ndarray, arm: str, group: str, null_reps: int, seed: int):
    records = []
    for i in range(len(codes)):
        z = codes[i].astype(np.float32)
        same = {name: fn(z) for name, fn in METRICS.items()}
        null = {name: [] for name in METRICS}
        for r in range(null_reps):
            zn = across_mosaic(codes, i, r, seed).astype(np.float32)
            for name, fn in METRICS.items():
                null[name].append(fn(zn))
        for name in METRICS:
            across = float(np.mean(null[name]))
            effect = (across - same[name]) if name == "lipschitz" else (same[name] - across)
            records.append(dict(image=i, arm=arm, group=group, metric=name,
                                same_image=same[name], across_image=across, effect=effect))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="ARM:PATH")
    ap.add_argument("--dino_block", type=int, required=True)
    ap.add_argument("--parquet", default=str(REPO / "imagenet_val_2k.parquet"))
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--side", type=int, default=16)
    ap.add_argument("--null_reps", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--n_perm", type=int, default=100000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print("[GPU]", gpu_fingerprint(args.device), flush=True)
    specs = [(s.split(":", 1)[0], Path(s.split(":", 1)[1])) for s in args.ckpt]
    check_gpu_match([p.parent for _, p in specs])
    rows = pd.read_parquet(args.parquet).sample(args.n_images, random_state=args.sample_seed)
    dino = BlockDINO(args.device, dino_block=args.dino_block)
    all_records = []

    for arm, path in specs:
        ae, groups = load_sae(str(path), args.device)
        g0 = int(groups[0])
        print(f"\n[{arm}] groups={groups} encoding", flush=True)
        full = encode_images(ae, dino, rows, args.side, args.device)
        slices = {"Full": full, "G0": full[..., :g0], "G1": full[..., g0:]}
        for group, codes in slices.items():
            print(f"[{arm}/{group}] shape={codes.shape}", flush=True)
            all_records.extend(evaluate(codes, arm, group, args.null_reps,
                                        args.sample_seed + 100 * args.dino_block))
        del full

    raw = pd.DataFrame(all_records)
    raw.insert(0, "block", args.dino_block)
    raw.to_csv(out / "smoothness_same_vs_across_raw.csv", index=False)
    summary = []
    for (arm, group, metric), d in raw.groupby(["arm", "group", "metric"]):
        mean, sd, lo, hi, p = bootstrap(d.effect.to_numpy(), args.n_boot,
                                        args.sample_seed + 31 * args.dino_block)
        summary.append(dict(block=args.dino_block, arm=arm, group=group, metric=metric,
                            n_images=len(d), same_mean=d.same_image.mean(),
                            across_mean=d.across_image.mean(), effect=mean, image_sd=sd,
                            ci95_low=lo, ci95_high=hi, bootstrap_p=p))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "smoothness_same_vs_across_summary.csv", index=False)
    paired = paired_arm_comparisons(raw, args.n_boot, args.n_perm,
                                    args.sample_seed + 47 * args.dino_block)
    paired.insert(0, "block", args.dino_block)
    paired.to_csv(out / "smoothness_spatial_vs_baseline_paired.csv", index=False)
    with open(out / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("\n", summary.to_string(index=False), flush=True)
    print("\n[Spatial - baseline paired comparisons]\n",
          paired.to_string(index=False), flush=True)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
