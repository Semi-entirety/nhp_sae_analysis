"""Why does `diag` OVERSHOOT instead of nulling the energy shortcut?

THE OBSERVATION (whiten_calib.py, job 42034391, class-diverse fit, control PASS)

    rho(contiguity, per-channel variance)   none  +0.401   ->  diag  -0.590
    variance ratio, most- vs least-contiguous third   7.91x  ->  0.31x

Dividing each channel by its own standard deviation should make every channel's
variance 1 and drive that rho to ~0. It does not: it reverses the sign with
p=3.6e-73, and the reversal is LARGE (a 3.2x gap the other way), so it is not
rank noise on a near-constant vector.

A first hypothesis -- that the std was fitted on a class-clustered sample -- was
tested and REFUTED: fixing the fit to 199 of 200 classes moved the number from
-0.573 to -0.590, i.e. not at all. The preprocessing is also identical on both
sides (Resize/ToTensor/Normalize, no augmentation), so that is not it either.

WHAT THIS SCRIPT SEPARATES

The transform divides by sigma fitted on the TRAINING parquet; the rho is then
measured on 200 held-out IMAGENET-VAL images. Two candidate causes:

  (1) DISTRIBUTION GAP. The two sets differ (train is a 200-class subset, val is
      drawn from 1000-class ImageNet val), so var_val(c)/var_train(c) is not 1 and
      its departure from 1 happens to track contiguity.
  (2) INTRINSIC / ESTIMATION. Something about the ratio itself, present even
      within one distribution. Contiguous channels are near-constant inside an
      image, so their variance is almost entirely BETWEEN-image: their effective
      sample size is the image count, not the token count, and their variance
      estimate is correspondingly less stable than a rough channel's.

The decisive contrast is a same-distribution held-out split:

      fit sigma on train half A  ->  measure rho on train half B

If that is ~0, cause (1); if it reproduces ~-0.6, cause (2). Both halves have the
same class composition and the same size, so nothing else differs.

Also reported, because they say WHICH quantity moves rather than just that the
ratio does: rho(contig, var) computed separately on each split, and the
between-image / within-image variance decomposition versus contiguity.

Nothing here changes any constant. It is a read-only diagnostic on the SAME
whiten_stats.npz that training will use.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

_CACHE = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache"
os.environ.setdefault("TORCH_HOME", _CACHE)
os.environ.setdefault("HF_HOME", "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/hf_cache")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for _p in [THIS.parent, THIS.parent.parent,
           THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dino_smoothness import autocorr_per_feature  # noqa: E402
from new_spatial_train_triplet import (  # noqa: E402
    BlockAwareDINOFeatureExtractor,
    _decode_image,
    build_transform,
)
from repro_utils import gpu_fingerprint  # noqa: E402

DINO_REPO = _CACHE + "/hub/facebookresearch_dinov2_main"
TRAIN_PARQUET = ("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/"
                 "T-SAE-Spatial/data/imagenet_data/train-00000-of-00001-18bc3231d015f1e8.parquet")
VAL_PARQUET = "/n/home12/binxuwang/T-SAE-Spatial/imagenet_val_2k.parquet"


def _rho(a, b):
    r = spearmanr(a, b)
    return float(r.correlation), float(r.pvalue)


@torch.no_grad()
def tokens_of(ex, T, df, dev, batch=16):
    """[n_img, n_tok, D] -- images kept separate so variance can be decomposed."""
    out, buf = [], []
    for _, r in df.iterrows():
        try:
            buf.append(T(_decode_image(r["image"])))
        except Exception:
            continue
        if len(buf) == batch:
            out.append(ex.patch_tokens(torch.stack(buf)).float().cpu())
            buf = []
    if buf:
        out.append(ex.patch_tokens(torch.stack(buf)).float().cpu())
    return torch.cat(out, 0)


def contiguity(t, dev):
    """Per-channel lag-1 autocorrelation on the patch grid. t: [n_img, n_tok, D]."""
    side = int(round(t.shape[1] ** 0.5))
    assert side * side == t.shape[1]
    maps = t.permute(0, 2, 1).reshape(t.shape[0], t.shape[2], side, side)
    ac, _ = autocorr_per_feature(maps.to(dev), lags=(1,))
    return ac[1]


def var_parts(t):
    """Total / between-image / within-image per-channel variance. t: [n_img, n_tok, D].

    Contiguous channels are near-constant inside an image, so their variance is
    almost all BETWEEN-image: effective sample size = image count, not token count.
    """
    per_img = t.mean(1)                          # [n_img, D]
    total = t.reshape(-1, t.shape[-1]).var(0, unbiased=False)
    between = per_img.var(0, unbiased=False)
    within = t.var(1, unbiased=False).mean(0)
    return (total.numpy().astype(np.float64),
            between.numpy().astype(np.float64),
            within.numpy().astype(np.float64))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", default=TRAIN_PARQUET)
    ap.add_argument("--val_parquet", default=VAL_PARQUET)
    ap.add_argument("--n_fit_images", type=int, default=1024)
    ap.add_argument("--n_val_images", type=int, default=200)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    print(f"[gpu] {gpu_fingerprint(dev)}", flush=True)

    ex = BlockAwareDINOFeatureExtractor("dinov2_vitb14", DINO_REPO, dev, dino_block=0, norm=True)
    T = build_transform("dinov2", 224)

    # Exactly the fit sample whiten_calib used, then split in half. Both halves are
    # random draws from the same 200-class file, so they differ ONLY by sampling.
    tr = pd.read_parquet(args.train_parquet)
    idx = np.random.default_rng(args.sample_seed).choice(
        len(tr), size=min(args.n_fit_images, len(tr)), replace=False)
    half = len(idx) // 2
    dfA, dfB = tr.iloc[sorted(idx[:half])], tr.iloc[sorted(idx[half:])]
    dfV = pd.read_parquet(args.val_parquet).sample(args.n_val_images,
                                                   random_state=args.sample_seed)
    print(f"[data] trainA {len(dfA)}  trainB {len(dfB)}  val {len(dfV)} images", flush=True)
    if "label" in tr.columns:
        print(f"[data] classes  A {dfA['label'].nunique()}  B {dfB['label'].nunique()}  "
              f"V {dfV['label'].nunique()}", flush=True)

    sets = {}
    for name, df in (("trainA", dfA), ("trainB", dfB), ("val", dfV)):
        t = tokens_of(ex, T, df, dev)
        tot, btw, wtn = var_parts(t)
        sets[name] = dict(contig=contiguity(t, dev), total=tot, between=btw, within=wtn,
                          n_img=t.shape[0])
        print(f"[{name}] {t.shape[0]} img x {t.shape[1]} tok x {t.shape[2]} chan", flush=True)
        del t
        torch.cuda.empty_cache()

    # The contiguity axis is a property of the substrate, not of a split; use the
    # val one so every number below is on the axis the recorded +0.401 was measured on.
    cv = sets["val"]["contig"]
    ok = np.isfinite(cv) & np.isfinite(sets["trainA"]["contig"]) & np.isfinite(sets["trainB"]["contig"])
    print(f"\n[axis] {ok.sum()} of {len(cv)} channels scored on all three splits", flush=True)
    ra, _ = _rho(cv[ok], sets["trainA"]["contig"][ok])
    rb, _ = _rho(cv[ok], sets["trainB"]["contig"][ok])
    print(f"[axis] contiguity is stable across splits: rho(val, trainA) {ra:+.3f}  "
          f"rho(val, trainB) {rb:+.3f}", flush=True)

    print("\n=== rho(contiguity, variance) measured WITHIN each split (no transform) ===")
    for name in ("trainA", "trainB", "val"):
        s = sets[name]
        r, p = _rho(cv[ok], s["total"][ok])
        rb_, _ = _rho(cv[ok], s["between"][ok])
        rw, _ = _rho(cv[ok], s["within"][ok])
        frac = s["between"] / np.maximum(s["total"], 1e-12)
        rf, _ = _rho(cv[ok], frac[ok])
        print(f"  {name:7s} total {r:+.3f} (p={p:.1e})   between {rb_:+.3f}   "
              f"within {rw:+.3f}   rho(contig, between/total) {rf:+.3f}", flush=True)

    print("\n=== the decisive contrast: divide by a std fitted on OTHER images ===")
    rows = []
    for fit, meas in (("trainA", "trainB"), ("trainB", "trainA"),
                      ("trainA", "val"), ("trainB", "val")):
        ratio = sets[meas]["total"] / np.maximum(sets[fit]["total"], 1e-12)
        r, p = _rho(cv[ok], ratio[ok])
        q = np.nanquantile(cv[ok], [1 / 3, 2 / 3])
        lo = float(ratio[ok][cv[ok] <= q[0]].mean())
        hi = float(ratio[ok][cv[ok] >= q[1]].mean())
        same = "same distribution" if meas.startswith("train") and fit.startswith("train") else "TRAIN -> VAL"
        print(f"  fit {fit:7s} -> measure {meas:7s}  rho {r:+.3f} (p={p:.1e})  "
              f"var ratio high/low {hi/max(lo,1e-12):.2f}x   [{same}]", flush=True)
        rows.append(dict(fit=fit, measure=meas, rho=r, p=p,
                         var_ratio_high_over_low=hi / max(lo, 1e-12), same_distribution=same))

    df = pd.DataFrame(rows)
    df.to_csv(out / "whiten_diag.csv", index=False)
    np.savez(out / "whiten_diag_channels.npz",
             contig_val=cv,
             **{f"{n}_{k}": sets[n][k] for n in sets for k in ("total", "between", "within")})
    (out / "whiten_diag.json").write_text(json.dumps(
        dict(gpu=gpu_fingerprint(dev), n_scored=int(ok.sum()),
             contig_stability=dict(trainA=ra, trainB=rb)), indent=2, default=str))
    print(f"\n[Saved] {out}", flush=True)


if __name__ == "__main__":
    main()
