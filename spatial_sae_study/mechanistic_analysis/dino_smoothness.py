"""
How spatially smooth are real DINOv2 features, actually?

The premise behind "the spatial SAE underperforms because DINOv2's features are
all smooth" has so far only been inferred. This measures it directly: the lag-1/2/4
spatial autocorrelation of each feature's own activation map, for

  * raw DINO patch-token dimensions (768 per block)
  * the SAE latents of each arm (spatial / baseline / vanilla)
  * PCA components

against the synthetic toy's reference points, where "smooth" was built to sit at
lag-1 ~ 0.47 and "rough" at ~ -0.004.

The point is the DISTRIBUTION, not the mean. If real features span a wide range
of autocorrelation, then "there is nothing to separate" is ruled out and the
spatial SAE's weak showing needs a different explanation. If they are uniformly
smooth, the premise is supported.
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache"
os.environ["TORCH_HOME"] = _CACHE
Path(_CACHE + "/hub/checkpoints").mkdir(parents=True, exist_ok=True)

import argparse
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for p in [THIS.parent, THIS.parent.parent, THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE
from fvu_compare import BlockDINO, T
from repro_utils import gpu_fingerprint, check_gpu_match


def load_sae(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "ae_state_dict" in sd:
        sd = sd["ae_state_dict"]
    act_dim, dict_size = sd["W_enc"].shape
    gs = [int(x) for x in sd["group_sizes"].tolist()]
    ae = TemporalMatryoshkaBatchTopKSAE(activation_dim=act_dim, dict_size=dict_size,
                                        k=int(sd["k"].item()), group_sizes=gs, temporal=False)
    ae.load_state_dict(sd, strict=False)
    return ae.eval().to(device), gs


def autocorr_per_feature(maps, lags=(1, 2, 4)):
    """maps: [n_img, F, side, side] -> {lag: [F]} averaged over images.

    Images where a feature is constant contribute nothing (a constant map has no
    defined autocorrelation); features that are never variable come back as NaN
    rather than being silently counted as 0.
    """
    x = maps - maps.mean(dim=(-2, -1), keepdim=True)
    v = (x ** 2).mean(dim=(-2, -1))
    ok = v > 1e-10
    out = {}
    for L in lags:
        h = (x[..., :, :-L] * x[..., :, L:]).mean(dim=(-2, -1))
        w = (x[..., :-L, :] * x[..., L:, :]).mean(dim=(-2, -1))
        ac = ((h + w) / 2) / (v + 1e-12)
        num = (ac * ok).sum(0); den = ok.sum(0)
        r = num / den.clamp(min=1)
        r[den == 0] = float("nan")
        out[L] = r.cpu().numpy()
    return out, ok.sum(0).cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="LABEL:PATH")
    ap.add_argument("--pca_model", default=None)
    ap.add_argument("--dino_block", type=int, default=8)
    ap.add_argument("--n_images", type=int, default=300)
    ap.add_argument("--min_active_images", type=int, default=10,
                    help="a latent must be variable in at least this many images to be scored")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    fp = gpu_fingerprint(args.device)
    print(f"[GPU] {fp['gpu_name']} (cc {fp['capability']}, torch {fp['torch']})", flush=True)
    if args.ckpt:
        check_gpu_match([Path(c.split(":", 1)[1]).parent for c in args.ckpt])

    import joblib
    df = pd.read_parquet(args.parquet).sample(args.n_images, random_state=0)
    dino = BlockDINO(args.device, dino_block=args.dino_block)
    saes = {}
    for spec in args.ckpt:
        lab, path = spec.split(":", 1)
        saes[lab] = load_sae(path, args.device)
    pca = joblib.load(args.pca_model) if args.pca_model else None

    reps = {}
    for _, r in df.iterrows():
        d = r["image"]; b = d["bytes"] if isinstance(d, dict) else d
        try:
            tok = dino.patch_tokens(T(Image.open(io.BytesIO(b)).convert("RGB")).unsqueeze(0))
        except Exception:
            continue
        side = int(np.sqrt(tok.shape[0]))
        reps.setdefault("raw_DINO", []).append(tok.T.reshape(-1, side, side).cpu())
        for lab, (ae, _) in saes.items():
            z = ae.encode(tok, use_threshold=False)
            reps.setdefault(lab, []).append(z.T.reshape(-1, side, side).cpu())
        if pca is not None:
            p = torch.tensor(pca.transform(tok.cpu().numpy()), dtype=torch.float32)
            reps.setdefault("PCA", []).append(p.T.reshape(-1, side, side))

    rows, dists = [], {}
    for lab, chunks in reps.items():
        M = torch.stack(chunks, 0).to(args.device)          # [n_img, F, side, side]
        ac, nseen = autocorr_per_feature(M)
        keep = nseen >= args.min_active_images
        a1 = ac[1][keep]
        dists[lab] = a1
        rows.append(dict(representation=lab, n_features=M.shape[1],
                         n_scored=int(keep.sum()), frac_scored=float(keep.mean()),
                         lag1_mean=float(np.nanmean(a1)), lag1_median=float(np.nanmedian(a1)),
                         lag1_p10=float(np.nanpercentile(a1, 10)),
                         lag1_p90=float(np.nanpercentile(a1, 90)),
                         lag2_mean=float(np.nanmean(ac[2][keep])),
                         lag4_mean=float(np.nanmean(ac[4][keep])),
                         frac_below_0p1=float(np.nanmean(a1 < 0.1)),
                         frac_above_0p4=float(np.nanmean(a1 > 0.4)))
                    )
        del M
        torch.cuda.empty_cache()
        print(f"  {lab:14s} scored {int(keep.sum())}/{len(keep)}  "
              f"lag1 median {np.nanmedian(a1):+.3f}  p10 {np.nanpercentile(a1,10):+.3f}  "
              f"p90 {np.nanpercentile(a1,90):+.3f}", flush=True)

    res = pd.DataFrame(rows); res.to_csv(out / "dino_smoothness.csv", index=False)
    np.savez_compressed(out / "dino_smoothness_dists.npz", **dists)
    print("\n" + res.to_string(index=False))

    colors = {"raw_DINO": "#555555", "spatial": "#1f77b4", "baseline": "#d62728",
              "vanilla": "#9467bd", "PCA": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for lab, v in dists.items():
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        ax.hist(v, bins=60, density=True, histtype="step", linewidth=2,
                color=colors.get(lab), label=f"{lab} (n={len(v)})")
    ax.axvline(0.47, ls="--", c="#1f77b4", lw=1.6, label="toy SMOOTH reference (0.47)")
    ax.axvline(-0.004, ls="--", c="#d62728", lw=1.6, label="toy ROUGH reference (-0.004)")
    ax.set_xlabel("lag-1 spatial autocorrelation of the feature's own activation map")
    ax.set_ylabel("density")
    ax.set_title(f"How smooth are real DINOv2 features? (block {args.dino_block}, "
                 f"{args.n_images} images)\ndashed lines = the synthetic toy's two design points")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "dino_smoothness.png", dpi=150, bbox_inches="tight")
    print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
