"""Smoothness / low-rank / energy across DINOv2 blocks, LayerNorm held ON.

WHY THIS EXISTS
---------------
`shortcut_analysis.py` (job 41630729) measured the two shortcuts at BLOCK 0 only,
across three conditions (post-LN / no-norm / SigLIP2 raw). Wendy asked on Sep 8 2026
for the complementary cut: hold the backbone at DINOv2 and the norm at post-LN, and
vary the BLOCK. Nothing in the repo has that -- `backbone_screen.csv` has per-block
smoothness DISTRIBUTIONS (lag1_median, spread, kurtosis) but no correlations, and
`dino_smoothness_b{0,4,8,11}` has per-channel contiguity but never paired it with
energy or low-rank participation.

THE THREE QUANTITIES, per raw DINO channel (768 of them, one number each)
------------------------------------------------------------------------
  contig    lag-1 spatial autocorrelation of that channel's own 16x16 activation
            map, averaged over images. Uses the SHARED `autocorr_per_feature` from
            dino_smoothness.py, not a re-implementation.
  energy    that channel's variance over all patch tokens.
  lowrank   1 - var(x - PCA_k(x)) / var(x) for that channel: how well the top k=50
            principal components of the token matrix capture it. Arm-independent.

WHAT IS THE CONTROL
-------------------
Block 0 is a LIVE control, not a reference read off disk: same backbone, same
post-LN convention, same parquet, same n_images=200, same sample_seed=0, same
n_pcs=50 as job 41630729. It must reproduce

    rho(contig, energy)  = +0.401        rho(contig, lowrank) = +0.783

If block 0 does not land near those, nothing else in the output means anything and
the run should be discarded rather than interpreted. The check is printed AND
asserted at a loose tolerance so a silent convention drift cannot pass.

WHAT IS NEW HERE BESIDES THE BLOCK AXIS
---------------------------------------
Two PARTIAL correlations, which the block-0 run never computed:

    rho(contig, lowrank | energy)   and   rho(contig, energy | lowrank)

These speak directly to the Sep-8 question of whether "smooth channels are low-rank"
and "smooth channels are high-variance" are separable claims. Partial Spearman is
computed from the three pairwise Spearman rhos by the standard first-order formula,
with df = n - 3.

A CAVEAT THE TABLE CANNOT CARRY ITSELF
--------------------------------------
Deep blocks are homogeneously smooth (recorded lag1 p90-p10: 0.351 / 0.193 / 0.075 /
0.054 at blocks 0/4/8/11). A correlation computed over a variable with almost no
spread is not comparable to one computed where the spread is 6x larger, even when
both are "significant". The spread column is printed on every row for exactly that
reason, and a split-half over images is run so instability is visible rather than
inferred.
"""
from __future__ import annotations

import os
from pathlib import Path

# Job 45454047 died at TIMEOUT having used 14 s of CPU in 40 min: a Lustre stall on
# holylfs06 blocked the backbone load. TSAE_TORCH_CACHE lets the launcher stage the
# hub dir onto node-local disk so a shared-filesystem hiccup cannot cost a walltime.
_CACHE = os.environ.get(
    "TSAE_TORCH_CACHE",
    "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache")
os.environ["TORCH_HOME"] = _CACHE
os.environ.setdefault("HF_HOME", "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
Path(_CACHE + "/hub/checkpoints").mkdir(parents=True, exist_ok=True)

import argparse
import io
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for _p in [THIS.parent, THIS.parent.parent,
           THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dino_smoothness import autocorr_per_feature  # noqa: E402
from new_spatial_train_triplet import (  # noqa: E402
    BlockAwareDINOFeatureExtractor,
    build_transform,
)
from repro_utils import gpu_fingerprint  # noqa: E402

# dino_smoothness.py re-points TORCH_HOME and torch.hub at its own hardcoded
# holylfs06 path when imported, so re-assert ours AFTER the imports above.
os.environ["TORCH_HOME"] = _CACHE
torch.hub.set_dir(_CACHE + "/hub")

DINO_REPO = _CACHE + "/hub/facebookresearch_dinov2_main"

# Recorded at block 0 by job 41630729 (`shortcut_raw.csv`), reproduced exactly by
# the whitening run (job 42035857). These are the control targets.
CONTROL_B0 = {"rho_energy": 0.401, "rho_lowrank": 0.783}
CONTROL_TOL = 0.05


def img_bytes(row):
    d = row["image"]
    return d["bytes"] if isinstance(d, dict) else d


def _rho(a, b):
    r = stats.spearmanr(a, b)
    return float(r.correlation), float(r.pvalue)


def partial_spearman(r_ab, r_ac, r_bc, n):
    """First-order partial correlation of a,b given c, from Spearman rhos.

    Returns (rho, p). df = n - 3. Returns (nan, nan) if the denominator degenerates.
    """
    den = np.sqrt(max(1.0 - r_ac ** 2, 0.0) * max(1.0 - r_bc ** 2, 0.0))
    if den < 1e-12:
        return float("nan"), float("nan")
    r = (r_ab - r_ac * r_bc) / den
    r = float(np.clip(r, -1.0, 1.0))
    df = n - 3
    if df <= 0 or abs(r) >= 1.0:
        return r, float("nan")
    t = r * np.sqrt(df / max(1.0 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), df))


@torch.no_grad()
def channel_stats(maps, X, n_pcs, device):
    """maps [n_img, D, side, side], X [n_tok, D] -> (contig, energy, lowrank) each [D]."""
    ac, _ = autocorr_per_feature(maps.to(device), lags=(1,))
    contig = ac[1]

    Xn = X.float().cpu().numpy()
    energy = Xn.var(0)

    p = PCA(n_components=n_pcs, svd_solver="randomized", random_state=0).fit(Xn)
    recon = p.inverse_transform(p.transform(Xn))
    resid_var = (Xn - recon).var(0)
    lowrank = 1.0 - resid_var / np.maximum(energy, 1e-12)
    evr = float(p.explained_variance_ratio_.sum())
    return contig, energy, lowrank, evr


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="/n/home12/binxuwang/T-SAE-Spatial/imagenet_val_2k.parquet")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    print(f"[gpu] {gpu_fingerprint(dev)}", flush=True)

    # df.sample, never df.head -- the recorded rule, and the block-0 control depends
    # on drawing the SAME 200 images as job 41630729.
    rows = pd.read_parquet(args.parquet).sample(args.n_images, random_state=args.sample_seed)
    T = build_transform("dinov2", args.image_size)
    imgs = []
    for _, r in rows.iterrows():
        try:
            imgs.append(T(Image.open(io.BytesIO(img_bytes(r))).convert("RGB")))
        except Exception:
            pass
    imgs = torch.stack(imgs, 0)
    n_img = imgs.shape[0]
    print(f"[data] {n_img}/{args.n_images} images decoded", flush=True)

    # One ViT, reused across blocks -- 12 copies of the same backbone is the only
    # thing that would make this run expensive.
    ex = BlockAwareDINOFeatureExtractor(args.model, DINO_REPO, dev, dino_block=0, norm=True)

    half = n_img // 2
    recs, per_channel = [], {}

    for b in args.blocks:
        ex.dino_block = b
        toks = []
        for i in range(0, n_img, args.batch):
            toks.append(ex.patch_tokens(imgs[i:i + args.batch]).float().cpu())
        tok = torch.cat(toks, 0)                       # [n_img, n_patch, D]
        n_patch, D = tok.shape[1], tok.shape[2]
        side = int(round(n_patch ** 0.5))
        assert side * side == n_patch, f"block {b}: {n_patch} patches is not square"
        maps = tok.permute(0, 2, 1).reshape(n_img, D, side, side)

        contig, energy, lowrank, evr = channel_stats(
            maps, tok.reshape(-1, D), args.n_pcs, dev)
        ok = np.isfinite(contig) & np.isfinite(lowrank) & np.isfinite(energy)
        n = int(ok.sum())

        r_ce, p_ce = _rho(contig[ok], energy[ok])
        r_cl, p_cl = _rho(contig[ok], lowrank[ok])
        r_el, p_el = _rho(energy[ok], lowrank[ok])
        pr_cl, pp_cl = partial_spearman(r_cl, r_ce, r_el, n)   # contig~lowrank | energy
        pr_ce, pp_ce = partial_spearman(r_ce, r_cl, r_el, n)   # contig~energy  | lowrank

        # smooth vs rough tercile variance ratio -- the "8.3x" number's definition
        q = np.nanquantile(contig[ok], [1 / 3, 2 / 3])
        lo = energy[ok & (contig <= q[0])].mean()
        hi = energy[ok & (contig >= q[1])].mean()

        # split-half over IMAGES: instability shows up as the two halves disagreeing
        sh = []
        for sl in (slice(0, half), slice(half, n_img)):
            c2, e2, l2, _ = channel_stats(maps[sl], tok[sl].reshape(-1, D), args.n_pcs, dev)
            m2 = np.isfinite(c2) & np.isfinite(l2) & np.isfinite(e2)
            sh.append((_rho(c2[m2], e2[m2])[0], _rho(c2[m2], l2[m2])[0]))

        rec = dict(
            block=b, n_dims=n, n_img=n_img, side=side,
            lag1_median=float(np.nanmedian(contig[ok])),
            lag1_p90_minus_p10=float(np.nanpercentile(contig[ok], 90)
                                     - np.nanpercentile(contig[ok], 10)),
            frac_below_0p1=float(np.mean(contig[ok] < 0.1)),
            pca50_evr=evr,
            rho_contig_energy=r_ce, p_contig_energy=p_ce,
            rho_contig_lowrank=r_cl, p_contig_lowrank=p_cl,
            rho_energy_lowrank=r_el, p_energy_lowrank=p_el,
            partial_contig_lowrank_given_energy=pr_cl, p_partial_cl=pp_cl,
            partial_contig_energy_given_lowrank=pr_ce, p_partial_ce=pp_ce,
            var_smooth_over_rough=float(hi / max(lo, 1e-9)),
            rho_contig_energy_h1=sh[0][0], rho_contig_energy_h2=sh[1][0],
            rho_contig_lowrank_h1=sh[0][1], rho_contig_lowrank_h2=sh[1][1],
        )
        recs.append(rec)
        per_channel[f"b{b}_contig"] = contig
        per_channel[f"b{b}_energy"] = energy
        per_channel[f"b{b}_lowrank"] = lowrank

        print(f"b{b:<2d} lag1_med {rec['lag1_median']:+.3f}  spread "
              f"{rec['lag1_p90_minus_p10']:.3f} | rho(c,E) {r_ce:+.3f} "
              f"rho(c,LR) {r_cl:+.3f} rho(E,LR) {r_el:+.3f} | partial "
              f"c~LR|E {pr_cl:+.3f}  c~E|LR {pr_ce:+.3f} | var ratio "
              f"{rec['var_smooth_over_rough']:.2f}x | PCA50 EVR {evr:.3f}", flush=True)
        del maps, tok
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(recs)
    df.to_csv(out / "block_shortcut_sweep.csv", index=False)
    np.savez(out / "block_shortcut_per_channel.npz", **per_channel)

    # ── the control, stated loudly ────────────────────────────────────────
    if 0 in args.blocks:
        r0 = df[df.block == 0].iloc[0]
        de = abs(r0.rho_contig_energy - CONTROL_B0["rho_energy"])
        dl = abs(r0.rho_contig_lowrank - CONTROL_B0["rho_lowrank"])
        print(f"\n[control] block 0 vs job 41630729: "
              f"rho(c,E) {r0.rho_contig_energy:+.3f} vs +0.401 (dev {de:.3f}); "
              f"rho(c,LR) {r0.rho_contig_lowrank:+.3f} vs +0.783 (dev {dl:.3f})", flush=True)
        assert de < CONTROL_TOL and dl < CONTROL_TOL, (
            "block-0 control did NOT reproduce -- the feature convention drifted, so "
            "no row in this table may be interpreted. Do not relax this tolerance to "
            "make a run pass.")
        print("[control] PASSED", flush=True)

    print(f"\n[done] wrote {out}/block_shortcut_sweep.csv", flush=True)


if __name__ == "__main__":
    main()
