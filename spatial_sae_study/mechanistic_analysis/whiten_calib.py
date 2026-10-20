"""Calibrate + qualify the channel-whitening arm BEFORE any GPU is spent training.

This is the same "measure the knob, then commit" step as token_scale_calib.py
(input_scale for the no-norm/SigLIP2 arms) and toy_shortcut_calib.py (the toy's
energy/low-rank injection gains). Both of those caught something: the first caught
that dropping LayerNorm silently rescales the contrastive/reconstruction balance
~2300x, the second caught that the low-rank knob is not separable from energy.

WHAT IT PRODUCES
----------------
1. `whiten_stats.npz` -- per-channel mean/std and the ZCA matrix of DINOv2 block 0
   post-LN tokens, fitted on the TRAINING parquet with the TRAINING transform.
   These are the constants training and every downstream analysis will load; they
   are never recomputed. (token_scale_calib made this mistake explicit: do NOT
   reuse numbers measured on the 150-image ImageNet-val set.)
2. `whiten_calib.csv` -- for each variant, everything needed to decide whether the
   arm is worth running and how to configure it.

FOUR THINGS IT HAS TO ANSWER, AND WHY EACH ONE CAN KILL THE EXPERIMENT
----------------------------------------------------------------------
(a) Does the transform actually sever the intended shortcut? `diag` should drive
    rho(contiguity, variance) to ~0. It is measured on HELD-OUT val images, not on
    the calibration set where it is 0 by construction, so the reported number is
    the residual that training will really see.
(b) Is there a smooth/rough axis LEFT afterwards? The whole point is to ask
    whether contrastive can sort on an axis matryoshka can no longer ride for
    free. If a transform flattens the contiguity distribution there is nothing to
    sort and the arm is vacuous. For `diag` this is provable -- the contiguity
    metric normalises each channel's map by its own mean/variance, so a positive
    per-channel rescale cannot move it -- and the script ASSERTS it element-wise
    rather than trusting the argument. For `zca` it is an open measurement.
(c) Is the result trainable? Reported with `concentration` from
    prenorm_scale_check.py: raw block 8 was ruled out because the top 1% of tokens
    carried 92.6% of the L2 loss. Equalising channel variance necessarily RAISES
    the weight of the small channels, which is the point, but it can also amplify
    a near-degenerate direction into a new outlier axis.
(d) What input_scale does it need? The contrastive term is scale-invariant while
    reconstruction is quadratic in token norm, so a transform that changes the
    token scale silently reweights the loss. Reference is the same one every
    block-0 checkpoint was trained at: dino_b0_postLN median 73.4468.

THE CONTROL, PRINTED BEFORE ANYTHING ELSE
-----------------------------------------
The `none` variant is the untouched post-LN substrate measured in this same
process on the same 200 images with the same n_pcs=50, so it must land on the
recorded shortcut_analysis values +0.401 (energy) and +0.783 (lowrank), with the
least- vs most-contiguous-third variance ratio near 8.3x. If the control does not
reproduce, no other row means anything.
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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for _p in [THIS.parent, THIS.parent.parent,
           THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from channel_transform import ChannelTransform, fit_channel_stats, sha256  # noqa: E402
from dino_smoothness import autocorr_per_feature  # noqa: E402
from new_spatial_train_triplet import (  # noqa: E402
    BlockAwareDINOFeatureExtractor,
    ParquetImageDataset,
    build_transform,
)
from prenorm_scale_check import concentration  # noqa: E402
from repro_utils import gpu_fingerprint  # noqa: E402

DINO_REPO = _CACHE + "/hub/facebookresearch_dinov2_main"
TRAIN_PARQUET = ("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/"
                 "T-SAE-Spatial/data/imagenet_data/train-00000-of-00001-18bc3231d015f1e8.parquet")
VAL_PARQUET = "/n/home12/binxuwang/T-SAE-Spatial/imagenet_val_2k.parquet"
# what every existing block-0 checkpoint was trained at (token_scale_calib.csv)
REF_TOKEN_NORM = 73.4468
# recorded shortcut_analysis values for the untouched post-LN substrate
RECORDED = dict(rho_energy=0.401, rho_lowrank=0.783, var_ratio=7.9)


def _rho(a, b):
    r = spearmanr(a, b)
    return float(r.correlation), float(r.pvalue)


def img_bytes(row):
    d = row["image"]
    return d["bytes"] if isinstance(d, dict) else d


@torch.no_grad()
def train_tokens(ex, transform, parquet, n_images, batch, device, seed=0):
    """[N, D] pooled tokens from the TRAINING parquet with the TRAINING transform.

    The images are a RANDOM sample of the whole file, never its first N rows. This
    parquet is perfectly class-SORTED (200 classes x 500 images, labels 0,0,...,199)
    so `range(1024)` is 3 classes, and the quantity being fitted here is per-channel
    VARIANCE -- precisely what class-clustering destroys. The first version of this
    script did exactly that and drove rho(contig, energy) to -0.573 instead of ~0.
    Training itself uses `shuffle=True` over all 100k rows, so the statistics it
    sees are the full-file statistics and the fit must match them.

    Same failure family as the FVU-0.903 incident: `df.sample`, never `df.head`,
    for anything variance-normalised.
    """
    ds = ParquetImageDataset(parquet, image_column="image", transform=transform)
    n = min(n_images, len(ds))
    idx = np.random.default_rng(seed).choice(len(ds), size=n, replace=False)
    loader = DataLoader(Subset(ds, sorted(int(i) for i in idx)),
                        batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    out = []
    for imgs in loader:
        t = ex.patch_tokens(imgs)
        out.append(t.reshape(-1, t.shape[-1]).float())
    return torch.cat(out)


def fit_class_coverage(parquet, n_images, seed):
    """How many distinct classes the fit sample actually spans, printed and asserted.

    The bug this guards against is invisible in every downstream number: a std
    fitted on a handful of classes is simply the wrong constant, and dividing by
    it looks like a working transform.
    """
    import pyarrow.parquet as pq
    try:
        lab = np.asarray(pq.read_table(parquet, columns=["label"])["label"])
    except Exception as e:                      # no label column -> nothing to check
        print(f"[fit] class coverage unavailable ({e})", flush=True)
        return None
    idx = np.random.default_rng(seed).choice(len(lab), size=min(n_images, len(lab)),
                                             replace=False)
    n_cls, n_all = len(set(lab[idx].tolist())), len(set(lab.tolist()))
    print(f"[fit] class coverage: {n_cls} of {n_all} classes in the fit sample "
          f"(first-{n_images}-rows would have been "
          f"{len(set(lab[:min(n_images, len(lab))].tolist()))})", flush=True)
    assert n_cls > 0.5 * min(n_all, n_images), (
        f"fit sample spans only {n_cls}/{n_all} classes -- per-channel variance "
        f"fitted on a class-clustered sample is the wrong constant")
    return n_cls


@torch.no_grad()
def val_maps(ex, transform, rows, device):
    """[n_img, D, side, side] on the val images -- one extraction, reused by all variants."""
    toks = []
    for _, r in rows.iterrows():
        try:
            im = Image.open(io.BytesIO(img_bytes(r))).convert("RGB")
        except Exception:
            continue
        toks.append(ex.patch_tokens(transform(im).unsqueeze(0)).squeeze(0).float())
    side = int(round(len(toks[0]) ** 0.5))
    assert side * side == toks[0].shape[0]
    return torch.stack(toks, 0), side          # [n_img, n_tok, D]


def measure(name, tv, side, n_pcs, dev):
    """All four qualification numbers for one variant. tv: [n_img, n_tok, D]."""
    maps = tv.permute(0, 2, 1).reshape(tv.shape[0], tv.shape[2], side, side)
    ac, _ = autocorr_per_feature(maps.to(dev), lags=(1,))
    contig = ac[1]
    del maps
    torch.cuda.empty_cache()

    X = tv.reshape(-1, tv.shape[-1])
    energy = X.var(0).cpu().numpy()
    ok = np.isfinite(contig)

    rv, pv = _rho(contig[ok], energy[ok])
    qc = np.nanquantile(contig[ok], [1 / 3, 2 / 3])
    lo = float(energy[contig <= qc[0]].mean())
    hi = float(energy[contig >= qc[1]].mean())

    Xn = X.cpu().numpy()
    p = PCA(n_components=n_pcs, random_state=0).fit(Xn)
    rec = p.inverse_transform(p.transform(Xn))
    lowrank = 1 - ((Xn - rec).var(0) / (energy + 1e-12))
    rl, pl = _rho(contig[ok], lowrank[ok])
    del Xn, rec

    c = concentration(X.to(dev), None)
    q = np.nanquantile(contig[ok], [0.1, 0.9])
    row = dict(
        variant=name,
        rho_contig_energy=rv, p_energy=pv,
        rho_contig_lowrank=rl, p_lowrank=pl,
        var_ratio_high_over_low=hi / max(lo, 1e-12),
        var_least_contig_third=lo, var_most_contig_third=hi,
        pca50_evr=float(p.explained_variance_ratio_.sum()),
        contig_median=float(np.nanmedian(contig[ok])),
        contig_p90_minus_p10=float(q[1] - q[0]),
        contig_frac_below_0p1=float(np.mean(contig[ok] < 0.1)),
        n_scored=int(ok.sum()),
        **c,
    )
    torch.cuda.empty_cache()
    return row, contig


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", default=TRAIN_PARQUET)
    ap.add_argument("--val_parquet", default=VAL_PARQUET)
    ap.add_argument("--n_fit_images", type=int, default=1024,
                    help="images used to FIT the per-channel stats (training parquet)")
    ap.add_argument("--n_val_images", type=int, default=200,
                    help="shortcut_analysis used 200 with random_state=0 -- keep it")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--zca_eps", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    fp = gpu_fingerprint(dev)
    print(f"[gpu] {fp}", flush=True)

    ex = BlockAwareDINOFeatureExtractor("dinov2_vitb14", DINO_REPO, dev, dino_block=0, norm=True)
    T = build_transform("dinov2", 224)

    # ── 1. fit the constants on the TRAINING parquet ──────────────────────────
    print(f"\n[fit] {args.n_fit_images} images RANDOMLY SAMPLED from the TRAINING parquet "
          f"(the same file and transform training uses, seed {args.sample_seed})", flush=True)
    n_cls = fit_class_coverage(args.train_parquet, args.n_fit_images, args.sample_seed)
    Xtr = train_tokens(ex, T, args.train_parquet, args.n_fit_images, args.batch, dev,
                       seed=args.sample_seed)
    print(f"[fit] {Xtr.shape[0]} tokens x {Xtr.shape[1]} channels", flush=True)
    stats = fit_channel_stats(Xtr, zca_eps=args.zca_eps)
    npz = out / "whiten_stats.npz"
    np.savez(npz, **stats)
    sha = sha256(npz)
    print(f"[fit] saved {npz}  sha256 {sha[:16]}", flush=True)
    ev = stats["eigvals"]
    print(f"[fit] eigenvalue spectrum: max {ev[0]:.4g}  median {np.median(ev):.4g}  "
          f"min {ev[-1]:.4g}  condition {ev[0]/max(ev[-1],1e-30):.3g}", flush=True)
    print(f"[fit] channel std: max {stats['std'].max():.4g}  median "
          f"{np.median(stats['std']):.4g}  min {stats['std'].min():.4g}", flush=True)

    # ── 2. input_scale per variant, measured on the TRAINING parquet ──────────
    # The reference is the `none` median measured HERE, on the same class-diverse
    # sample, not the hardcoded 73.4468: that constant came from token_scale_calib,
    # which read the first 512 rows = 2 classes. Existing block-0 post-LN ckpts
    # trained at input_scale=1.0 on the shuffled full file, so the full-file `none`
    # median IS what they saw. The old constant is printed alongside, not silently
    # dropped -- the no-norm / SigLIP2 ckpts are already baked at scales derived
    # from it and nothing here retroactively changes them.
    scales = {}
    med_none = None
    for mode in ("none", "diag", "zca"):
        tr = ChannelTransform(mode, None if mode == "none" else npz, device=dev).apply(Xtr)
        med = float(tr.norm(dim=-1).median())
        if mode == "none":
            med_none = med
            print(f"[scale] reference = `none` median {med:.4f} on {args.n_fit_images} "
                  f"class-diverse images  (token_scale_calib recorded "
                  f"{REF_TOKEN_NORM:.4f} from 2 classes; "
                  f"{100*(med/REF_TOKEN_NORM-1):+.1f}%)", flush=True)
        scales[mode] = dict(token_norm_median=med,
                            input_scale_to_ref=med_none / med,
                            input_scale_to_old_ref=REF_TOKEN_NORM / med)
        print(f"[scale] {mode:5s} token L2 median {med:10.4f}  -> input_scale "
              f"{med_none/med:.6f}", flush=True)
        del tr
    del Xtr
    torch.cuda.empty_cache()

    # ── 3. qualify each variant on the SAME 200 val images as shortcut_analysis ─
    rows = pd.read_parquet(args.val_parquet).sample(args.n_val_images,
                                                    random_state=args.sample_seed)
    tv, side = val_maps(ex, T, rows, dev)
    print(f"\n[val] {tv.shape[0]} images x {tv.shape[1]} tokens ({side}x{side} grid) "
          f"x {tv.shape[2]} channels", flush=True)

    recs, contigs = [], {}
    for mode in ("none", "diag", "zca"):
        ct = ChannelTransform(mode, None if mode == "none" else npz, device=dev)
        r, contig = measure(mode, ct.apply(tv), side, args.n_pcs, dev)
        r.update(scales[mode])
        r["channel_stats_sha256"] = "" if mode == "none" else sha
        recs.append(r)
        contigs[mode] = contig
        print(f"\n--- {mode} ---", flush=True)
        print(f"  rho(contig, energy)   {r['rho_contig_energy']:+.3f} (p={r['p_energy']:.2e})"
              f"   var ratio high/low {r['var_ratio_high_over_low']:.2f}x", flush=True)
        print(f"  rho(contig, lowrank)  {r['rho_contig_lowrank']:+.3f} (p={r['p_lowrank']:.2e})"
              f"   PCA-{args.n_pcs} EVR {r['pca50_evr']:.3f}", flush=True)
        print(f"  contiguity  median {r['contig_median']:+.3f}  p90-p10 "
              f"{r['contig_p90_minus_p10']:.3f}  frac<0.1 {r['contig_frac_below_0p1']:.3f}",
              flush=True)
        print(f"  trainability  top1%-token energy {r['top1pct_token_energy_share']:.3f}"
              f"   top10-chan var {r['top10_chan_var_share']:.3f}"
              f"   chanvar max/med {r['chan_var_max_over_median']:.1f}", flush=True)

    # ── 4. the assert that makes `diag` the minimal knob ──────────────────────
    a, b = contigs["none"], contigs["diag"]
    m = np.isfinite(a) & np.isfinite(b)
    dmax = float(np.abs(a[m] - b[m]).max())
    print(f"\n[assert] diag leaves every channel's contiguity unchanged "
          f"(the metric is scale-free): max abs diff {dmax:.3e} over {m.sum()} channels",
          flush=True)
    assert dmax < 1e-4, (
        f"per-channel rescaling moved contiguity by {dmax:.3e}; the smooth/rough axis "
        f"is NOT preserved and `diag` is no longer a one-variable manipulation")

    # ── 5. the control ────────────────────────────────────────────────────────
    n = recs[0]
    print(f"\n[CONTROL] untouched post-LN, this process, same 200 images, n_pcs={args.n_pcs}")
    print(f"  rho(contig, energy)  {n['rho_contig_energy']:+.3f}  vs recorded "
          f"{RECORDED['rho_energy']:+.3f}")
    print(f"  rho(contig, lowrank) {n['rho_contig_lowrank']:+.3f}  vs recorded "
          f"{RECORDED['rho_lowrank']:+.3f}")
    print(f"  var ratio            {n['var_ratio_high_over_low']:.2f}x  vs recorded "
          f"{RECORDED['var_ratio']:.1f}x")
    ok_ctrl = (abs(n["rho_contig_energy"] - RECORDED["rho_energy"]) < 0.05
               and abs(n["rho_contig_lowrank"] - RECORDED["rho_lowrank"]) < 0.05)
    print(f"  -> control {'PASS' if ok_ctrl else 'FAIL -- nothing else here is readable'}")

    df = pd.DataFrame(recs)
    df.to_csv(out / "whiten_calib.csv", index=False)
    np.savez(out / "whiten_calib_contig.npz", **contigs)
    (out / "control.json").write_text(json.dumps(
        dict(control_pass=bool(ok_ctrl), recorded=RECORDED,
             observed={k: n[k] for k in ("rho_contig_energy", "rho_contig_lowrank",
                                         "var_ratio_high_over_low")},
             diag_contig_max_abs_diff=dmax, channel_stats_sha256=sha,
             gpu=fp), indent=2, default=str))
    print("\n" + df[["variant", "rho_contig_energy", "rho_contig_lowrank",
                     "contig_median", "contig_p90_minus_p10", "contig_frac_below_0p1",
                     "top1pct_token_energy_share", "top10_chan_var_share",
                     "token_norm_median", "input_scale_to_ref"]].to_string(index=False))
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
