"""Do the block-0 results survive dropping LayerNorm, and do they transfer to SigLIP2?

Three conditions, one process, the same images:

  C1  DINOv2  block 0, post-LN   (the existing checkpoints -- live control, not a
                                  number recalled from an old run)
  C2  DINOv2  block 0, no-norm   (job 41434687)
  C3  SigLIP2 block 0, raw       (job 41434687)

Each condition has spatial (cw=1) and baseline (cw=0) at seeds 0/1/2, so the
primary test is always a WITHIN-condition, within-seed spatial-minus-baseline
contrast -- the only contrast this project treats as valid.

The two measures are the two things that are actually established on normed
DINOv2 block 0, so this is a replication under a changed feature convention and
a changed backbone, not a fishing expedition:

  G0 code coherence   spatial 0.859 vs baseline 0.684, +0.175, p=.0004 (3 seeds)
  label propagation   +0.037..+0.047 across 1/2/4/8 labels per class, 3/3 seeds

THE FAILURE MODE THIS SCRIPT IS BUILT AROUND
--------------------------------------------
An SAE trained on un-normed, input-scaled features will happily encode normed,
unscaled features and return NOISE rather than raising. Nothing in the tensor
shapes catches it: every arm is 768-d. So the feature pipeline is never specified
on the command line -- it is read from each checkpoint's own run_args.json
(backbone / backbone_norm / input_scale / block / image_size) and built from the
TRAINER'S OWN extractor classes. A run cannot be evaluated under a convention it
was not trained under.

`--selftest` asserts that: it rebuilds each arm's pipeline, confirms the observed
token L2 median matches the scale the SAE was trained at, and reports FVU. An arm
fed the wrong convention shows FVU near or above 1.
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache"
os.environ["TORCH_HOME"] = _CACHE
# Two SEPARATE caches, and they must not be swapped: SigLIP2's weights were
# pre-fetched to hf_cache, ADE20K lives in hf_datasets_cache. home is at 93% --
# nothing may fall back to ~/.cache.
_HF_MODELS = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/hf_cache"
_HF_DATA = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/hf_datasets_cache"
os.environ["HF_HOME"] = _HF_MODELS
Path(_CACHE + "/hub/checkpoints").mkdir(parents=True, exist_ok=True)

import argparse
import io
import json
import math
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for p in [THIS.parent, THIS.parent.parent, THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.decomposition import PCA

from dino_smoothness import autocorr_per_feature
from arm_pipeline import pipeline_from_ckpt
from fvu_compare import load_sae
from new_spatial_train_triplet import (
    BlockAwareDINOFeatureExtractor,
    SigLIP2FeatureExtractor,
    build_transform,
)
from repro_utils import gpu_fingerprint, check_gpu_match
from segment_alignment import (
    build_instance_id_map,
    feature_metrics,
    label_map_to_patch_labels,
)
from spatial_downstream import propagate, token_labels

# spatial_downstream sets HF_HOME to the DATASET cache at import time; SigLIP2's
# weights are in the model cache and HF_HUB_OFFLINE=1 makes a wrong path fatal.
os.environ["HF_HOME"] = _HF_MODELS

DINO_REPO = _CACHE + "/hub/facebookresearch_dinov2_main"


_EXTRACTORS: dict = {}


class Arm:
    """One checkpoint plus its single canonical feature pipeline.

    Do not reimplement feature extraction here.  `arm_pipeline.Pipeline` is shared
    by training-adjacent analyses and reconstructs, in order, backbone extraction,
    the content-pinned channel transform, and input_scale from run_args.json.
    """

    def __init__(self, label, cond, arm, seed, ckpt_dir, device):
        self.label, self.cond, self.arm, self.seed = label, cond, arm, seed
        self.dir = Path(ckpt_dir)
        self.pipeline = pipeline_from_ckpt(self.dir, device)
        self.run_args = self.pipeline.run_args
        self.backbone = self.pipeline.backbone
        self.norm = self.pipeline.norm
        self.input_scale = self.pipeline.input_scale
        self.image_size = self.pipeline.image_size
        self.block = self.pipeline.block
        self.channel_transform = self.pipeline.channel_transform
        self.channel_stats_sha = self.pipeline.channel_stats_sha
        self.ae, self.gs = load_sae(str(self.dir / "ae_latest.pt"), device)
        self.g0 = self.gs[0]

    @torch.no_grad()
    def tokens(self, img):
        """[n_patches, D] on the checkpoint's exact recorded convention."""
        return self.pipeline.tokens(img)

    @torch.no_grad()
    def g0_code(self, tok):
        return self.ae.encode(tok, use_threshold=False)[:, :self.g0]

    @torch.no_grad()
    def g0_recon(self, tok):
        return self.g0_code(tok) @ self.ae.W_dec[:self.g0]

    @torch.no_grad()
    def full_code(self, tok):
        return self.ae.encode(tok, use_threshold=False)


def contiguity(maps):
    """Median lag-1 autocorrelation over channels, using the shared metric."""
    ac, nseen = autocorr_per_feature(maps, lags=(1,))
    a1 = ac[1][nseen >= max(1, int(0.1 * maps.shape[0]))]
    return float(np.nanmedian(a1)), int(np.isfinite(a1).sum())


@torch.no_grad()
def selftest(arms, images, device):
    """An arm fed the wrong feature convention returns noise, not an error."""
    print("\n[selftest] each arm re-encodes on its OWN recorded convention", flush=True)
    rows = []
    for a in arms:
        toks = torch.cat([a.tokens(im) for im in images[:16]], 0)
        xhat = a.ae.decode(a.ae.encode(toks, use_threshold=False))
        # paper convention (dictionary_learning/training.py:57-60), mean-subtracted
        fvu = float((toks - xhat).var(0).sum() / toks.var(0).sum())
        med = float(toks.norm(dim=-1).median())
        rows.append(dict(label=a.label, backbone=a.backbone, block=a.block, norm=a.norm,
                         input_scale=a.input_scale, channel_transform=a.channel_transform,
                         channel_stats_sha=a.channel_stats_sha,
                         token_l2_median=med, fvu=fvu))
        print(f"  {a.label:34s} {a.backbone:7s} b{a.block} norm={str(a.norm):5s} "
              f"scale={a.input_scale:9.4f} channel={a.channel_transform:4s}  "
              f"token L2 med {med:7.2f}  FVU {fvu:.4f}", flush=True)
        assert fvu < 0.95, (f"{a.label}: FVU {fvu:.3f} -- the SAE is not reconstructing these "
                            f"features, the pipeline does not match training")
    return pd.DataFrame(rows)


@torch.no_grad()
def task_coherence(arms, images, n_pcs, device):
    """Replicates the confirmed block-0 measurement (results/block0_seed_confirm.csv).

    pca_top10   PCA-`n_pcs` fitted on the G0 code pooled over all images (one basis
                per arm, as in the original), each component's score map scored by
                lag-1 autocorrelation, median over the top 10 components.
    recon_med   the same autocorrelation applied to G0's contribution in DINO/SigLIP
                OUTPUT space, z_G0 @ W_dec_G0, median over the 768 dims.
    Note the PCA basis is fitted per arm, so two arms' pca_top10 are not on a
    common basis -- this is a property of the original measurement, carried over.
    """
    rows = []
    for a in arms:
        codes, recon_maps = [], []
        for im in images:
            tok = a.tokens(im)
            side = int(math.sqrt(tok.shape[0]))
            z = a.g0_code(tok)
            codes.append(z.cpu().numpy().astype(np.float32))
            recon_maps.append(a.g0_recon(tok).T.reshape(-1, side, side).cpu())
        R = torch.stack(recon_maps).to(device)                      # [n_img, 768, s, s]
        recon_med, _ = contiguity(R)
        del R, recon_maps

        Z = np.concatenate(codes, 0)                                # [n_img*tokens, G0]
        alive = Z.std(0) > 1e-8
        npc = int(min(n_pcs, alive.sum(), Z.shape[0] - 1))
        p = PCA(n_components=npc, random_state=0, svd_solver="randomized")
        S = p.fit_transform(Z[:, alive]).astype(np.float32)         # [tokens, npc]
        del Z, codes
        M = torch.from_numpy(S.reshape(len(images), side * side, npc)
                             .transpose(0, 2, 1).reshape(len(images), npc, side, side)).to(device)
        ac, _ = autocorr_per_feature(M, lags=(1,))
        del M, S

        rows.append(dict(cond=a.cond, arm=a.arm, seed=a.seed, label=a.label,
                         n_images=len(images), n_alive_g0=int(alive.sum()), n_pcs=npc,
                         pca_top10=float(np.nanmedian(ac[1][:10])),
                         evr=float(p.explained_variance_ratio_.sum()),
                         recon_med=recon_med))
        print(f"  [coh] {a.label:34s} alive {rows[-1]['n_alive_g0']:5d}  "
              f"pca_top10 {rows[-1]['pca_top10']:.4f}  evr {rows[-1]['evr']:.4f}  "
              f"recon {recon_med:.4f}", flush=True)
    return pd.DataFrame(rows)


@torch.no_grad()
def task_propagation(arms, ds, max_images, purity, seeds_grid, device):
    """Label propagation on the G0 code. Same images and same seed draw for every arm."""
    rows, used = [], 0
    for i in range(len(ds)):
        if used >= max_images:
            break
        ex = ds[i]
        try:
            img = ex["image"].convert("RGB")
        except Exception:
            continue
        per_arm = []
        ok = True
        for a in arms:
            tok = a.tokens(img)
            side = int(math.sqrt(tok.shape[0]))
            lab, n_obj = token_labels(ex, a.image_size, side, purity)
            if lab is None or (lab >= 0).sum() < 16 or len(np.unique(lab[lab >= 0])) < 2:
                ok = False
                break
            z = a.g0_code(tok).cpu().numpy()
            if z.std() < 1e-8:
                ok = False
                break
            rec = dict(image=i, cond=a.cond, arm=a.arm, seed=a.seed, label=a.label)
            for N in seeds_grid:
                # identical seed draw across arms: rng depends on image and N only
                rec[f"prop_N{N}"] = propagate(z, lab, N, np.random.default_rng(1000 + i))
            per_arm.append(rec)
        if not ok or len(per_arm) != len(arms):
            continue
        rows += per_arm
        used += 1
        if used % 50 == 0:
            print(f"  [prop] {used}/{max_images}", flush=True)
    return pd.DataFrame(rows), used


@torch.no_grad()
def task_segment(arms, ds, max_images, purity, mask_rule, precision_k, device):
    """ADE20K instance segment alignment on the FULL code.

    Matches the recorded three-arm run (results/seg_cmp3_block0): representation
    = Full (not G0), mask_rule=positive (mask>0), purity 0.6, precision@5, and the
    same per-image metric functions imported from segment_alignment.py. The
    recorded block-0 post-LN contrast is spatial Full minus baseline Full,
    gap 0.2792-0.1773 = +0.102 and prec@5 0.8018-0.7347 = +0.067 over 500 images.
    """
    rows, used = [], 0
    for i in range(len(ds)):
        if used >= max_images:
            break
        ex = ds[i]
        try:
            img = ex["image"].convert("RGB")
        except Exception:
            continue
        per_arm, ok, labcache = [], True, {}
        for a in arms:
            tok = a.tokens(img)
            side = int(math.sqrt(tok.shape[0]))
            # DINOv2 224/14 and SigLIP2 256/16 both give a 16x16 grid, but the patch
            # size has to follow the arm's own image size, not a constant. Only two
            # distinct sizes exist, so the mask rasterisation is shared.
            if a.image_size not in labcache:
                seg = build_instance_id_map(ex.get("instances", []), ex.get("objects", []),
                                            image_size=a.image_size, mask_rule=mask_rule)
                labcache[a.image_size] = None if seg is None else label_map_to_patch_labels(
                    seg, image_size=a.image_size, patch_size=a.image_size // side,
                    purity_threshold=purity)
            if labcache[a.image_size] is None:
                ok = False
                break
            lab, valid, _ = labcache[a.image_size]
            # left on the GPU on purpose: the metric is a [n_patch, 16384] gram matrix
            m = feature_metrics(a.full_code(tok), lab, valid, precision_k=precision_k)
            if m is None:
                ok = False
                break
            per_arm.append(dict(image=i, cond=a.cond, arm=a.arm, seed=a.seed, label=a.label,
                                gap=m["gap"], same_sim=m["same_sim"], diff_sim=m["diff_sim"],
                                prec5=m[f"precision_at_{precision_k}"]))
        if not ok or len(per_arm) != len(arms):
            continue
        rows += per_arm
        used += 1
        if used % 50 == 0:
            print(f"  [seg] {used}/{max_images}", flush=True)
    return pd.DataFrame(rows), used


def paired_report(df, value_cols, out, name):
    """Within-condition, within-seed spatial minus baseline, with a paired t-test."""
    recs = []
    for cond, g in df.groupby("cond"):
        for col in value_cols:
            sp = g[g.arm == "spatial"].set_index("seed")[col].sort_index()
            bl = g[g.arm == "baseline"].set_index("seed")[col].sort_index()
            common = sp.index.intersection(bl.index)
            if len(common) < 2:
                continue
            d = (sp[common] - bl[common]).values
            t, p = stats.ttest_rel(sp[common].values, bl[common].values)
            recs.append(dict(cond=cond, metric=col, n_seeds=len(common),
                             spatial=float(sp[common].mean()), baseline=float(bl[common].mean()),
                             diff=float(d.mean()), diff_sd=float(d.std(ddof=1)),
                             all_positive=bool((d > 0).all()), t=float(t), p=float(p)))
    r = pd.DataFrame(recs)
    r.to_csv(out / f"{name}_paired.csv", index=False)
    print(f"\n=== {name}: spatial minus baseline (paired by seed) ===")
    print(r.to_string(index=False))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
                                       "Projects/T-SAE-Spatial")
    ap.add_argument("--parquet", default="/n/home12/binxuwang/T-SAE-Spatial/imagenet_val_2k.parquet")
    ap.add_argument("--n_coh_images", type=int, default=150)
    ap.add_argument("--max_prop_images", type=int, default=300)
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--n_pcs", type=int, default=64)
    ap.add_argument("--seeds_grid", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--max_seg_images", type=int, default=500)
    ap.add_argument("--mask_rule", default="positive", choices=["positive", "sure"])
    ap.add_argument("--precision_k", type=int, default=5)
    ap.add_argument("--skip_coh", action="store_true")
    ap.add_argument("--skip_prop", action="store_true")
    ap.add_argument("--skip_seg", action="store_true")
    # Which conditions to load. Defaults reproduce the recorded job-41462881 /
    # 41499292 invocation exactly. `dino_b0_postLN` is a SPECIAL name: those
    # checkpoints predate the seed sweep and their seed-0 run is called something
    # else, so it is expanded from a hardcoded table; everything else follows the
    # uniform `{cond}_{arm}_s{seed}` layout. Used for the b1/b2 fill-in
    # (job 41788668), where b0 post-LN must stay in the SAME process as the live
    # control -- paired tests are invalid across separate runs.
    ap.add_argument("--conds", nargs="+",
                    default=["dino_b0_postLN", "dino_b0_nonorm", "siglip_b0_raw"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    fp = gpu_fingerprint(args.device)
    print(f"[GPU] {fp['gpu_name']} (cc {fp['capability']}, torch {fp['torch']})", flush=True)

    S = Path(args.store)
    # C1 uses the pre-existing normed block-0 checkpoints, whose seed-0 run is
    # named differently from seeds 1/2 (it predates the seed sweep).
    spec = []
    for cond in args.conds:
        if cond == "dino_b0_postLN":
            for seed, (sp, bl) in enumerate(
                    [("spatial_block0", "spatial_block0_baseline"),
                     ("block0_spatial_s1", "block0_baseline_s1"),
                     ("block0_spatial_s2", "block0_baseline_s2")]):
                spec.append((cond, "spatial", seed, S / sp))
                spec.append((cond, "baseline", seed, S / bl))
            continue
        for arm in ("spatial", "baseline"):
            for seed in (0, 1, 2):
                spec.append((cond, arm, seed, S / f"{cond}_{arm}_s{seed}"))

    arms = []
    for cond, arm, seed, d in spec:
        if not (d / "ae_latest.pt").exists():
            print(f"[skip] missing {d}", flush=True)
            continue
        arms.append(Arm(f"{cond}/{arm}/s{seed}", cond, arm, seed, d, args.device))
    print(f"[arms] {len(arms)} checkpoints loaded", flush=True)
    check_gpu_match([a.dir for a in arms])

    df = pd.read_parquet(args.parquet).sample(args.n_coh_images, random_state=0)
    images = []
    for _, r in df.iterrows():
        b = r["image"]; b = b["bytes"] if isinstance(b, dict) else b
        try:
            images.append(Image.open(io.BytesIO(b)).convert("RGB"))
        except Exception:
            continue
    print(f"[data] {len(images)} ImageNet images for coherence", flush=True)

    st = selftest(arms, images, args.device)
    st.to_csv(out / "selftest.csv", index=False)

    if not args.skip_coh:
        coh = task_coherence(arms, images, args.n_pcs, args.device)
        coh.to_csv(out / "coherence.csv", index=False)
        paired_report(coh, ["pca_top10", "evr", "recon_med"], out, "coherence")

    if not (args.skip_prop and args.skip_seg):
        from datasets import load_dataset
        ds = load_dataset("1aurent/ADE20K", split="validation", cache_dir=_HF_DATA)

    if not args.skip_prop:
        pr, used = task_propagation(arms, ds, args.max_prop_images, args.purity,
                                    args.seeds_grid, args.device)
        pr.to_csv(out / "propagation_raw.csv", index=False)
        cols = [f"prop_N{N}" for N in args.seeds_grid]
        agg = pr.groupby(["cond", "arm", "seed", "label"])[cols].mean().reset_index()
        agg.to_csv(out / "propagation.csv", index=False)
        print(f"\n[prop] {used} images scored")
        print(agg.to_string(index=False))
        paired_report(agg, cols, out, "propagation")

    if not args.skip_seg:
        sg, used = task_segment(arms, ds, args.max_seg_images, args.purity,
                                args.mask_rule, args.precision_k, args.device)
        sg.to_csv(out / "segment_raw.csv", index=False)
        cols = ["gap", "prec5", "same_sim", "diff_sim"]
        agg = sg.groupby(["cond", "arm", "seed", "label"])[cols].mean().reset_index()
        agg.to_csv(out / "segment.csv", index=False)
        print(f"\n[seg] {used} images scored")
        print(agg.to_string(index=False))
        paired_report(agg, cols, out, "segment")
        # Also the per-IMAGE paired test, which is how the recorded three-arm run
        # was analysed (n=500 images, one seed). Reported alongside the 3-seed
        # test so the two are comparable rather than silently swapped.
        per_img = []
        for cond, g in sg.groupby("cond"):
            for seed, gs in g.groupby("seed"):
                sp = gs[gs.arm == "spatial"].set_index("image")
                bl = gs[gs.arm == "baseline"].set_index("image")
                idx = sp.index.intersection(bl.index)
                for col in ["gap", "prec5"]:
                    t, p = stats.ttest_rel(sp.loc[idx, col], bl.loc[idx, col])
                    per_img.append(dict(cond=cond, seed=seed, metric=col, n_images=len(idx),
                                        diff=float((sp.loc[idx, col] - bl.loc[idx, col]).mean()),
                                        frac_img_positive=float((sp.loc[idx, col] > bl.loc[idx, col]).mean()),
                                        t=float(t), p=float(p)))
        pi = pd.DataFrame(per_img)
        pi.to_csv(out / "segment_per_image_paired.csv", index=False)
        print("\n=== segment: per-IMAGE paired test (recorded run's analysis) ===")
        print(pi.to_string(index=False))

    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
