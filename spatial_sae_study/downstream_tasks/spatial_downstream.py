"""
Downstream tasks matched to the property the spatial SAE actually has.

What was confirmed (block 0, 3 seeds): the contrastive loss does NOT change which
features G0 holds (reconstruction contiguity ns) but DOES reorganise the code so
that G0's dominant directions vary coherently across space.

Every existing evaluation is blind to that:
  class accuracy / image-ID   pool the whole image, destroying spatial layout
  transformation invariance   global stability, not within-image organisation
  segment alignment           pairwise similarity; never requires carving the image
  FVU                         pure reconstruction

Two tasks that do require within-image spatial organisation:

  TASK 1  within-image unsupervised segmentation. Cluster an image's patches by
          their code and score the clusters against the per-token ground truth.
          An incoherent code gives speckled regions; a coherent one gives
          contiguous regions that follow object boundaries.

  TASK 2  label-efficient propagation. Reveal N labelled patches per image and
          classify the rest by nearest neighbour in code space. A coherent code
          means one label covers a region, so the advantage should be largest at
          N=1 and shrink as N grows -- a falsifiable shape, not just a direction.

CONTROL, applied by default: the spatial arm's G0 code is also more compressible
(64 PCs explain 95.3% vs baseline's 88.4%). Clustering could favour it for that
reason alone, with nothing to do with spatial coherence. Both arms are therefore
also projected onto the SAME number of principal components before clustering,
and both the raw and the PC-matched result are reported. If the advantage
disappears under matching, the finding is compressibility, not coherence.
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache"
os.environ["TORCH_HOME"] = _CACHE
os.environ["HF_HOME"] = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/hf_datasets_cache"
Path(_CACHE + "/hub/checkpoints").mkdir(parents=True, exist_ok=True)

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for p in [THIS.parent, THIS.parent.parent, THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from fvu_compare import BlockDINO, T, load_sae
from repro_utils import gpu_fingerprint, check_gpu_match


def token_labels(ex, image_size, side, purity=0.6, min_px=0):
    """Per-token class label from ADE20K, -1 where no class covers the token."""
    objs = [(j, o["name"]) for j, o in enumerate(ex["objects"])
            if o["parts"]["part_level"] == 0]
    if not objs:
        return None, None
    names = sorted({n for _, n in objs})
    idx = {n: i for i, n in enumerate(names)}
    ps = image_size // side
    cov = np.zeros((len(names), side * side), dtype=np.float32)
    for j, n in objs:
        if j >= len(ex["instances"]):
            continue
        a = np.array(ex["instances"][j].resize((image_size, image_size),
                                               resample=Image.Resampling.NEAREST)) == 255
        cov[idx[n]] += a.reshape(side, ps, side, ps).mean(axis=(1, 3)).reshape(-1)
    best = cov.argmax(0); bv = cov.max(0)
    lab = np.where(bv >= purity, best, -1)
    return lab, len(names)


def cluster_score(F, lab, n_clusters, seed=0):
    m = lab >= 0
    if m.sum() < 8 or len(np.unique(lab[m])) < 2:
        return None
    k = max(2, min(n_clusters, int(m.sum()) - 1))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(F)
    return adjusted_rand_score(lab[m], km.labels_[m])


def propagate(F, lab, n_per_class, rng):
    """Reveal n_per_class labelled patches PER CLASS, classify the rest by NN.

    The budget is per class, not per image. The earlier version handed out one
    seed per present class and only then topped up to a per-image total, so with
    an ADE20K median of 11 classes per image the totals 1, 2 and 5 all collapsed
    to the same "one per class" setting and N had no effect -- which silently
    destroyed the shape prediction it was meant to test.
    """
    m = np.where(lab >= 0)[0]
    classes = np.unique(lab[m])
    if len(m) < len(classes) * n_per_class + 5 or len(classes) < 2:
        return None
    seeds = []
    for c in classes:
        pool = m[lab[m] == c]
        take = min(n_per_class, max(1, len(pool) - 1))
        seeds += list(rng.choice(pool, size=take, replace=False))
    seeds = np.array(seeds)
    rest = np.setdiff1d(m, seeds)
    if len(rest) == 0:
        return None
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    sim = Fn[rest] @ Fn[seeds].T
    pred = lab[seeds][sim.argmax(1)]
    return float((pred == lab[rest]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL:PATH")
    ap.add_argument("--dino_block", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--n_pcs", type=int, default=32,
                    help="PC-matched control: both arms projected to this many components")
    ap.add_argument("--seeds_grid", nargs="+", type=int, default=[1, 2, 4, 8],
                    help="labelled patches PER CLASS (not per image)")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    fp = gpu_fingerprint(args.device)
    print(f"[GPU] {fp['gpu_name']} torch {fp['torch']}", flush=True)
    check_gpu_match([Path(c.split(":", 1)[1]).parent for c in args.ckpt])

    from datasets import load_dataset
    ds = load_dataset("1aurent/ADE20K", split="validation", cache_dir=os.environ["HF_HOME"])
    dino = BlockDINO(args.device, dino_block=args.dino_block)
    saes = {}
    for spec in args.ckpt:
        lab, path = spec.split(":", 1)
        saes[lab] = load_sae(path, args.device)
        print(f"[SAE] {lab} groups={saes[lab][1]}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    used = 0
    for i in range(len(ds)):
        if used >= args.max_images:
            break
        ex = ds[i]
        try:
            img = ex["image"].convert("RGB")
            tok = dino.patch_tokens(T(img).unsqueeze(0))
        except Exception:
            continue
        side = int(math.sqrt(tok.shape[0]))
        lab, n_obj = token_labels(ex, args.image_size, side, args.purity)
        if lab is None or (lab >= 0).sum() < 16 or len(np.unique(lab[lab >= 0])) < 2:
            continue
        for arm, (ae, gs) in saes.items():
            g0 = gs[0]
            with torch.no_grad():
                z = ae.encode(tok, use_threshold=False)[:, :g0].cpu().numpy()
            if z.std() < 1e-8:
                continue
            npc = min(args.n_pcs, z.shape[0] - 1, int((z.std(0) > 0).sum()))
            if npc < 2:
                continue
            Zp = PCA(n_components=npc, random_state=0).fit_transform(z)
            rec = dict(image=i, arm=arm, n_obj=n_obj, n_labelled=int((lab >= 0).sum()))
            a_raw = cluster_score(z, lab, n_obj)
            a_pc = cluster_score(Zp, lab, n_obj)
            rec["ari_raw"] = a_raw
            rec["ari_pcmatched"] = a_pc
            for N in args.seeds_grid:
                rec[f"prop_raw_N{N}"] = propagate(z, lab, N, np.random.default_rng(1000 + i))
                rec[f"prop_pc_N{N}"] = propagate(Zp, lab, N, np.random.default_rng(1000 + i))
            rows.append(rec)
        used += 1
        if used % 50 == 0:
            print(f"  {used}/{args.max_images}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out / "spatial_downstream_raw.csv", index=False)
    num = [c for c in d.columns if c.startswith(("ari_", "prop_"))]
    print(f"\n[Data] {used} images, {len(d)} rows\n")
    print(d.groupby("arm")[num].mean().round(4).to_string())
    d.groupby("arm")[num].mean().to_csv(out / "spatial_downstream_summary.csv")
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
