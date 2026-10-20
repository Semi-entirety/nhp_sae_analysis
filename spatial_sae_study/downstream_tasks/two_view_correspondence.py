"""
TASK A -- two-view patch correspondence.

Why this task. Every evaluation that pools an image (class accuracy, image-ID)
is blind to the one property the spatial SAE is confirmed to have: within-image
code organisation. Correspondence is the opposite -- it is nothing but
within-image localisation, and its readout is rank-based (argmax over candidate
patches), which is the regime where the spatial arm has so far won.

Ground truth is free. Two random crops of the same image are taken with known
boxes, so the true match of any patch in view A is computed from the crop
geometry -- no annotation, no labels, no dataset beyond raw ImageNet images.

PRE-REGISTERED PREDICTION (before running). The contrastive loss pulls together
patches within radius 2 during training. That is exactly the neighbourhood a
correspondence task must resolve. So:

    spatial should be WORSE than baseline at strict localisation
    (exact patch, and error <= 1 patch), and its error distribution should show
    excess mass specifically at distances 1-2 patches -- the training radius.

    If instead spatial is better at strict localisation, the "smoothing blurs
    local detail" account is wrong and I will report it as refuted.

This is deliberately a task where the spatial arm can lose. It was chosen for
that reason: the previous two downstream tasks were designed after knowing the
property they tested, which limits what a win could support.

Similarity is computed on the raw G0 code (no basis-matching problem) and, for
consistency with the other analyses, on a 32-d PCA fitted JOINTLY on both views
-- fitting per view would put the two views in different bases and make the
similarity meaningless.
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
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for p in [THIS.parent, THIS.parent.parent, THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.decomposition import PCA
from scipy.stats import ttest_rel
from fvu_compare import BlockDINO, load_sae
from repro_utils import gpu_fingerprint, check_gpu_match

NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def two_views(img, canon, out_size, rng, lo=0.5, hi=0.9):
    """Two random square crops of the canonical image, with their boxes."""
    im = img.convert("RGB").resize((canon, canon), Image.Resampling.BILINEAR)
    boxes = []
    views = []
    for _ in range(2):
        s = rng.uniform(lo, hi) * canon
        x0 = rng.uniform(0, canon - s)
        y0 = rng.uniform(0, canon - s)
        boxes.append((x0, y0, s))
        views.append(im.crop((x0, y0, x0 + s, y0 + s))
                       .resize((out_size, out_size), Image.Resampling.BILINEAR))
    return views, boxes


def correspondence(boxes, side, out_size):
    """For each patch of view A, the true patch index in view B (-1 if outside).

    Patch (r, c) centre in view-A pixels -> canonical coords via box A ->
    view-B pixels via box B -> grid cell. A patch whose centre falls outside
    view B has no correspondence and is excluded.
    """
    ps = out_size / side
    r, c = np.divmod(np.arange(side * side), side)
    xa = (c + 0.5) * ps
    ya = (r + 0.5) * ps
    (x0a, y0a, sa), (x0b, y0b, sb) = boxes
    xc = x0a + xa / out_size * sa
    yc = y0a + ya / out_size * sa
    xb = (xc - x0b) / sb * out_size
    yb = (yc - y0b) / sb * out_size
    cb = np.floor(xb / ps).astype(int)
    rb = np.floor(yb / ps).astype(int)
    ok = (cb >= 0) & (cb < side) & (rb >= 0) & (rb < side)
    gt = np.where(ok, rb * side + cb, -1)
    return gt


def score(FA, FB, gt, side):
    """argmax-similarity match, scored against the geometric ground truth."""
    m = np.where(gt >= 0)[0]
    if len(m) < 10:
        return None
    An = FA / (np.linalg.norm(FA, axis=1, keepdims=True) + 1e-9)
    Bn = FB / (np.linalg.norm(FB, axis=1, keepdims=True) + 1e-9)
    pred = (An[m] @ Bn.T).argmax(1)
    pr, pc = np.divmod(pred, side)
    gr, gc = np.divmod(gt[m], side)
    d = np.sqrt((pr - gr) ** 2 + (pc - gc) ** 2)
    return dict(exact=float((d == 0).mean()), d1=float((d <= 1).mean()),
                d2=float((d <= 2).mean()), d3=float((d <= 3).mean()),
                mean_err=float(d.mean()), median_err=float(np.median(d)),
                n=len(m), hist=np.bincount(np.minimum(d.round().astype(int), 8),
                                           minlength=9) / len(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL:PATH")
    ap.add_argument("--parquet", default="/n/home12/binxuwang/T-SAE-Spatial/data/imagenet_data/train-00000-of-00001-18bc3231d015f1e8.parquet")
    ap.add_argument("--dino_block", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--canon", type=int, default=448)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_pcs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    fp = gpu_fingerprint(args.device)
    print(f"[GPU] {fp['gpu_name']} (cc {fp['capability']}, torch {fp['torch']})", flush=True)
    gc = check_gpu_match([Path(c.split(":", 1)[1]).parent for c in args.ckpt])
    with open(out / "gpu_check.json", "w") as f:
        json.dump({"analysis": fp, "check": gc}, f, indent=2, default=str)

    df = pd.read_parquet(args.parquet)
    print(f"[Data] {len(df)} ImageNet rows from {Path(args.parquet).name}", flush=True)
    dino = BlockDINO(args.device, dino_block=args.dino_block)
    saes = {}
    for spec in args.ckpt:
        lab, path = spec.split(":", 1)
        saes[lab] = load_sae(path, args.device)
        print(f"[SAE] {lab} groups={saes[lab][1]}", flush=True)

    rows, used = [], 0
    for i in range(len(df)):
        if used >= args.max_images:
            break
        try:
            b = df.iloc[i]["image"]
            img = Image.open(BytesIO(b["bytes"] if isinstance(b, dict) else b))
            rng = np.random.default_rng(args.seed * 100003 + i)
            views, boxes = two_views(img, args.canon, args.image_size, rng)
            tok = [dino.patch_tokens(NORM(v).unsqueeze(0)) for v in views]
        except Exception:
            continue
        side = int(math.sqrt(tok[0].shape[0]))
        gt = correspondence(boxes, side, args.image_size)
        if (gt >= 0).sum() < 10:
            continue

        recs = {}
        ok = True
        for arm, (ae, gs) in saes.items():
            with torch.no_grad():
                z = [ae.encode(t, use_threshold=False)[:, :gs[0]].cpu().numpy() for t in tok]
            if z[0].std() < 1e-8 or z[1].std() < 1e-8:
                ok = False; break
            r_raw = score(z[0], z[1], gt, side)
            both = np.concatenate(z, 0)                       # joint basis, both views
            npc = min(args.n_pcs, both.shape[0] - 1, int((both.std(0) > 0).sum()))
            if r_raw is None or npc < 2:
                ok = False; break
            P = PCA(n_components=npc, random_state=0).fit(both)
            r_pc = score(P.transform(z[0]), P.transform(z[1]), gt, side)
            if r_pc is None:
                ok = False; break
            rec = dict(image=i, arm=arm, n_corr=r_raw["n"])
            for k, v in r_raw.items():
                if k == "hist":
                    for j, hv in enumerate(v):
                        rec[f"hist{j}"] = hv
                elif k != "n":
                    rec[f"raw_{k}"] = v
            for k, v in r_pc.items():
                if k not in ("hist", "n"):
                    rec[f"pc_{k}"] = v
            recs[arm] = rec
        if ok and len(recs) == len(saes):
            rows.extend(recs.values())
            used += 1
        if used and used % 50 == 0:
            print(f"  {used}/{args.max_images}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out / "correspondence_raw.csv", index=False)
    print(f"\n[Data] {used} image pairs, mean {d['n_corr'].mean():.0f} "
          f"corresponding patches per pair\n")
    print("PRE-REGISTERED: spatial WORSE at strict localisation (exact, <=1), "
          "with excess error mass at 1-2 patches\n")
    A = d[d.arm == "spatial"].set_index("image")
    B = d[d.arm == "baseline"].set_index("image")
    ix = A.index.intersection(B.index)
    cols = [c for c in d.columns if c.startswith(("raw_", "pc_", "hist"))]
    print(f"{'metric':<18}{'spatial':>9}{'baseline':>10}{'diff':>9}{'p':>11}")
    for c in cols:
        a, b = A.loc[ix, c].values, B.loc[ix, c].values
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 4:
            continue
        t, p = ttest_rel(a[m], b[m])
        print(f"{c:<18}{a[m].mean():>9.4f}{b[m].mean():>10.4f}"
              f"{a[m].mean() - b[m].mean():>+9.4f}{p:>11.2e}")
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
