"""
Fraction of Variance Unexplained for the three SAE arms, per DINO block.

    FVU = ||x - xhat||^2 / ||x - mean(x)||^2

measured on HELD-OUT images (the SAEs were trained on a different parquet).

A note on "raw data FVU"
-----------------------
Raw DINO tokens are the reconstruction TARGET, so raw-vs-raw FVU is 0 by
definition and is not an informative baseline. Two references that are
informative are reported instead:
  * PCA-256 reconstruction  -- a dense linear code, i.e. what you get without
    sparsity at a comparable (much smaller) parameter budget
  * mean predictor          -- FVU = 1 by construction, the trivial upper bound

For the two matryoshka arms the G0-only prefix reconstruction is also reported,
since the prefix is what the matryoshka objective actually optimises. Vanilla
has a single group, so its G0 column is empty by construction.
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache"
os.environ["TORCH_HOME"] = _CACHE
Path(_CACHE + "/hub/checkpoints").mkdir(parents=True, exist_ok=True)

import argparse
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

torch.hub.set_dir(_CACHE + "/hub")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))
from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE


class BlockDINO:
    def __init__(self, device, dino_block=None, model_name="dinov2_vitb14",
                 repo=_CACHE + "/hub/facebookresearch_dinov2_main"):
        self.device = device; self.dino_block = dino_block
        self.model = torch.hub.load(repo, model_name, source="local", trust_repo=True).to(device).eval()

    @torch.no_grad()
    def patch_tokens(self, t):
        x = t.to(self.device)
        if self.dino_block is None:
            return self.model.forward_features(x)["x_norm_patchtokens"].squeeze(0)
        return self.model.get_intermediate_layers(
            x, n=[self.dino_block], reshape=False, return_class_token=False, norm=True)[0].squeeze(0)


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


T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="HELD-OUT images (not the training parquet)")
    ap.add_argument("--store", required=True)
    ap.add_argument("--pca_dir", default=None)
    ap.add_argument("--blocks", nargs="+", type=int, default=[0, 4, 8, 11])
    ap.add_argument("--n_images", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd, joblib
    df = pd.read_parquet(args.parquet)
    rows_img = df.sample(min(args.n_images, len(df)), random_state=0)

    ARMS = [("spatial", ""), ("baseline", "_baseline"), ("vanilla", "_vanilla")]
    recs = []
    for blk in args.blocks:
        dino = BlockDINO(args.device, dino_block=blk)
        toks = []
        for _, r in rows_img.iterrows():
            d = r["image"]; b = d["bytes"] if isinstance(d, dict) else d
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                toks.append(dino.patch_tokens(T(img).unsqueeze(0)))
            except Exception:
                continue
        X = torch.cat(toks, 0)                                  # [N_tokens, D]
        denom = float((X - X.mean(0, keepdim=True)).pow(2).sum())
        print(f"[block {blk}] {X.shape[0]} held-out tokens", flush=True)

        for arm, suf in ARMS:
            ck = Path(args.store) / f"spatial_block{blk}{suf}" / "ae_latest.pt"
            if not ck.exists():
                print(f"  {arm}: MISSING {ck}"); continue
            ae, gs = load_sae(str(ck), args.device)
            with torch.no_grad():
                z = ae.encode(X, use_threshold=False)
                fvu = float((X - ae.decode(z)).pow(2).sum()) / denom
                if len(gs) > 1:                                  # matryoshka prefix
                    g0 = gs[0]
                    rec0 = z[:, :g0] @ ae.W_dec[:g0] + ae.b_dec
                    fvu_g0 = float((X - rec0).pow(2).sum()) / denom
                    g0act = float((z[:, :g0] > 0).float().sum(1).mean())
                else:
                    fvu_g0 = float("nan"); g0act = float("nan")
                act = float((z > 0).float().sum(1).mean())
                alive = int((z > 0).any(0).sum())
            recs.append(dict(block=blk, arm=arm, fvu=fvu, fvu_G0_prefix=fvu_g0,
                             active_per_token=act, G0_active_per_token=g0act,
                             alive_latents=alive, dict_size=z.shape[1]))
            print(f"  {arm:9s} FVU={fvu:.4f}  FVU(G0 prefix)={fvu_g0:.4f}  alive={alive}", flush=True)

        if args.pca_dir:
            p = Path(args.pca_dir) / f"pca_block{blk}" / f"pca_b{blk}_pca_model.pkl"
            if p.exists():
                pca = joblib.load(p)
                Xn = X.cpu().numpy()
                rec = pca.inverse_transform(pca.transform(Xn))
                fvu = float(((Xn - rec) ** 2).sum()) / denom
                recs.append(dict(block=blk, arm=f"PCA-{pca.n_components_}", fvu=fvu,
                                 fvu_G0_prefix=float("nan"),
                                 active_per_token=float(pca.n_components_),
                                 G0_active_per_token=float("nan"),
                                 alive_latents=int(pca.n_components_),
                                 dict_size=int(pca.n_components_)))
                print(f"  PCA-{pca.n_components_}   FVU={fvu:.4f}", flush=True)
        recs.append(dict(block=blk, arm="mean predictor", fvu=1.0, fvu_G0_prefix=float("nan"),
                         active_per_token=0.0, G0_active_per_token=float("nan"),
                         alive_latents=0, dict_size=0))
        del dino, X
        torch.cuda.empty_cache()

    d = pd.DataFrame(recs); d.to_csv(outdir / "fvu_compare.csv", index=False)
    print("\n" + d.to_string(index=False))

    piv = d[d["arm"].isin(["spatial", "baseline", "vanilla"])].pivot(
        index="block", columns="arm", values="fvu")
    pivg = d[d["arm"].isin(["spatial", "baseline"])].pivot(
        index="block", columns="arm", values="fvu_G0_prefix")
    pca_row = d[d["arm"].str.startswith("PCA")].set_index("block")["fvu"] if (d["arm"].str.startswith("PCA")).any() else None

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    x = np.arange(len(piv.index)); w = 0.26
    for i, arm in enumerate(["vanilla", "baseline", "spatial"]):
        if arm in piv:
            ax.bar(x + (i - 1) * w, piv[arm], w, label=arm,
                   color={"vanilla": "#9467bd", "baseline": "#d62728", "spatial": "#1f77b4"}[arm],
                   edgecolor="k", lw=.5)
    if pca_row is not None:
        ax.plot(x, pca_row.reindex(piv.index).values, "gd--", ms=9, label="PCA-256")
    ax.set_xticks(x); ax.set_xticklabels([f"block {b}" for b in piv.index])
    ax.set_ylabel("FVU  (lower = better reconstruction)")
    ax.set_title("Full reconstruction on held-out images\n(raw is the target, so its FVU is 0 by definition)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    for i, arm in enumerate(["baseline", "spatial"]):
        if arm in pivg:
            ax.bar(x + (i - 0.5) * w * 1.4, pivg[arm], w * 1.4, label=f"{arm} G0 prefix",
                   color={"baseline": "#d62728", "spatial": "#1f77b4"}[arm], alpha=.75,
                   edgecolor="k", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels([f"block {b}" for b in piv.index])
    ax.set_ylabel("FVU of the G0-only prefix")
    ax.set_title("Matryoshka prefix: G0 alone\n(vanilla has one group, so no prefix)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.suptitle(f"FVU across the three arms, {args.n_images} held-out images", fontsize=12)
    fig.tight_layout(); fig.savefig(outdir / "fvu_compare.png", dpi=150, bbox_inches="tight")
    print(f"[Saved] {outdir}")


if __name__ == "__main__":
    main()
