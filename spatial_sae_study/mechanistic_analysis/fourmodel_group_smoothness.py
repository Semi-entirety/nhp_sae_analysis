"""Unified G0/G1 smoothness audit for the four block-8/11 SAE arms.

Runs all arms on the same held-out images and DINO pipeline.  Reports both
latent-code lag-1 autocorrelation and decoded group-output autocorrelation.
The latent comparison includes scorable coverage and a G0-targeted nearest
neighbour match on token activation frequency to expose sparsity confounding.
"""
from __future__ import annotations

import argparse, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial import cKDTree

PROJ = Path("/n/home12/binxuwang/T-SAE-Spatial")
for p in [PROJ / "scripts", PROJ,
          PROJ / "temporal-saes" / "dictionary_learning"]:
    sys.path.insert(0, str(p))

from arm_pipeline import pipeline_from_ckpt, assert_same_pipeline
from dino_smoothness import autocorr_per_feature
from fvu_compare import load_sae
from repro_utils import gpu_fingerprint, check_gpu_match

ROOT = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/T-SAE-Spatial")


def image_bytes(row):
    x = row["image"]
    return x["bytes"] if isinstance(x, dict) else x


def summarize_code(block, arm, group, ac, nseen, rate, min_images):
    keep = (nseen >= min_images) & np.isfinite(ac)
    x = ac[keep]
    return dict(block=block, arm=arm, group=group, n_features=len(ac),
                n_scored=int(keep.sum()), frac_scored=float(keep.mean()),
                lag1_mean=float(np.mean(x)) if len(x) else np.nan,
                lag1_median=float(np.median(x)) if len(x) else np.nan,
                lag1_p10=float(np.quantile(x, .1)) if len(x) else np.nan,
                lag1_p90=float(np.quantile(x, .9)) if len(x) else np.nan,
                active_rate_mean=float(np.mean(rate)),
                active_rate_median=float(np.median(rate)))


def frequency_match(ac0, n0, r0, ac1, n1, r1, min_images):
    """Match every scorable G0 latent to nearest G1 latent in log activation rate."""
    k0 = (n0 >= min_images) & np.isfinite(ac0) & (r0 > 0)
    k1 = (n1 >= min_images) & np.isfinite(ac1) & (r1 > 0)
    if not k0.any() or not k1.any():
        return dict(n_g0=int(k0.sum()), n_g1=int(k1.sum()), matched_n=0,
                    matched_median_diff=np.nan, matched_mean_diff=np.nan,
                    median_abs_lograte_gap=np.nan)
    lr0 = np.log10(r0[k0]).reshape(-1, 1)
    lr1 = np.log10(r1[k1]).reshape(-1, 1)
    dist, idx = cKDTree(lr1).query(lr0, k=1)
    dif = ac0[k0] - ac1[k1][idx]
    return dict(n_g0=int(k0.sum()), n_g1=int(k1.sum()), matched_n=len(dif),
                matched_median_diff=float(np.median(dif)),
                matched_mean_diff=float(np.mean(dif)),
                median_abs_lograte_gap=float(np.median(dist)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, choices=[8, 11], required=True)
    ap.add_argument("--parquet", default=str(PROJ / "imagenet_val_2k.parquet"))
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--min-active-images", type=int, default=10)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    names = {
        "spatial": f"spatial_block{a.block}",
        "baseline": f"spatial_block{a.block}_baseline",
        "vanilla": f"spatial_block{a.block}_vanilla",
        "prefix": f"pfx_b{a.block}_prefix_s0",
    }
    dirs = {k: ROOT / v for k, v in names.items()}
    pipes = {k: pipeline_from_ckpt(v, a.device) for k, v in dirs.items()}
    pipe = assert_same_pipeline(*pipes.values())
    aes = {k: load_sae(str(v / "ae_latest.pt"), a.device)[0] for k, v in dirs.items()}
    width = int(.2 * 16384)
    for arm, d in dirs.items():
        cfg = json.load(open(d / "run_args.json"))
        assert cfg["dino_block"] == a.block
        if arm in ("spatial", "baseline"):
            assert cfg["group_fractions"] == [.2, .8]
        else:
            assert cfg["group_fractions"] == [1.0]
    print("[gpu]", gpu_fingerprint(a.device), flush=True)
    print("[pipeline]", pipe.describe(), "width", width, flush=True)

    df = pd.read_parquet(a.parquet).sample(a.n_images, random_state=0)
    toks = []
    for _, row in df.iterrows():
        try:
            toks.append(pipe.tokens(Image.open(io.BytesIO(image_bytes(row))).convert("RGB")))
        except Exception as e:
            print("[skip]", type(e).__name__, flush=True)
    assert len(toks) >= int(.9 * a.n_images)
    side = pipe.side(toks[0].shape[0])
    assert all(len(t) == side * side for t in toks)

    code_rows, recon_rows, match_rows, self_rows = [], [], [], []
    for arm, ae in aes.items():
        z_maps, y0_maps, y1_maps = [], [], []
        active = torch.zeros(16384, dtype=torch.float64)
        ntok = 0
        for t in toks:
            z = ae.encode(t, use_threshold=False)
            assert z.shape[1] == 16384
            active += (z > 0).sum(0).cpu().double(); ntok += len(z)
            z_maps.append(z.T.reshape(-1, side, side).cpu())
            y0 = z[:, :width] @ ae.W_dec[:width]
            y1 = z[:, width:] @ ae.W_dec[width:]
            y0_maps.append(y0.T.reshape(-1, side, side).cpu())
            y1_maps.append(y1.T.reshape(-1, side, side).cpu())
        rate = (active / ntok).numpy()
        Z = torch.stack(z_maps).to(a.device)
        ac0, n0 = autocorr_per_feature(Z[:, :width], lags=(1,))
        ac1, n1 = autocorr_per_feature(Z[:, width:], lags=(1,))
        ac0, ac1 = ac0[1], ac1[1]
        code_rows += [summarize_code(a.block, arm, "G0", ac0, n0, rate[:width], a.min_active_images),
                      summarize_code(a.block, arm, "G1", ac1, n1, rate[width:], a.min_active_images)]
        mr = frequency_match(ac0, n0, rate[:width], ac1, n1, rate[width:], a.min_active_images)
        mr.update(block=a.block, arm=arm); match_rows.append(mr)
        del Z, z_maps
        torch.cuda.empty_cache()

        for group, maps in [("G0", y0_maps), ("G1", y1_maps)]:
            M = torch.stack(maps).to(a.device)
            ac, n = autocorr_per_feature(M, lags=(1,)); x = ac[1][n > 0]
            recon_rows.append(dict(block=a.block, arm=arm, group=group,
                                   n_dims=M.shape[1], n_scored=int((n > 0).sum()),
                                   lag1_mean=float(np.mean(x)), lag1_median=float(np.median(x)),
                                   lag1_p10=float(np.quantile(x, .1)), lag1_p90=float(np.quantile(x, .9))))
            del M
        X = torch.cat(toks[:16]); xh = ae.decode(ae.encode(X, use_threshold=False))
        fvu = float((X-xh).var(0).sum()/X.var(0).sum())
        assert fvu < .95
        self_rows.append(dict(block=a.block, arm=arm, fvu=fvu,
                              token_l2=float(X.norm(dim=-1).median())))
        print("[arm done]", arm, code_rows[-2:], recon_rows[-2:], mr, flush=True)
        torch.cuda.empty_cache()

    pd.DataFrame(code_rows).to_csv(out / "code_smoothness.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(out / "decoded_smoothness.csv", index=False)
    pd.DataFrame(match_rows).to_csv(out / "frequency_matched.csv", index=False)
    pd.DataFrame(self_rows).to_csv(out / "selftest.csv", index=False)
    (out / "gpu_check.json").write_text(json.dumps(check_gpu_match(list(dirs.values())), indent=2, default=str))
    (out / "DONE").write_text("ok\n")
    print("FOURMODEL GROUP SMOOTHNESS DONE", out, flush=True)


if __name__ == "__main__":
    main()
