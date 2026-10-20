"""The two shortcuts that route a feature to G0 without any contrastive loss.

WHAT THIS REPLACES
------------------
On Aug 19 2026 three one-off /tmp scripts produced the "two shortcuts" result:
`raw_energy.py` -> results/raw_contig_vs_energy.csv, `energy_strat.py` ->
results/energy_stratified.csv, and `why_diff.py` (printed only, never saved).
None went into the repo and all three are gone from disk; their source was
recovered from the session transcript on Aug 24 2026. This is the repo version.

A LABEL THAT WAS WRONG AND IS FIXED HERE
----------------------------------------
`energy_stratified.csv`'s rho was reported -- in fig5's y-axis and in a message to
Wendy -- as "rho(contiguity, low-rank participation)". It is not. It is
contiguity vs the dimension's **G0 SHARE**, and that is why the CSV has an `arm`
column at all: low-rank participation is a property of raw DINO and cannot depend
on which SAE you look at. The two are computed and named separately here:

  energy          per-dim variance of the raw backbone features
  lowrank         1 - var(x - PCA_50(x)) / var(x), per dim: how well the top 50
                  PCs capture that dim. ARM-INDEPENDENT.
  g0_share        var(z_G0 @ W_dec_G0) / (var(G0 part) + var(G1 part)), per dim:
                  how much of that dim's reconstruction G0 supplies. PER ARM.

THE QUESTION
------------
On real DINO block 0, matryoshka ALONE sorts contiguous dims into G0 -- baseline
(contrastive_weight=0) does it as well as spatial, so there is nothing left for
the contrastive term to add on that axis. The account is that "contiguous" is
confounded with two things that make a dim cheap for a capacity-limited G0 to
reconstruct: it is high-energy, and it sits in the data's dominant low-rank
structure. The toy severed both by design (analytically matched energy/density,
random dictionary) and there baseline does not sort at all.

PRE-REGISTERED PREDICTION (written before this script was ever run)
-------------------------------------------------------------------
Baseline's G0 code coherence collapses when the borrowed block-11 LayerNorm is
dropped (post-LN .669 -> no-norm .300) and sits in between on SigLIP2 raw (.557),
while spatial barely moves (.846/.713/.845). If baseline's coherence is bought by
riding these two shortcuts, then shortcut strength on the RAW features should
follow the same ordering:

    rho(contiguity, energy)   and   rho(contiguity, lowrank)
    post-LN  >  SigLIP2 raw  >  DINOv2 no-norm

and likewise rho(contiguity, g0_share) should fall for baseline in the same order.
If the shortcuts are just as strong on raw block 0 as post-LN, this account is
DEAD and should be dropped, not patched. This is the 6th mechanism story in this
project and 4 of the previous 5 died -- treat a confirmation with suspicion and a
refutation as the expected outcome.

WHAT IS AND IS NOT A CONTROL
----------------------------
DINOv2 block 0 post-LN is recomputed in this same process, on the same images, so
its rho_var must land near the recorded +0.408 and its baseline-vs-spatial
quartile pattern near .731/.582/.487/.359 vs .695/.541/.405/.303. If the control
does not reproduce, nothing else in the output means anything.

Two differences from the Aug-19 run are deliberate and must be stated with any
result: (1) contiguity comes from the shared `autocorr_per_feature` in
dino_smoothness.py rather than the ad-hoc `ac1` (algebraically the same function
-- same h/v average, same 1e-10 constant-map mask, same average over images);
(2) each arm's preprocessing is the one its checkpoint was trained on, taken from
run_args.json, so the DINOv2 resize is bicubic here where the old ad-hoc `T` used
the torchvision default. Neither should move a Spearman rho materially, and the
post-LN control is what checks that claim.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

THIS = Path(__file__).resolve()
for _p in [THIS.parent, THIS.parent.parent,
           THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from arm_pipeline import pipeline_from_ckpt, assert_same_pipeline  # noqa: E402
from dino_smoothness import autocorr_per_feature  # noqa: E402
from fvu_compare import load_sae  # noqa: E402
from repro_utils import check_gpu_match, gpu_fingerprint  # noqa: E402

SAE_ROOT = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
                "Projects/T-SAE-Spatial")

# (condition, spatial_dir, baseline_dir) per seed. Seed 0 of the post-LN arm is
# named differently from seeds 1/2 -- that is historical, not a mistake.
CONDITIONS = {
    "dino_b0_postLN": [("spatial_block0", "spatial_block0_baseline"),
                       ("block0_spatial_s1", "block0_baseline_s1"),
                       ("block0_spatial_s2", "block0_baseline_s2")],
    "dino_b0_nonorm": [(f"dino_b0_nonorm_spatial_s{s}", f"dino_b0_nonorm_baseline_s{s}")
                       for s in (0, 1, 2)],
    "siglip_b0_raw": [(f"siglip_b0_raw_spatial_s{s}", f"siglip_b0_raw_baseline_s{s}")
                      for s in (0, 1, 2)],
    # Channel-whitened block 0 (job 42035857). Same backbone / block / LayerNorm as
    # dino_b0_postLN -- only the per-channel rescale differs, so post-LN is the live
    # control for this row. NOT pairable against it: assert_same_pipeline compares
    # channel_transform and channel_stats_sha and will refuse, by design.
    "dino_b0_whiten": [(f"dino_b0_whiten_spatial_s{s}", f"dino_b0_whiten_baseline_s{s}")
                       for s in (0, 1, 2)],
}


def img_bytes(row):
    d = row["image"]
    return d["bytes"] if isinstance(d, dict) else d


@torch.no_grad()
def features(pipe, rows, device):
    """[n_img, D, side, side] maps and [n_tok, D] pooled tokens, one arm's convention."""
    toks = []
    for _, r in rows.iterrows():
        try:
            toks.append(pipe.tokens(Image.open(io.BytesIO(img_bytes(r))).convert("RGB")))
        except Exception:
            pass
    side = pipe.side(toks[0].shape[0])
    maps = torch.stack([t.T.reshape(-1, side, side) for t in toks], 0)
    return toks, maps, torch.cat(toks, 0)


def quartile_rhos(contig, y, energy, ok, min_n=20):
    """Overall rho plus rho within each energy quartile of the 768 dims."""
    out = [dict(ebin=0, n=int(ok.sum()), **_rho(contig[ok], y[ok]),
                mean_y=float(np.mean(y[ok])))]
    q = np.nanquantile(energy[ok], [0.25, 0.5, 0.75])
    eb = np.digitize(energy, q)
    for b in range(4):
        m = ok & (eb == b)
        if m.sum() < min_n:
            continue
        out.append(dict(ebin=b + 1, n=int(m.sum()), **_rho(contig[m], y[m]),
                        mean_y=float(np.mean(y[m]))))
    return out


def _rho(a, b):
    r = spearmanr(a, b)
    return dict(rho=float(r.correlation), p=float(r.pvalue))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/n/home12/binxuwang/T-SAE-Spatial/imagenet_val_2k.parquet",
                    help="the Aug-19 run used this file with n_images=200, random_state=0")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    print(f"[gpu] {gpu_fingerprint(dev)}", flush=True)

    rows = pd.read_parquet(args.parquet).sample(args.n_images, random_state=args.sample_seed)
    raw_rows, share_rows, group_rows, ckpt_dirs = [], [], [], []

    for cond in args.conditions:
        seeds = CONDITIONS[cond]
        pipes = [pipeline_from_ckpt(SAE_ROOT / sp, dev) for sp, _ in seeds]
        pipes += [pipeline_from_ckpt(SAE_ROOT / bl, dev) for _, bl in seeds]
        pipe = assert_same_pipeline(*pipes)   # all 6 arms must see identical features
        print(f"\n=== {cond}: {pipe.describe()} ===", flush=True)

        toks, maps, X = features(pipe, rows, dev)
        ac, _ = autocorr_per_feature(maps.to(dev), lags=(1,))
        contig = ac[1]
        del maps
        torch.cuda.empty_cache()

        energy = X.var(0).cpu().numpy()
        absmean = X.abs().mean(0).cpu().numpy()
        ok = np.isfinite(contig)

        # ── shortcut 1: energy ────────────────────────────────────────────
        rv, ra = _rho(contig[ok], energy[ok]), _rho(contig[ok], absmean[ok])
        qc = np.nanquantile(contig[ok], [1 / 3, 2 / 3])
        lo, hi = energy[contig <= qc[0]].mean(), energy[contig >= qc[1]].mean()
        print(f"  shortcut 1  contiguity vs VARIANCE   rho={rv['rho']:+.3f} (p={rv['p']:.1e})"
              f"   vs mean|x| rho={ra['rho']:+.3f}", flush=True)
        print(f"              least-contiguous third var {lo:.3f}  most {hi:.3f}"
              f"  ratio {hi / max(lo, 1e-9):.2f}x", flush=True)

        # ── shortcut 2: low-rank participation (arm-independent) ──────────
        Xn = X.cpu().numpy()
        p = PCA(n_components=args.n_pcs, random_state=0).fit(Xn)
        rec = p.inverse_transform(p.transform(Xn))
        lowrank = 1 - ((Xn - rec).var(0) / (energy + 1e-12))
        for r in quartile_rhos(contig, lowrank, energy, ok):
            print(f"  shortcut 2  lowrank  ebin={r['ebin']}  n={r['n']:3d}  "
                  f"rho={r['rho']:+.3f} (p={r['p']:.2e})  mean capture {r['mean_y']:.3f}",
                  flush=True)
            raw_rows.append(dict(cond=cond, quantity="lowrank", **r))
        for r in quartile_rhos(contig, energy, energy, ok):
            raw_rows.append(dict(cond=cond, quantity="energy_selfcheck", **r))
        raw_rows.append(dict(cond=cond, quantity="energy", ebin=0, n=int(ok.sum()),
                             rho=rv["rho"], p=rv["p"], mean_y=float(energy[ok].mean()),
                             rho_absmean=ra["rho"], var_low=float(lo), var_high=float(hi)))
        del Xn, rec
        torch.cuda.empty_cache()

        # ── G0 share: the arm-dependent quantity (this is what fig5 plots) ─
        for seed, (sp, bl) in enumerate(seeds):
            for arm, name in (("spatial", sp), ("baseline", bl)):
                d = SAE_ROOT / name
                ae, gs = load_sae(str(d / "ae_latest.pt"), dev)
                g0 = gs[0]
                fvu = pipe.selftest(ae, [Image.open(io.BytesIO(img_bytes(r))).convert("RGB")
                                         for _, r in rows.iloc[:16].iterrows()],
                                    label=f"{cond}/{arm}/s{seed}")
                # Each group's DENSE output contribution z_g @ W_dec_g, kept per image
                # so its own contiguity can be measured. This is Wendy's Aug-19 fix
                # applied to BOTH groups: a k=32/16384 sparse latent has a spiky map by
                # construction, so latent-level contiguity is partly a sparsity
                # artifact, and only ~18-22% of latents are even scorable. The dense
                # 768-d reconstruction has neither problem.
                side = int(round(toks[0].shape[0] ** 0.5))
                v0 = torch.zeros(X.shape[1], device=dev)
                v1 = torch.zeros(X.shape[1], device=dev)
                m0, m1 = [], []
                for t_ in toks:
                    z = ae.encode(t_, use_threshold=False)
                    y0 = z[:, :g0] @ ae.W_dec[:g0]
                    y1 = z[:, g0:] @ ae.W_dec[g0:]
                    v0 += y0.var(0)
                    v1 += y1.var(0)
                    m0.append(y0.T.reshape(-1, side, side))
                    m1.append(y1.T.reshape(-1, side, side))
                g0frac = (v0 / (v0 + v1 + 1e-12)).cpu().numpy()
                qr = quartile_rhos(contig, g0frac, energy, ok)
                for r in qr:
                    share_rows.append(dict(cond=cond, arm=arm, seed=seed, fvu=fvu, **r))
                print(f"  g0_share  {arm:8s} s{seed}  overall rho={qr[0]['rho']:+.3f}"
                      f"  quartiles " + "/".join(f"{r['rho']:+.3f}" for r in qr[1:])
                      + f"  (fvu {fvu:.3f})", flush=True)

                # (i) ABSOLUTE g0_share level by CONTIGUITY tercile. The rho above is
                # rank-based, so it says whether the ORDERING tracks smoothness but not
                # how much of a smooth channel G0 actually takes. Wendy asked for the
                # level directly ("G0 intake 所有 smooth feature"), so report it.
                lvl = {}
                for name_, m in (("rough", contig <= qc[0]),
                                 ("mid", (contig > qc[0]) & (contig < qc[1])),
                                 ("smooth", contig >= qc[1])):
                    mm = m & ok
                    lvl[name_] = float(g0frac[mm].mean()) if mm.sum() else float("nan")
                # (ii) G0-vs-G1 SMOOTHNESS of the dense group reconstructions.
                acg0, n0 = autocorr_per_feature(torch.stack(m0, 0), lags=(1,))
                acg1, n1 = autocorr_per_feature(torch.stack(m1, 0), lags=(1,))
                del m0, m1
                torch.cuda.empty_cache()
                g1_energy_share = float((v1.sum() / (v0.sum() + v1.sum() + 1e-12)).item())
                grp = dict(cond=cond, arm=arm, seed=seed, fvu=fvu,
                           g0share_rough=lvl["rough"], g0share_mid=lvl["mid"],
                           g0share_smooth=lvl["smooth"],
                           recon_contig_G0=float(np.nanmedian(acg0[1])),
                           recon_contig_G1=float(np.nanmedian(acg1[1])),
                           n_scored_G0=int((n0 > 0).sum()), n_scored_G1=int((n1 > 0).sum()),
                           g1_energy_share=g1_energy_share)
                group_rows.append(grp)
                print(f"            g0_share LEVEL by contiguity tercile: "
                      f"rough {lvl['rough']:.3f}  mid {lvl['mid']:.3f}  "
                      f"smooth {lvl['smooth']:.3f}", flush=True)
                print(f"            dense recon contiguity  G0 {grp['recon_contig_G0']:+.3f}"
                      f" ({grp['n_scored_G0']}/768 scored)   G1 "
                      f"{grp['recon_contig_G1']:+.3f} ({grp['n_scored_G1']}/768)"
                      f"   G1 share of total recon variance {g1_energy_share:.4f}", flush=True)
                ckpt_dirs.append(d)
                del ae
                torch.cuda.empty_cache()
        del toks, X
        torch.cuda.empty_cache()

    raw = pd.DataFrame(raw_rows)
    share = pd.DataFrame(share_rows)
    raw.to_csv(out / "shortcut_raw.csv", index=False)
    share.to_csv(out / "shortcut_g0_share.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "shortcut_group_smoothness.csv", index=False)

    print("\n=== shortcut strength on the RAW features (the pre-registered test) ===")
    print(raw[raw.quantity.isin(["energy", "lowrank"]) & (raw.ebin == 0)]
          [["cond", "quantity", "rho", "p"]].to_string(index=False))
    print("\n=== g0_share: spatial vs baseline, overall and by energy quartile ===")
    piv = (share[share.ebin >= 0].groupby(["cond", "ebin", "arm"]).rho.mean()
           .unstack("arm"))
    piv["baseline_minus_spatial"] = piv["baseline"] - piv["spatial"]
    print(piv.to_string())
    piv.to_csv(out / "shortcut_g0_share_summary.csv")

    # Reported by default, per the Aug-17 reproducibility rule. The 6 post-LN
    # checkpoints predate the fingerprint field, so they come back UNRECORDED.
    gpu = check_gpu_match(ckpt_dirs)
    (out / "gpu_check.json").write_text(json.dumps(gpu, indent=2, default=str))
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
