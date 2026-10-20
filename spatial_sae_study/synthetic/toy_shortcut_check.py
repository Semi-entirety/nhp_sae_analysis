"""
Do the two real-DINO shortcuts exist in the synthetic top-k toy?

Why this is worth a run
-----------------------
"The toy is shortcut-free" has so far been INFERRED from its design (random
dictionary, analytically matched energy and density across the smooth and rough
groups) and never MEASURED. Meanwhile the real-DINO shortcuts were measured, by
`shortcut_analysis.py`, on the 768 raw channels:

    DINOv2 b0 post-LN   rho(contig, low-rank participation) = +0.783
                        rho(contig, per-dim variance)       = +0.401

Those two numbers are the reason baseline sorts smooth features into G0 for free
on real DINO but not in the toy. If the explanation is right, the SAME two
correlations computed on the SAME code path over the toy substrate must be ~0.

This script therefore re-uses, verbatim:
  * `autocorr_per_feature` from dino_smoothness.py          (contiguity)
  * the energy / low-rank / g0_share formulas and the energy-quartile
    stratification of shortcut_analysis.py                  (the shortcuts)
and only swaps the substrate: DINOv2 patch tokens -> TopKSpatialSource samples.
Both are [n_img, 256 tokens, 768 channels] on a 16x16 grid, so the two tables are
directly comparable, which is the entire point.

Three quantities, all over the 768 OBSERVED CHANNELS (the unit every real-DINO
heterogeneity number uses -- NOT the ground-truth features):
  1. the contiguity DISTRIBUTION (median / p90-p10 / frac<0.1), against
     DINOv2 b0's 0.069 / 0.346 / 0.672;
  2. the two shortcuts, overall and within energy quartiles;
  3. rho(contig, g0_share) per arm, the toy analogue of real DINO's block-0
     +0.625 (baseline) / +0.568 (spatial).

Also reported in ground-truth CODE space, where the toy's smooth/rough split was
built to live, so it is visible whether that split survives the random mixing
into 768 observed channels.

PRE-REGISTERED (written before the run): if the shortcut account is correct,
both channel-space rhos are ~0 (|rho| < 0.1) in the toy while real DINO b0 is
+0.783 / +0.401. Anything substantially non-zero falsifies "the toy is
shortcut-free by construction" and forces the real-DINO/toy divergence to be
re-explained.

Reproducibility: the dictionary is ALWAYS loaded from each run's saved
ground_truth.npz, never regenerated from the seed (torch CUDA RNG is not
reproducible across GPU architectures -- regenerating on a different GPU gave a
completely unrelated dictionary, max abs diff 0.24). Evaluation images are fresh
draws from the same process, which is intended.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

THIS = Path(__file__).resolve()
for p in [THIS.parent, THIS.parent.parent,
          THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dino_smoothness import autocorr_per_feature, load_sae
from synth_topk_gen import TopKConfig, TopKSpatialSource
from repro_utils import check_gpu_match

STORE = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang")
TOY = STORE / "Projects/T-SAE-Spatial/synth_topk"


def _rho(a, b):
    r = spearmanr(a, b)
    return dict(rho=float(r.correlation), p=float(r.pvalue))


def quartile_rhos(contig, y, energy, ok, min_n=20):
    """Overall rho plus rho within each energy quartile -- shortcut_analysis.py's."""
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


def source_for_run(run_dir, device):
    """Rebuild the generator, taking the dictionary from the SAVED ground truth."""
    a = json.loads((Path(run_dir) / "run_args.json").read_text())
    cfg = TopKConfig(d_model=a["d_model"], side=a["side"], d_true=a["d_true"],
                     k=a["gen_k"], corr_lengths=list(a["corr_lengths"]),
                     group_names=[f"l={c:g}" for c in a["corr_lengths"]],
                     seed=a["data_seed"])
    src = TopKSpatialSource(cfg, device=device)
    gt = np.load(Path(run_dir) / "ground_truth.npz")
    src.dictionary = torch.tensor(gt["dictionary"], device=device,
                                  dtype=src.dictionary.dtype)
    src.group_id = torch.tensor(gt["group_id"], device=device)
    if src.dictionary.shape[0] != cfg.d_true:
        raise RuntimeError(f"{run_dir}: saved dictionary has {src.dictionary.shape[0]} "
                           f"rows, expected d_true={cfg.d_true}")
    return src, a


def dist_stats(contig):
    """The three columns every heterogeneity table in this project reports."""
    v = contig[np.isfinite(contig)]
    return dict(n_scored=int(v.size),
                lag1_median=float(np.median(v)),
                lag1_p90_minus_p10=float(np.nanpercentile(v, 90) - np.nanpercentile(v, 10)),
                frac_below_0p1=float(np.mean(v < 0.1)),
                skew=float(pd.Series(v).skew()),
                excess_kurtosis=float(pd.Series(v).kurtosis()))


def maps_from_tokens(X, side):
    """[n_img, P, F] -> [n_img, F, side, side]; token p = row*side + col."""
    n_img, P, F = X.shape
    assert P == side * side, f"{P} tokens != {side}x{side}"
    return X.reshape(n_img, side, side, F).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=200)   # = the real-DINO run
    ap.add_argument("--n_pcs", type=int, default=50)       # = the real-DINO run
    ap.add_argument("--d_true", nargs="+", type=int, default=[768, 1536])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    # --base / --conds exist so the SAME measurement runs on the shortcut-injection
    # arms (job 41994343) without a second copy of any formula. Defaults reproduce
    # the original job-41988315 invocation byte for byte: base = synth_topk and the
    # single condition "toy" maps to the un-prefixed `spatial_d{D}_s{s}` dirs.
    ap.add_argument("--base", default=str(TOY))
    ap.add_argument("--conds", nargs="+", default=["toy"],
                    help="checkpoint-name prefixes, e.g. ctrl E L LL; 'toy' = no prefix")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    base = Path(a.base)
    dev = a.device
    raw_rows, dist_rows, g0_rows, code_rows = [], [], [], []
    used_dirs = []

    for cname, D in [(c, d) for c in a.conds for d in a.d_true]:
        for s in a.seeds:
            pfx = "" if cname == "toy" else f"{cname}_"
            sp_dir = base / f"{pfx}spatial_d{D}_s{s}"
            bl_dir = base / f"{pfx}baseline_d{D}_s{s}"
            if not sp_dir.exists():
                print(f"[skip] {sp_dir} missing"); continue
            used_dirs += [sp_dir, bl_dir]
            cond = f"{cname}_d{D}_s{s}"

            src, args_sp = source_for_run(sp_dir, dev)
            # spatial and baseline at the same seed must share one substrate --
            # if they do not, nothing in this project's toy pairing is valid.
            gt_bl = np.load(bl_dir / "ground_truth.npz")
            dd = float(np.abs(gt_bl["dictionary"] - src.dictionary.cpu().numpy()).max())
            assert dd == 0.0, f"{cond}: spatial/baseline dictionaries differ by {dd}"
            assert args_sp["data_seed"] == json.loads(
                (bl_dir / "run_args.json").read_text())["data_seed"]
            print(f"\n=== {cond}  (paired substrate verified, max diff {dd:.1e}) ===",
                  flush=True)

            side = src.cfg.side
            X, code = src.sample_images(a.n_images, return_code=True)

            # ---- contiguity of each OBSERVED CHANNEL -------------------------
            ac, _ = autocorr_per_feature(maps_from_tokens(X, side).to(dev), lags=(1,))
            contig = ac[1]
            Xt = X.reshape(-1, X.shape[-1])                    # [n_tok, 768]
            energy = Xt.var(0).cpu().numpy()
            ok = np.isfinite(contig)
            dist_rows.append(dict(cond=cond, condition=cname, space="channel", d_true=D, seed=s,
                                  n_units=int(contig.size), **dist_stats(contig)))

            # ---- contiguity in ground-truth CODE space (where the split lives)
            ac_c, _ = autocorr_per_feature(maps_from_tokens(code, side).to(dev), lags=(1,))
            gid = src.group_id.cpu().numpy()
            dist_rows.append(dict(cond=cond, condition=cname, space="code", d_true=D, seed=s,
                                  n_units=int(ac_c[1].size), **dist_stats(ac_c[1])))
            for gi, nm in enumerate(src.cfg.group_names):
                cc = ac_c[1][gid == gi]
                code_rows.append(dict(cond=cond, condition=cname, group=nm, n=int(cc.size),
                                      **dist_stats(cc)))

            # ---- shortcut 1: energy ------------------------------------------
            qc = np.nanquantile(contig[ok], [1 / 3, 2 / 3])
            lo = float(energy[contig <= qc[0]].mean())
            hi = float(energy[contig >= qc[1]].mean())
            r_e = _rho(contig[ok], energy[ok])
            raw_rows.append(dict(cond=cond, condition=cname, d_true=D, seed=s, quantity="energy",
                                 ebin=0, n=int(ok.sum()), **r_e,
                                 var_low_third=lo, var_high_third=hi,
                                 var_ratio=hi / (lo + 1e-12)))

            # ---- shortcut 2: low-rank participation (arm-independent) --------
            Xn = Xt.cpu().numpy()
            p = PCA(n_components=a.n_pcs, random_state=0).fit(Xn)
            rec = p.inverse_transform(p.transform(Xn))
            lowrank = 1 - ((Xn - rec).var(0) / (energy + 1e-12))
            for r in quartile_rhos(contig, lowrank, energy, ok):
                raw_rows.append(dict(cond=cond, condition=cname, d_true=D, seed=s,
                                     quantity="lowrank", **r))
            print(f"  rho(contig, energy)  = {r_e['rho']:+.3f}  (p={r_e['p']:.2g})"
                  f"   var ratio hi/lo = {hi/(lo+1e-12):.2f}x")
            print(f"  rho(contig, lowrank) = {raw_rows[-5]['rho']:+.3f}"
                  f"   (EVR{a.n_pcs} = {p.explained_variance_ratio_.sum():.3f})")
            print(f"  channel contiguity: median {np.median(contig[ok]):+.3f}  "
                  f"spread {np.nanpercentile(contig[ok],90)-np.nanpercentile(contig[ok],10):.3f}  "
                  f"rough {np.mean(contig[ok]<0.1):.3f}")

            # ---- arm-dependent: rho(contig, that channel's G0 share) ---------
            for arm, rd in (("spatial", sp_dir), ("baseline", bl_dir)):
                ae, gs = load_sae(str(rd / "ae_latest.pt"), dev)
                g0 = gs[0]
                v0 = torch.zeros(Xt.shape[1], device=dev)
                v1 = torch.zeros(Xt.shape[1], device=dev)
                sse = torch.zeros((), device=dev)
                for i in range(X.shape[0]):
                    t_ = X[i]
                    z = ae.encode(t_, use_threshold=False)
                    r0 = z[:, :g0] @ ae.W_dec[:g0]
                    r1 = z[:, g0:] @ ae.W_dec[g0:] if g0 < z.shape[1] else torch.zeros_like(r0)
                    v0 += r0.var(0); v1 += r1.var(0)
                    sse += (t_ - (r0 + r1 + ae.b_dec)).pow(2).sum()
                fvu = float(sse / (Xt - Xt.mean(0)).pow(2).sum())
                assert fvu < 0.95, f"{cond}/{arm}: FVU {fvu:.3f} -- wrong substrate?"
                g0frac = (v0 / (v0 + v1 + 1e-12)).cpu().numpy()
                for r in quartile_rhos(contig, g0frac, energy, ok):
                    g0_rows.append(dict(cond=cond, condition=cname, d_true=D, seed=s, arm=arm,
                                        fvu=fvu, **r))
                print(f"  {arm:>8}: FVU {fvu:.3f}   rho(contig, g0_share) = "
                      f"{g0_rows[-5]['rho']:+.3f}")
                del ae
                torch.cuda.empty_cache()

            del X, code, Xt, Xn
            torch.cuda.empty_cache()

    pd.DataFrame(raw_rows).to_csv(out / "toy_shortcut_raw.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out / "toy_contig_dist.csv", index=False)
    pd.DataFrame(code_rows).to_csv(out / "toy_code_groups.csv", index=False)
    g0 = pd.DataFrame(g0_rows)
    g0.to_csv(out / "toy_g0_share.csv", index=False)
    if len(g0):
        piv = (g0.pivot_table(index=["condition", "d_true", "ebin"], columns="arm", values="rho")
                 .reset_index())
        piv["baseline_minus_spatial"] = piv.get("baseline") - piv.get("spatial")
        piv.to_csv(out / "toy_g0_share_summary.csv", index=False)

    rep = check_gpu_match([d for d in used_dirs if (d / "run_args.json").exists()])
    (out / "gpu_check.json").write_text(json.dumps(rep, indent=2, default=str))
    print("\nGPU CHECK:", json.dumps(rep, default=str)[:600])
    print("\nwrote", out)


if __name__ == "__main__":
    main()
