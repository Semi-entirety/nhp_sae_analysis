"""
Recovery analysis for the synthetic spatial-SAE toy case.

Questions, in the order Wendy posed them:

  Q1  Does the SAE learn neurons for each ground-truth direction v_k (smooth)
      and u_k (rough)?  -> max cosine similarity between each GT direction and
      the learned decoder columns, restricted to ALIVE latents.

  Q2  Are the neurons for v_k preferentially in G0?  -> for each GT direction,
      locate its best-matching latent and record which matryoshka group it
      falls in; compare the smooth and rough groups with Fisher's exact test.
      The relevant null is NOT 50/50: it is the fraction of ALIVE latents that
      live in G0, which is reported alongside.

  Q3  Is any G0 preference caused by the contrastive loss, or by the matryoshka
      prefix reconstruction alone?  -> the same analysis is run on a
      contrastive_weight=0 checkpoint trained on identical data.

Also reports the lag-1 spatial autocorrelation of every alive latent's own
activation map, which is the model-side counterpart of the smooth/rough
distinction built into the generator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import fisher_exact, mannwhitneyu, binomtest

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in [SCRIPTS_DIR, REPO_ROOT, REPO_ROOT / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE
from synth_spatial_gen import SynthConfig, SyntheticSpatialSource


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


def lag1_autocorr_maps(z, side):
    """z: [I, P, F] -> per-feature mean lag-1 spatial autocorrelation, [F]."""
    I, P, Fd = z.shape
    m = z.view(I, side, side, Fd).permute(0, 3, 1, 2)          # [I, F, s, s]
    m = m - m.mean(dim=(-2, -1), keepdim=True)
    var = (m ** 2).mean(dim=(-2, -1))                           # [I, F]
    h = (m[..., :, :-1] * m[..., :, 1:]).mean(dim=(-2, -1))
    v = (m[..., :-1, :] * m[..., 1:, :]).mean(dim=(-2, -1))
    ac = ((h + v) / 2) / (var + 1e-12)
    ok = var > 1e-10
    num = (ac * ok).sum(0)
    den = ok.sum(0).clamp(min=1)
    out = num / den
    out[ok.sum(0) == 0] = float("nan")
    return out


@torch.no_grad()
def encode_all(ae, src, n_img, device, chunk=16):
    zs = []
    for i in range(0, n_img, chunk):
        b = min(chunk, n_img - i)
        X = src.sample_images(b)                                # [b, P, D]
        z = ae.encode(X.reshape(-1, X.shape[-1]), use_threshold=False)
        zs.append(z.view(b, X.shape[1], -1))
    return torch.cat(zs, 0)


def source_for_run(ckpt, device, fallback_cfg):
    """Rebuild the EXACT generator this checkpoint was trained on.

    Each run has its own data_seed, hence its own ground-truth dictionary.
    Regenerating with a fixed seed and comparing every checkpoint against it
    silently measures every run except data_seed=0 against the wrong ground
    truth, which reads as 'recovery failed at chance level'.
    """
    run_dir = Path(ckpt).parent
    args_path = run_dir / "run_args.json"
    if not args_path.exists():
        return SyntheticSpatialSource(fallback_cfg, device=device)
    a = json.loads(args_path.read_text())
    cfg = SynthConfig(d_model=a.get("d_model", 768), side=a.get("side", 16),
                      k_smooth=a.get("k_smooth", 32), k_rough=a.get("k_rough", 32),
                      gp_sigma=a.get("gp_sigma", 2.0), tau=a.get("tau", 1.15),
                      sparsify=a.get("sparsify", True),
                      noise_std=a.get("noise_std", 0.0), seed=a.get("data_seed", 0))
    src = SyntheticSpatialSource(cfg, device=device)
    gt_path = run_dir / "ground_truth.npz"
    if gt_path.exists():   # verify the regenerated dictionary is the trained-on one
        gt = np.load(gt_path)
        ref = torch.tensor(gt["dictionary"], device=device, dtype=src.dictionary.dtype)
        err = float((ref - src.dictionary).abs().max())
        if err > 1e-4:
            raise RuntimeError(f"{run_dir.name}: regenerated dictionary differs from "
                               f"ground_truth.npz (max abs err {err:.2e})")
    return src


def analyse_one(name, ckpt, src, device, n_img, outdir):
    ae, gs = load_sae(ckpt, device)
    g0 = gs[0]
    W = ae.W_dec.detach()                                       # [dict, D]
    dict_size = W.shape[0]

    z = encode_all(ae, src, n_img, device)                      # [I, P, F]
    fire = (z > 0).any(dim=1).any(dim=0)                        # alive anywhere
    n_alive = int(fire.sum())
    alive_g0 = int(fire[:g0].sum())
    null_g0_frac = alive_g0 / max(n_alive, 1)

    ac = lag1_autocorr_maps(z, src.cfg.side)                    # [F]

    Wn = F.normalize(W, dim=-1)
    G = F.normalize(src.dictionary, dim=-1)                     # [K, D]
    cos = G @ Wn.T                                              # [K, dict]
    cos = cos.masked_fill(~fire.unsqueeze(0), float("-inf"))    # alive latents only

    best_cos, best_idx = cos.max(dim=1)
    best_cos = best_cos.cpu().numpy(); best_idx = best_idx.cpu().numpy()
    is_smooth = src.is_smooth.cpu().numpy()
    in_g0 = best_idx < g0

    # CHANCE BASELINE. With dict_size latents in D dimensions, the max cosine
    # between an arbitrary direction and the learned dictionary is already well
    # above zero (~sqrt(2 ln(dict)/D)). Without this control a "recovery" of
    # 0.16 would look like partial success when it is pure chance.
    rnd = F.normalize(torch.randn(512, W.shape[1], device=device), dim=-1)
    rnd_cos = (rnd @ Wn.T).masked_fill(~fire.unsqueeze(0), float("-inf")).max(dim=1).values
    chance_cos = float(rnd_cos.mean())
    chance_cos_p95 = float(torch.quantile(rnd_cos, 0.95))

    # SUBSPACE alignment. With dense Gaussian coefficients the individual axes
    # v_k / u_k are not identifiable (a Gaussian is rotationally symmetric
    # within its subspace), but the SUBSPACES they span still are. For each
    # alive latent direction w: what share of its energy lies in span(V) vs
    # span(U)?  This is the metric that stays meaningful with sparsify=False.
    # Note span(V) and span(U) are not exactly orthogonal (random dirs, max
    # off-diagonal cosine ~0.12), so the two shares are only near-complementary.
    Qv = torch.linalg.qr(src.V.T)[0]                            # [D, k_smooth]
    Qu = torch.linalg.qr(src.U.T)[0]                            # [D, k_rough]
    Wa = Wn[fire]
    ev = (Wa @ Qv).pow(2).sum(-1)
    eu = (Wa @ Qu).pow(2).sum(-1)
    smooth_share = (ev / (ev + eu + 1e-12)).cpu().numpy()
    idx_alive = np.arange(dict_size)[fire.cpu().numpy()]
    sh_g0 = smooth_share[idx_alive < g0]
    sh_g1 = smooth_share[idx_alive >= g0]
    p_share = (mannwhitneyu(sh_g0, sh_g1).pvalue
               if len(sh_g0) > 1 and len(sh_g1) > 1 else np.nan)

    rows = []
    for kk in range(len(is_smooth)):
        rows.append(dict(run=name, gt_index=kk,
                         kind="smooth_v" if is_smooth[kk] else "rough_u",
                         best_cos=float(best_cos[kk]), best_latent=int(best_idx[kk]),
                         best_in_G0=bool(in_g0[kk]),
                         best_latent_autocorr=float(ac[best_idx[kk]])))
    per_gt = pd.DataFrame(rows)

    s, r = is_smooth, ~is_smooth
    tbl = [[int((in_g0 & s).sum()), int((~in_g0 & s).sum())],
           [int((in_g0 & r).sum()), int((~in_g0 & r).sum())]]
    odds, p_fisher = fisher_exact(tbl)
    p_smooth_vs_null = binomtest(tbl[0][0], sum(tbl[0]), null_g0_frac).pvalue
    p_rough_vs_null = binomtest(tbl[1][0], sum(tbl[1]), null_g0_frac).pvalue
    p_cos = mannwhitneyu(best_cos[s], best_cos[r]).pvalue

    ac_np = ac.cpu().numpy(); fire_np = fire.cpu().numpy()
    ac_g0 = ac_np[:g0][fire_np[:g0]]; ac_g1 = ac_np[g0:][fire_np[g0:]]
    ac_g0 = ac_g0[np.isfinite(ac_g0)]; ac_g1 = ac_g1[np.isfinite(ac_g1)]
    p_ac = mannwhitneyu(ac_g0, ac_g1).pvalue if len(ac_g0) > 1 and len(ac_g1) > 1 else np.nan

    summ = dict(
        run=name, dict_size=dict_size, G0_size=g0,
        n_alive=n_alive, alive_in_G0=alive_g0, null_G0_frac=null_g0_frac,
        recov_cos_smooth=float(best_cos[s].mean()), recov_cos_rough=float(best_cos[r].mean()),
        recov_cos_p=float(p_cos),
        chance_cos=chance_cos, chance_cos_p95=chance_cos_p95,
        n_recovered_smooth_cos50=int((best_cos[s] > 0.5).sum()),
        n_recovered_rough_cos50=int((best_cos[r] > 0.5).sum()),
        n_smooth_above_chance95=int((best_cos[s] > chance_cos_p95).sum()),
        n_rough_above_chance95=int((best_cos[r] > chance_cos_p95).sum()),
        smooth_in_G0=tbl[0][0], smooth_total=sum(tbl[0]),
        rough_in_G0=tbl[1][0], rough_total=sum(tbl[1]),
        smooth_G0_frac=tbl[0][0] / max(sum(tbl[0]), 1),
        rough_G0_frac=tbl[1][0] / max(sum(tbl[1]), 1),
        fisher_odds=float(odds), fisher_p=float(p_fisher),
        smooth_vs_null_p=float(p_smooth_vs_null), rough_vs_null_p=float(p_rough_vs_null),
        mean_autocorr_G0=float(np.mean(ac_g0)) if len(ac_g0) else np.nan,
        mean_autocorr_G1=float(np.mean(ac_g1)) if len(ac_g1) else np.nan,
        autocorr_G0_vs_G1_p=float(p_ac),
        sparsify=bool(src.cfg.sparsify),
        smooth_share_G0=float(np.mean(sh_g0)) if len(sh_g0) else np.nan,
        smooth_share_G1=float(np.mean(sh_g1)) if len(sh_g1) else np.nan,
        smooth_share_p=float(p_share),
    )
    per_lat = pd.DataFrame(dict(
        run=name, latent=np.arange(dict_size)[fire_np],
        group=np.where(np.arange(dict_size)[fire_np] < g0, "G0", "G1"),
        autocorr=ac_np[fire_np]))
    return summ, per_gt, per_lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name=path pairs, e.g. spatial=/path/ae_latest.pt")
    ap.add_argument("--gp_sigma", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.15)
    ap.add_argument("--k_smooth", type=int, default=32)
    ap.add_argument("--k_rough", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--side", type=int, default=16)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--n_eval_images", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    fallback_cfg = SynthConfig(d_model=args.d_model, side=args.side, k_smooth=args.k_smooth,
                               k_rough=args.k_rough, gp_sigma=args.gp_sigma, tau=args.tau,
                               seed=args.data_seed)

    summaries, per_gts, per_lats = [], [], []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        # per-run generator: each data_seed has its own ground-truth dictionary
        src = source_for_run(path, args.device, fallback_cfg)
        print(f"[Run] {name}: {path} (data_seed={src.cfg.seed})", flush=True)
        s, pg, pl = analyse_one(name, path, src, args.device, args.n_eval_images, outdir)
        summaries.append(s); per_gts.append(pg); per_lats.append(pl)
        print(json.dumps(s, indent=2, default=str), flush=True)

    summ = pd.DataFrame(summaries)
    per_gt = pd.concat(per_gts, ignore_index=True)
    per_lat = pd.concat(per_lats, ignore_index=True)
    summ.to_csv(outdir / "synth_recovery_summary.csv", index=False)
    per_gt.to_csv(outdir / "synth_recovery_per_gt.csv", index=False)
    per_lat.to_csv(outdir / "synth_recovery_per_latent.csv", index=False)
    print(summ.to_string(index=False))

    runs = summ["run"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    x = np.arange(len(runs)); w = 0.38
    ax.bar(x - w/2, summ["recov_cos_smooth"], w, label="smooth $v_k$", color="#1f77b4")
    ax.bar(x + w/2, summ["recov_cos_rough"], w, label="rough $u_k$", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(runs, rotation=15, ha="right")
    ax.set_ylabel("max cosine to best alive latent"); ax.set_title("Q1  dictionary recovery")
    for i, rrow in summ.iterrows():
        ax.plot([i - 0.5, i + 0.5], [rrow["chance_cos"]] * 2, ls="--", c="k", lw=1.2)
    ax.plot([], [], ls="--", c="k", lw=1.2, label="chance (random direction)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - w/2, summ["smooth_G0_frac"], w, label="smooth $v_k$", color="#1f77b4")
    ax.bar(x + w/2, summ["rough_G0_frac"], w, label="rough $u_k$", color="#d62728")
    for i, rrow in summ.iterrows():
        ax.plot([i - 0.5, i + 0.5], [rrow["null_G0_frac"]] * 2, ls="--", c="k", lw=1.2)
        ax.text(i, max(rrow["smooth_G0_frac"], rrow["rough_G0_frac"]) + 0.02,
                f"p={rrow['fisher_p']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(runs, rotation=15, ha="right")
    ax.set_ylabel("fraction whose best latent is in G0")
    ax.set_title("Q2  are smooth features in G0?\n(dashed = alive-latent null)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    for i, run in enumerate(runs):
        d = per_lat[per_lat["run"] == run]
        for gname, c, off in [("G0", "#1f77b4", -0.18), ("G1", "#d62728", 0.18)]:
            vals = d[d["group"] == gname]["autocorr"].dropna().values
            if len(vals) == 0:
                continue
            ax.violinplot([vals], positions=[i + off], widths=0.3, showmeans=True)
    ax.set_xticks(x); ax.set_xticklabels(runs, rotation=15, ha="right")
    ax.set_ylabel("lag-1 spatial autocorrelation of latent map")
    ax.set_title("Q3  learned latents: G0 (left) vs G1 (right)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Synthetic toy: can the spatial SAE recover smooth vs rough generative features?", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "synth_recovery.png", dpi=150, bbox_inches="tight")
    print(f"[Saved] {outdir}")


if __name__ == "__main__":
    main()
