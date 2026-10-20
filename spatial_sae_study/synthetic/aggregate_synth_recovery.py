"""
Aggregate the per-seed synthetic recovery results.

Two things the per-run summary cannot do:
  1. pool the 2x2 G0 tables across seeds (per-seed n is only 32+32)
  2. compare spatial vs baseline as a PAIRED contrast -- within a seed the two
     runs saw identical data and identical init, so seed is a valid pairing unit
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_rel, wilcoxon


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--per_gt_csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    RUN_RE = r"^(?P<arm>spatial|baseline)_d(?P<dict>\d+)_k(?P<kk>\d+)_s(?P<seed>\d+)$"
    s = pd.read_csv(args.summary_csv)
    pg = pd.read_csv(args.per_gt_csv)
    for df in (s, pg):
        p = df["run"].str.extract(RUN_RE)
        for c in ("arm", "dict", "kk", "seed"):
            df[c] = p[c]
        for c in ("dict", "kk", "seed"):
            df[c] = df[c].astype(int)
    s["config"] = "d" + s["dict"].astype(str) + "_k" + s["kk"].astype(str)
    pg["config"] = "d" + pg["dict"].astype(str) + "_k" + pg["kk"].astype(str)

    rows = []
    for (dsz, kk_, arm), g in pg.groupby(["dict", "kk", "arm"]):
        sm = g[g["kind"] == "smooth_v"]; ro = g[g["kind"] == "rough_u"]
        tbl = [[int(sm["best_in_G0"].sum()), int((~sm["best_in_G0"]).sum())],
               [int(ro["best_in_G0"].sum()), int((~ro["best_in_G0"]).sum())]]
        odds, p = fisher_exact(tbl)
        null = s[(s["dict"] == dsz) & (s["kk"] == kk_) & (s["arm"] == arm)]["null_G0_frac"].mean()
        rows.append(dict(dict_size=dsz, k=kk_, config=f"d{dsz}_k{kk_}",
                         arm=arm, n_seeds=g["seed"].nunique(),
                         smooth_in_G0=tbl[0][0], smooth_n=sum(tbl[0]),
                         rough_in_G0=tbl[1][0], rough_n=sum(tbl[1]),
                         smooth_G0_frac=tbl[0][0]/max(sum(tbl[0]),1),
                         rough_G0_frac=tbl[1][0]/max(sum(tbl[1]),1),
                         null_G0_frac=null, odds=odds, fisher_p=p, sig=stars(p),
                         recov_cos_smooth=sm["best_cos"].mean(),
                         recov_cos_rough=ro["best_cos"].mean()))
    pooled = pd.DataFrame(rows).sort_values(["dict_size", "k", "arm"])
    pooled.to_csv(outdir / "synth_pooled_by_arm.csv", index=False)
    print("=== Pooled across seeds (Q2: are smooth features in G0?) ===")
    print(pooled.to_string(index=False))

    # paired spatial vs baseline, seed as the pairing unit
    prows = []
    metrics = ["smooth_G0_frac", "rough_G0_frac", "recov_cos_smooth", "recov_cos_rough",
               "mean_autocorr_G0", "mean_autocorr_G1", "n_alive"]
    for cfg, g in s.groupby("config"):
        sp = g[g["arm"] == "spatial"].sort_values("seed")
        bl = g[g["arm"] == "baseline"].sort_values("seed")
        common = sorted(set(sp["seed"]) & set(bl["seed"]))
        sp = sp[sp["seed"].isin(common)]; bl = bl[bl["seed"].isin(common)]
        for m in metrics:
            a, b = sp[m].to_numpy(float), bl[m].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 2:
                continue
            p = ttest_rel(a[ok], b[ok]).pvalue
            prows.append(dict(config=cfg, metric=m, n=int(ok.sum()),
                              spatial=a[ok].mean(), baseline=b[ok].mean(),
                              diff=a[ok].mean()-b[ok].mean(), paired_t_p=p, sig=stars(p)))
    paired = pd.DataFrame(prows)
    paired.to_csv(outdir / "synth_paired_spatial_vs_baseline.csv", index=False)
    print("\n=== Paired spatial vs baseline (Q3: is it the contrastive loss?) ===")
    print(paired.to_string(index=False))

    # figure: one column pair per (dict, k) config
    configs = sorted(pooled["config"].unique())
    fig, axes = plt.subplots(1, len(configs) * 2, figsize=(6.2 * len(configs), 5))
    axes = np.atleast_1d(axes)
    for di, dsz in enumerate(configs):
        sub = pooled[pooled["config"] == dsz]
        ax = axes[di * 2]
        arms = sub["arm"].tolist(); x = np.arange(len(arms)); w = 0.38
        ax.bar(x - w/2, sub["smooth_G0_frac"], w, label="smooth $v_k$", color="#1f77b4")
        ax.bar(x + w/2, sub["rough_G0_frac"], w, label="rough $u_k$", color="#d62728")
        for i, (_, rr) in enumerate(sub.iterrows()):
            ax.plot([i-0.5, i+0.5], [rr["null_G0_frac"]]*2, ls="--", c="k", lw=1.2)
            ax.text(i, max(rr["smooth_G0_frac"], rr["rough_G0_frac"])+0.03,
                    f"{rr['sig']}\np={rr['fisher_p']:.1e}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(arms)
        ax.set_ylabel("fraction of GT features whose best latent is in G0")
        ax.set_title(f"{dsz}\nQ2: smooth vs rough into G0")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

        ax = axes[di * 2 + 1]
        ax.bar(x - w/2, sub["recov_cos_smooth"], w, label="smooth $v_k$", color="#1f77b4")
        ax.bar(x + w/2, sub["recov_cos_rough"], w, label="rough $u_k$", color="#d62728")
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_xticks(x); ax.set_xticklabels(arms)
        ax.set_ylabel("max cosine to best alive latent")
        ax.set_title(f"{dsz}\nQ1: dictionary recovery")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Synthetic toy, pooled over 3 seeds — dashed line = alive-latent null", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "synth_pooled.png", dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {outdir}")


if __name__ == "__main__":
    main()
