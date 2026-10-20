"""Cumulative PCA variance curve of the G0 sparse code, spatial vs baseline.

Wendy, Sep 14 2026: "能不能试一下 component 更少的 PCA 对比 baseline 和 spatial 的差异。
比如 PCA-10, PCA-20, PCA-30, PCA-40, PCA-50。"

The recorded measurement (`task_coherence` in backbone_eval.py) reports explained
variance at ONE cutoff, 64, fixed in advance. The paper explicitly calls that a
"PCA-64 diagnostic rather than evidence of greater compressibility in general; a
full cumulative variance curve would be needed for a cutoff-free claim". This is
that curve. PCA components are nested, so a single fit to `--n_pcs` components
gives every smaller k for free from the cumulative sum -- there is no reason to
refit per k, and refitting would only add solver noise between the k's.

LIVE CONTROL
------------
evr@64 must reproduce the recorded value for each arm to `--tol` (default 0.005):
  dino_b0_postLN  spatial .9473/.9526/.9533   baseline .8759/.8790/.8867
  dino_b1_postLN  spatial .9541/.9523/.9499   baseline .9198/.9249/.9187
  dino_b2_postLN  spatial .9479/.9469/.9440   baseline .9104/.9140/.9193
  dino_b0_nonorm  spatial .8628/.8677/.8508   baseline .6652/.6515/.6572
(sources: results/backbone_eval/coherence.csv, results/b12_eval/coherence.csv)
The script ASSERTS this. Do not relax the tolerance to make a future run pass.

TWO CONTROLS THE 64-ONLY MEASUREMENT NEVER HAD
----------------------------------------------
alive-matched   The two arms do not have the same number of alive G0 latents
                (b0 post-LN: spatial 1740/1823/1985 vs baseline 1855/1895/1935).
                A code with fewer alive dimensions concentrates more easily, and
                that bias is WORSE at small k -- exactly the regime asked about.
                So every curve is recomputed after subsampling both arms of a
                (cond, seed) pair to the same alive count, using one shared rng
                draw per pair. EVR is a ratio, so dropping columns is legitimate.
shuffled null   Each latent independently permuted across tokens: every marginal
                distribution is preserved, all cross-latent structure destroyed.
                This is the no-structure reference at every k. Neither of this
                project's two code-geometry metrics has ever had a chance level.

WHAT THIS DOES NOT FIX
----------------------
The PCA basis is fitted PER ARM. Two arms' curves are therefore not expressed in
a common basis. That is a property of the original measurement, carried over
deliberately rather than silently changed, and it is the long-standing Aug-21
flag. The alive-matched control does NOT address it.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path

# backbone_eval hardcodes TORCH_HOME to holylfs06 at import time. A holylfs06
# stall inside torch.hub.load ate an entire walltime once (job 45454047, TIMEOUT
# at 40 min on 14 s of CPU), so the sbatch stages the backbone to node-local
# $TMPDIR -- which means the override has to be re-asserted AFTER the import.
_WANT = os.environ.get("TSAE_TORCH_CACHE")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbone_eval as BE  # noqa: E402

if _WANT:
    os.environ["TORCH_HOME"] = _WANT
    import torch.hub as _hub

    _hub.set_dir(str(Path(_WANT) / "hub"))
    print(f"[stage] TORCH_HOME re-asserted -> {_WANT}", flush=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

# Recorded evr@64, for the live control. Keyed (cond, arm, seed).
RECORDED = {
    ("dino_b0_postLN", "spatial", 0): 0.9473, ("dino_b0_postLN", "baseline", 0): 0.8759,
    ("dino_b0_postLN", "spatial", 1): 0.9526, ("dino_b0_postLN", "baseline", 1): 0.8790,
    ("dino_b0_postLN", "spatial", 2): 0.9533, ("dino_b0_postLN", "baseline", 2): 0.8867,
    ("dino_b1_postLN", "spatial", 0): 0.9541, ("dino_b1_postLN", "baseline", 0): 0.9198,
    ("dino_b1_postLN", "spatial", 1): 0.9523, ("dino_b1_postLN", "baseline", 1): 0.9249,
    ("dino_b1_postLN", "spatial", 2): 0.9499, ("dino_b1_postLN", "baseline", 2): 0.9187,
    ("dino_b2_postLN", "spatial", 0): 0.9479, ("dino_b2_postLN", "baseline", 0): 0.9104,
    ("dino_b2_postLN", "spatial", 1): 0.9469, ("dino_b2_postLN", "baseline", 1): 0.9140,
    ("dino_b2_postLN", "spatial", 2): 0.9440, ("dino_b2_postLN", "baseline", 2): 0.9193,
    ("dino_b0_nonorm", "spatial", 0): 0.8628, ("dino_b0_nonorm", "baseline", 0): 0.6652,
    ("dino_b0_nonorm", "spatial", 1): 0.8677, ("dino_b0_nonorm", "baseline", 1): 0.6515,
    ("dino_b0_nonorm", "spatial", 2): 0.8508, ("dino_b0_nonorm", "baseline", 2): 0.6572,
}


def cum_evr(Z, n_pcs, seed=0):
    """Cumulative explained-variance ratio of Z, index k-1 == EVR at k components."""
    npc = int(min(n_pcs, Z.shape[1], Z.shape[0] - 1))
    p = PCA(n_components=npc, random_state=seed, svd_solver="randomized")
    p.fit(Z)
    return np.cumsum(p.explained_variance_ratio_).astype(np.float64)


@torch.no_grad()
def g0_codes(arm, images):
    """Pooled G0 code over all images, [n_img * n_tokens, G0], float32 on CPU."""
    out = []
    for im in images:
        out.append(arm.g0_code(arm.tokens(im)).cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/"
                                       "binxuwang/Projects/T-SAE-Spatial")
    ap.add_argument("--parquet", default="/n/home12/binxuwang/T-SAE-Spatial/imagenet_val_2k.parquet")
    ap.add_argument("--n_images", type=int, default=150)
    ap.add_argument("--n_pcs", type=int, default=256)
    ap.add_argument("--ks", nargs="+", type=int,
                    default=[5, 10, 20, 30, 40, 50, 64, 100, 128, 200, 256])
    ap.add_argument("--conds", nargs="+",
                    default=["dino_b0_postLN", "dino_b1_postLN", "dino_b2_postLN",
                             "dino_b0_nonorm"])
    ap.add_argument("--tol", type=float, default=0.005)
    # Use this for plumbing checks ONLY. It skips the live control, which on a
    # handful of images would fail for a reason that has nothing to do with the
    # code being wrong. Never use it for a run whose numbers get reported.
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--null_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fp = BE.gpu_fingerprint(args.device)
    print(f"[GPU] {fp['gpu_name']} (cc {fp['capability']}, torch {fp['torch']})", flush=True)

    S = Path(args.store)
    spec = []
    for cond in args.conds:
        if cond == "dino_b0_postLN":
            # seed 0 predates the seed sweep and is named differently
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
        arms.append(BE.Arm(f"{cond}/{arm}/s{seed}", cond, arm, seed, d, args.device))
    print(f"[arms] {len(arms)} checkpoints loaded", flush=True)
    BE.check_gpu_match([a.dir for a in arms])

    # Same images, same sample_seed, as the recorded run -- otherwise the live
    # control below is not a control.
    df = pd.read_parquet(args.parquet).sample(args.n_images, random_state=0)
    images = []
    for _, r in df.iterrows():
        b = r["image"]
        b = b["bytes"] if isinstance(b, dict) else b
        try:
            images.append(Image.open(io.BytesIO(b)).convert("RGB"))
        except Exception:
            continue
    print(f"[data] {len(images)} ImageNet images", flush=True)

    # --- pass 1: raw curve + shuffled null, one arm at a time ---------------
    curves, alive_cols, rows = {}, {}, []
    for a in arms:
        key = (a.cond, a.arm, a.seed)
        Z = g0_codes(a, images)
        alive = Z.std(0) > 1e-8
        Za = Z[:, alive]
        c = cum_evr(Za, args.n_pcs)

        rng = np.random.default_rng(args.null_seed)
        Zs = Za.copy()
        for j in range(Zs.shape[1]):
            rng.shuffle(Zs[:, j])
        cn = cum_evr(Zs, args.n_pcs)
        del Zs

        curves[key] = dict(raw=c, null=cn)
        alive_cols[key] = np.flatnonzero(alive)
        rows.append(dict(cond=a.cond, arm=a.arm, seed=a.seed, label=a.label,
                         n_images=len(images), n_tokens=int(Za.shape[0]),
                         n_alive_g0=int(alive.sum()),
                         **{f"evr{k}": float(c[min(k, len(c)) - 1]) for k in args.ks},
                         **{f"null{k}": float(cn[min(k, len(cn)) - 1]) for k in args.ks}))
        print(f"  [raw ] {a.label:34s} alive {int(alive.sum()):5d}  "
              + "  ".join(f"k{k} {c[min(k, len(c)) - 1]:.4f}" for k in (10, 50, 64))
              + f"   null k64 {cn[min(64, len(cn)) - 1]:.4f}", flush=True)
        del Z, Za

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "pca_curve_raw.csv", index=False)

    # --- live control -------------------------------------------------------
    bad = []
    if args.smoke:
        print("[ctrl] !! --smoke: LIVE CONTROL SKIPPED. These numbers are plumbing "
              "output only and must not be reported.", flush=True)
        raw["SMOKE_UNVALIDATED"] = True
    for _, r in (raw.iterrows() if not args.smoke else []):
        rec = RECORDED.get((r["cond"], r["arm"], int(r["seed"])))
        if rec is None:
            print(f"  [ctrl] {r['label']:34s} no recorded value", flush=True)
            continue
        dev = abs(r["evr64"] - rec)
        flag = "OK " if dev <= args.tol else "!! "
        print(f"  [ctrl] {flag}{r['label']:34s} evr64 {r['evr64']:.4f} vs recorded "
              f"{rec:.4f}  dev {dev:.4f}", flush=True)
        if dev > args.tol:
            bad.append((r["label"], r["evr64"], rec, dev))
    assert not bad, ("LIVE CONTROL FAILED -- evr@64 does not reproduce the recorded "
                     f"measurement: {bad}. Do NOT relax --tol to make this pass.")
    if not args.smoke:
        print("[ctrl] live control PASSED on all recorded arms", flush=True)

    # --- pass 2: alive-matched curves, recomputed per (cond, seed) pair -----
    # Rerunning the encoder is cheaper than holding 24 code matrices in RAM.
    mrows = []
    pairs = sorted({(a.cond, a.seed) for a in arms})
    by_key = {(a.cond, a.arm, a.seed): a for a in arms}
    for cond, seed in pairs:
        sp, bl = by_key.get((cond, "spatial", seed)), by_key.get((cond, "baseline", seed))
        if sp is None or bl is None:
            print(f"[match] incomplete pair {cond} s{seed}, skipped", flush=True)
            continue
        m = min(len(alive_cols[(cond, "spatial", seed)]),
                len(alive_cols[(cond, "baseline", seed)]))
        for a in (sp, bl):
            key = (a.cond, a.arm, a.seed)
            cols = alive_cols[key]
            # One rng per PAIR, re-seeded identically for both arms: the two
            # subsamples are the same draw, not two independent ones.
            rng = np.random.default_rng(1000 + seed)
            take = np.sort(rng.choice(len(cols), size=m, replace=False))
            Z = g0_codes(a, images)[:, cols[take]]
            c = cum_evr(Z, args.n_pcs)
            mrows.append(dict(cond=cond, arm=a.arm, seed=seed, label=a.label,
                              n_alive_matched=int(m),
                              **{f"evr{k}": float(c[min(k, len(c)) - 1]) for k in args.ks}))
            print(f"  [match] {a.label:34s} m {m:5d}  "
                  + "  ".join(f"k{k} {c[min(k, len(c)) - 1]:.4f}" for k in (10, 50, 64)),
                  flush=True)
            del Z

    matched = pd.DataFrame(mrows)
    matched.to_csv(out / "pca_curve_alive_matched.csv", index=False)

    # --- paired spatial-minus-baseline at every k ---------------------------
    def paired(dfin, tag):
        recs = []
        for cond, g in dfin.groupby("cond"):
            s = g[g.arm == "spatial"].set_index("seed").sort_index()
            b = g[g.arm == "baseline"].set_index("seed").sort_index()
            seeds = sorted(set(s.index) & set(b.index))
            if len(seeds) < 2:
                continue
            for k in args.ks:
                col = f"evr{k}"
                d = (s.loc[seeds, col] - b.loc[seeds, col]).to_numpy()
                t, p = stats.ttest_rel(s.loc[seeds, col], b.loc[seeds, col])
                recs.append(dict(variant=tag, cond=cond, k=k, n_seeds=len(seeds),
                                 spatial=float(s.loc[seeds, col].mean()),
                                 baseline=float(b.loc[seeds, col].mean()),
                                 diff=float(d.mean()),
                                 diff_sd=float(d.std(ddof=1)),
                                 all_positive=bool((d > 0).all()),
                                 t=float(t), p=float(p)))
        return pd.DataFrame(recs)

    # The shuffled null is ARM-SPECIFIC: it keeps each arm's own marginals, and
    # those differ between arms (different sparsity, different activation scale).
    # So a spatial-minus-baseline gap is only evidence of extra cross-latent
    # structure if it EXCEEDS the gap between the two arms' nulls. `null` tests
    # the nulls against each other; `excess` tests (observed minus own null),
    # which is the quantity that actually answers the question.
    nul = raw.copy()
    for k in args.ks:
        nul[f"evr{k}"] = nul[f"null{k}"]
    exc = raw.copy()
    for k in args.ks:
        exc[f"evr{k}"] = raw[f"evr{k}"] - raw[f"null{k}"]

    pr = pd.concat([paired(raw, "raw"), paired(matched, "alive_matched"),
                    paired(nul, "shuffled_null"), paired(exc, "excess_over_null")],
                   ignore_index=True)
    pr.to_csv(out / "pca_curve_paired.csv", index=False)

    print("\n=== spatial minus baseline, cumulative EVR of the G0 code ===")
    print("(3 paired seeds per cond; p from a paired t-test on 3 points -- "
          "underpowered, read `all_positive` alongside it)")
    for tag in ("raw", "alive_matched", "shuffled_null", "excess_over_null"):
        sub = pr[pr.variant == tag]
        if len(sub):
            print(f"\n-- {tag} --")
            print(sub.drop(columns=["variant"]).to_string(index=False))

    json.dump(dict(n_images=len(images), n_pcs=args.n_pcs, ks=args.ks,
                   conds=args.conds, tol=args.tol, null_seed=args.null_seed,
                   gpu=fp, label="query_paired_seed0_2_only"),
              open(out / "run_args.json", "w"), indent=2)

    # Full curves, for a cutoff-free plot rather than a table of cutoffs.
    np.savez_compressed(out / "pca_curves.npz",
                        **{f"{k[0]}|{k[1]}|s{k[2]}|{w}": v[w]
                           for k, v in curves.items() for w in ("raw", "null")})
    print(f"\n[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
