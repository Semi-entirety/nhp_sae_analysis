"""
Calibrate the two shortcut knobs on the toy BEFORE spending any training.

Wendy (Aug 26 2026): add energy and low-rank participation into the toy one at a
time, and see whether spatial still sorts smooth-vs-rough better than baseline.

That experiment only means something if the injected shortcut is comparable in
STRENGTH to the real one, so this script picks the knob settings by measurement
rather than by guess -- the same way `token_scale_calib.py` picked --input_scale.

Targets, from `results/shortcut_analysis/shortcut_raw.csv` (DINOv2 b0 post-LN):
    rho(contiguity, energy)                = +0.401
    rho(contiguity, low-rank participation) = +0.783
Toy control, from `results/toy_shortcut/` (job 41988315):
    +0.059 / +0.072 (d768), variance ratio 1.01x, PCA-50 EVR 0.23.

The measurement is IMPORTED from toy_shortcut_check.py, which in turn imports
`autocorr_per_feature` from dino_smoothness.py and copies shortcut_analysis.py's
formulas -- so the calibration, the toy control and the real-DINO numbers are all
the same code path. That is the only reason the numbers can be compared.

Also reported per setting, because both knobs can damage the toy in ways that
would confound the training result:
  * dictionary coherence (the low-rank knob crowds features together, which hurts
    identifiability on its own -- nothing to do with routing);
  * the CROSS term, i.e. how much the energy knob moves low-rank and vice versa.
    These two shortcuts are correlated in real DINO too, so perfect separation is
    not expected; the achieved pair is reported instead of being claimed.
No training, no checkpoints, pure measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

THIS = Path(__file__).resolve()
for p in [THIS.parent, THIS.parent.parent,
          THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dino_smoothness import autocorr_per_feature
from synth_topk_gen import TopKConfig, TopKSpatialSource
from toy_shortcut_check import _rho, dist_stats, maps_from_tokens

REAL = dict(energy=0.401, lowrank=0.783, var_ratio=8.3)


@torch.no_grad()
def measure(cfg, n_img, n_pcs, device):
    src = TopKSpatialSource(cfg, device=device)
    X = src.sample_images(n_img)
    side = cfg.side
    ac, _ = autocorr_per_feature(maps_from_tokens(X, side).to(device), lags=(1,))
    contig = ac[1]
    Xt = X.reshape(-1, X.shape[-1])
    energy = Xt.var(0).cpu().numpy()
    ok = np.isfinite(contig)

    q = np.nanquantile(contig[ok], [1 / 3, 2 / 3])
    lo = float(energy[contig <= q[0]].mean())
    hi = float(energy[contig >= q[1]].mean())

    Xn = Xt.cpu().numpy()
    p = PCA(n_components=n_pcs, random_state=0).fit(Xn)
    rec = p.inverse_transform(p.transform(Xn))
    lowrank = 1 - ((Xn - rec).var(0) / (energy + 1e-12))

    mu, mean_mu = src.coherence()
    gid = src.group_id.cpu().numpy()
    row = dict(rho_energy=_rho(contig[ok], energy[ok])["rho"],
               rho_lowrank=_rho(contig[ok], lowrank[ok])["rho"],
               var_ratio=hi / (lo + 1e-12),
               evr=float(p.explained_variance_ratio_.sum()),
               dict_coh_max=mu, dict_coh_mean=mean_mu,
               token_l2_med=float(Xt.norm(dim=1).median()),
               **dist_stats(contig))
    # per-group energy actually delivered, the thing the L2 loss weighs
    dnorm = src.dictionary.norm(dim=1).cpu().numpy()
    for gi, nm in enumerate(cfg.group_names):
        row[f"rownorm_{nm}"] = float(dnorm[gid == gi].mean())
    del X, Xt, Xn
    torch.cuda.empty_cache()
    return row


def base_cfg(d_true, seed):
    return dict(d_model=768, side=16, d_true=d_true, k=8,
                corr_lengths=[0.0, 4.0], group_names=["l=0", "l=4"], seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_true", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--gains", nargs="+", type=float,
                    default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--mixes", nargs="+", type=float,
                    default=[0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])
    ap.add_argument("--lowrank_dims", nargs="+", type=int, default=[50, 128, 256])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    # ---- invariance check: the knobs OFF must reproduce the shipped toy -------
    b = base_cfg(a.d_true, a.seed)
    plain = TopKSpatialSource(TopKConfig(**b), device=a.device)
    saved = Path("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
                 f"Projects/T-SAE-Spatial/synth_topk/spatial_d{a.d_true}_s{a.seed}/ground_truth.npz")
    if saved.exists():
        d = float(np.abs(np.load(saved)["dictionary"] - plain.dictionary.cpu().numpy()).max())
        print(f"[control] knobs OFF vs the SHIPPED ground_truth.npz: max abs diff {d:.3e}")
        print("[control] (must be ~0, else the added code perturbed the RNG stream)")
    else:
        print(f"[control] {saved} missing -- cannot check")

    rows = []
    print(f"\ntarget (DINOv2 b0 post-LN): rho_energy {REAL['energy']:+.3f}  "
          f"rho_lowrank {REAL['lowrank']:+.3f}  var_ratio {REAL['var_ratio']}x\n")
    print(f"{'condition':>22} {'rho_E':>7} {'rho_LR':>7} {'varR':>7} {'EVR50':>7} "
          f"{'coh':>6} {'med':>7} {'spread':>7} {'rough':>6}")

    def run(label, knob, value, **kw):
        cfg = TopKConfig(**b, **kw)
        r = measure(cfg, a.n_images, a.n_pcs, a.device)
        r.update(condition=label, knob=knob, value=value,
                 lowrank_dim=kw.get("lowrank_dim", 0))
        rows.append(r)
        print(f"{label:>22} {r['rho_energy']:+7.3f} {r['rho_lowrank']:+7.3f} "
              f"{r['var_ratio']:7.2f} {r['evr']:7.3f} {r['dict_coh_max']:6.3f} "
              f"{r['lag1_median']:+7.3f} {r['lag1_p90_minus_p10']:7.3f} "
              f"{r['frac_below_0p1']:6.3f}", flush=True)

    # energy knob: gain on the SMOOTH group only (real DINO: contiguous dims have
    # the higher variance), sparsity and code autocorrelation untouched
    for g in a.gains:
        run(f"E gain={g:g}", "gain", g, group_gains=[1.0, g])

    # low-rank knob: pull the smooth rows toward a shared subspace, rows kept at
    # unit norm so per-feature energy does NOT move
    for r_ in a.lowrank_dims:
        for m in a.mixes:
            if m == 0 and r_ != a.lowrank_dims[0]:
                continue                       # mix=0 is the same run for every r
            run(f"L r={r_} a={m:g}", "mix", m, lowrank_mix=[0.0, m], lowrank_dim=r_)

    df = pd.DataFrame(rows)
    df.to_csv(out / "toy_shortcut_calib.csv", index=False)

    print("\n---- closest settings to the real-DINO targets ----")
    e = df[df.knob == "gain"].copy()
    e["err"] = (e.rho_energy - REAL["energy"]).abs()
    le = df[df.knob == "mix"].copy()
    le["err"] = (le.rho_lowrank - REAL["lowrank"]).abs()
    for nm, d_ in (("energy", e), ("lowrank", le)):
        best = d_.nsmallest(3, "err")[["condition", "rho_energy", "rho_lowrank",
                                       "var_ratio", "evr", "dict_coh_max"]]
        print(f"\n{nm}:"); print(best.to_string(index=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
