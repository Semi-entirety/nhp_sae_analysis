"""
Synthetic generative process for the spatial-SAE toy case.

    V(i,j) = sum_k c_k(i,j) v_k  +  sum_k d_k(i,j) u_k

  v_k  ("smooth" directions)  carry coefficients c_k with strong spatial
       autocorrelation across token positions (Gaussian-filtered white noise).
  u_k  ("rough" directions)   carry coefficients d_k drawn i.i.d. per token,
       so they have ~zero autocorrelation between neighbouring tokens.

Design choice that makes the comparison clean
---------------------------------------------
Both coefficient families are standardised to zero mean / unit variance BEFORE
the sparsifying rectification, and use the SAME threshold tau. Therefore c and
d have identical marginal distributions, identical expected sparsity and
identical energy -- the ONLY thing that differs is spatial autocorrelation.
Without this, "smooth vs rough" would be confounded with magnitude or density
and any recovery difference would be uninterpretable.

Sparsity is applied because a BatchTopK SAE assumes a sparse non-negative code;
with dense Gaussian coefficients every ground-truth feature is active on every
token and the recovery question is ill-posed for reasons unrelated to the
spatial design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SynthConfig:
    d_model: int = 768        # matches DINOv2 ViT-B/14 token dim
    side: int = 16            # 16x16 = 256 tokens, matches 224px / patch14
    k_smooth: int = 32        # number of v_k
    k_rough: int = 32         # number of u_k
    gp_sigma: float = 2.0     # Gaussian-filter width, in tokens, for c_k
    tau: float = 1.15         # rectification threshold; P(z>tau) ~ 0.125
    sparsify: bool = True     # if False, coefficients stay dense signed Gaussians
    noise_std: float = 0.0    # optional isotropic observation noise
    orthogonal: bool = False  # if True, make v/u exactly orthonormal
    seed: int = 0

    def to_json(self):
        return asdict(self)


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    rad = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-rad, rad + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g, rad


def smooth_fields(n: int, side: int, sigma: float, gen: torch.Generator, device) -> torch.Tensor:
    """[n, side, side] Gaussian-filtered white noise, circular padding (stationary).

    Normalised ANALYTICALLY, not per-field: for unit-variance white noise passed
    through a separable kernel g, the output std is exactly ||g||^2. Using the
    per-field sample std instead would be a noisy estimate on a correlated field
    (few effective degrees of freedom), which distorts the marginal and makes the
    smooth fields differ from the rough ones in density/energy as well as in
    autocorrelation -- exactly the confound this design is trying to avoid.

    Uses exact circular convolution via FFT, so there is no limit on sigma
    relative to the grid (F.pad(mode="circular") refuses to wrap more than once,
    which caps sigma at ~side/6 and blocks wide-kernel configurations).
    """
    w = torch.randn(n, side, side, generator=gen, device=device)
    ax = torch.arange(side, device=device, dtype=torch.float32)
    dist = torch.minimum(ax, side - ax)               # circular distance from 0
    g1 = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    K = g1[:, None] * g1[None, :]                     # separable, sums to 1
    out = torch.fft.irfft2(torch.fft.rfft2(w) * torch.fft.rfft2(K), s=(side, side))
    return out / (K.pow(2).sum().sqrt() + 1e-12)      # exact unit variance


class SyntheticSpatialSource:
    """Generates batches of synthetic 'images' of token activations."""

    def __init__(self, cfg: SynthConfig, device="cuda"):
        self.cfg = cfg
        self.device = device
        gen = torch.Generator(device=device).manual_seed(cfg.seed)
        self.gen = gen

        K = cfg.k_smooth + cfg.k_rough
        if cfg.orthogonal:
            A = torch.randn(cfg.d_model, K, generator=gen, device=device)
            Q, _ = torch.linalg.qr(A)
            Dmat = Q[:, :K].T.contiguous()
        else:
            Dmat = torch.randn(K, cfg.d_model, generator=gen, device=device)
            Dmat = F.normalize(Dmat, dim=-1)
        self.V = Dmat[:cfg.k_smooth]          # [k_smooth, d_model]  smooth dirs
        self.U = Dmat[cfg.k_smooth:]          # [k_rough,  d_model]  rough dirs
        self.dictionary = Dmat                # [K, d_model]
        self.is_smooth = torch.zeros(K, dtype=torch.bool, device=device)
        self.is_smooth[:cfg.k_smooth] = True

    def sample_coeffs(self, n_img: int):
        """Return (C, Dd) each [n_img, P, k_*] non-negative sparse coefficients."""
        cfg = self.cfg
        P = cfg.side * cfg.side

        # both families are exactly N(0,1) marginally by construction
        c = smooth_fields(n_img * cfg.k_smooth, cfg.side, cfg.gp_sigma, self.gen, self.device)
        c = c.view(n_img, cfg.k_smooth, cfg.side, cfg.side)

        d = torch.randn(n_img, cfg.k_rough, cfg.side, cfg.side,
                        generator=self.gen, device=self.device)

        # identical rectification for both -> identical marginals, only
        # the spatial autocorrelation differs.
        # With sparsify=False the coefficients stay dense signed Gaussians. That
        # is the control case: a Gaussian latent distribution is rotationally
        # symmetric inside its own subspace, so the individual axes v_k / u_k
        # are NOT identifiable -- only the subspace they span is. Sparsity (or
        # any other non-Gaussianity) is what makes the dictionary recoverable.
        if cfg.sparsify:
            c = F.relu(c - cfg.tau)
            d = F.relu(d - cfg.tau)

        C = c.view(n_img, cfg.k_smooth, P).permute(0, 2, 1).contiguous()
        Dd = d.view(n_img, cfg.k_rough, P).permute(0, 2, 1).contiguous()
        return C, Dd

    def sample_images(self, n_img: int, return_coeffs: bool = False):
        """Return tokens [n_img, P, d_model]."""
        C, Dd = self.sample_coeffs(n_img)
        X = C @ self.V + Dd @ self.U
        if self.cfg.noise_std > 0:
            X = X + self.cfg.noise_std * torch.randn(
                X.shape, generator=self.gen, device=self.device)
        if return_coeffs:
            return X, C, Dd
        return X


def lag1_autocorr(field: torch.Tensor) -> float:
    """Mean 4-neighbour lag-1 spatial autocorrelation of a [.., side, side] field."""
    x = field - field.mean(dim=(-2, -1), keepdim=True)
    var = (x ** 2).mean(dim=(-2, -1)) + 1e-12
    h = (x[..., :, :-1] * x[..., :, 1:]).mean(dim=(-2, -1))
    v = (x[..., :-1, :] * x[..., 1:, :]).mean(dim=(-2, -1))
    return float((((h + v) / 2) / var).mean())


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Sanity-check the synthetic generator.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gp_sigma", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = SynthConfig(gp_sigma=args.gp_sigma, tau=args.tau)
    src = SyntheticSpatialSource(cfg, device=args.device)
    X, C, Dd = src.sample_images(64, return_coeffs=True)

    side = cfg.side
    cm = C.view(64, side, side, cfg.k_smooth).permute(0, 3, 1, 2)
    dm = Dd.view(64, side, side, cfg.k_rough).permute(0, 3, 1, 2)

    stats = dict(
        token_shape=list(X.shape),
        smooth_density=float((C > 0).float().mean()),
        rough_density=float((Dd > 0).float().mean()),
        smooth_mean_active_per_token=float((C > 0).float().sum(-1).mean()),
        rough_mean_active_per_token=float((Dd > 0).float().sum(-1).mean()),
        smooth_energy=float((C ** 2).sum(-1).mean()),
        rough_energy=float((Dd ** 2).sum(-1).mean()),
        smooth_lag1_autocorr=lag1_autocorr(cm),
        rough_lag1_autocorr=lag1_autocorr(dm),
        dict_max_offdiag_cos=float(
            (src.dictionary @ src.dictionary.T
             - torch.eye(src.dictionary.shape[0], device=args.device)).abs().max()),
    )
    print(json.dumps(stats, indent=2))
    (outdir / "generator_stats.json").write_text(json.dumps(stats | cfg.to_json(), indent=2))

    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
    for j in range(6):
        axes[0, j].imshow(cm[0, j].cpu(), cmap="viridis")
        axes[0, j].set_title(f"$c_{{{j}}}$ smooth", fontsize=9); axes[0, j].axis("off")
        axes[1, j].imshow(dm[0, j].cpu(), cmap="viridis")
        axes[1, j].set_title(f"$d_{{{j}}}$ rough", fontsize=9); axes[1, j].axis("off")
    fig.suptitle(
        f"Coefficient fields, one image (side={side}, gp_sigma={cfg.gp_sigma}, tau={cfg.tau})\n"
        f"identical marginals by construction — density {stats['smooth_density']:.3f} vs "
        f"{stats['rough_density']:.3f}; lag-1 autocorr "
        f"{stats['smooth_lag1_autocorr']:.3f} vs {stats['rough_lag1_autocorr']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "generator_fields.png", dpi=150, bbox_inches="tight")
    print(f"[Saved] {outdir}")
