"""
Train the spatial SAE on the synthetic toy generative process.

Mirrors new_spatial_train_triplet.py exactly -- same trainer, same triplet
[B, 3, D] interface, same hyperparameters -- but the activations come from
SyntheticSpatialSource instead of DINOv2, so the ground-truth dictionary is
known and recovery can be measured directly.

The ground-truth dictionary is saved next to the checkpoints as
ground_truth.npz for analyze_synth_recovery.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in [SCRIPTS_DIR, REPO_ROOT, REPO_ROOT / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dictionary_learning.trainers.temporal_sequence_top_k_weight import TemporalMatryoshkaBatchTopKTrainer
from new_spatial_train_triplet import (
    save_latest_checkpoint, compute_diagnostics, append_metrics_csv,
)
from synth_spatial_gen import SynthConfig, SyntheticSpatialSource


class SyntheticTripletBuffer:
    """Yield [B, 3, D] = anchor, local-positive, batch-negative from synthetic data.

    Same sampling semantics as SpatialPatchTripletBuffer: the positive is a
    patch within Chebyshev distance <= radius of the anchor (excluding the
    anchor), the negative is any other patch in the current image batch outside
    the anchor's positive set.
    """

    def __init__(self, source: SyntheticSpatialSource, n_img: int, pairs_per_image: int,
                 radius: int = 2, device="cuda", seed: int = 0):
        self.src = source
        self.n_img = int(n_img)
        self.pairs = int(pairs_per_image)
        self.radius = int(radius)
        self.device = device
        self.side = source.cfg.side
        self.P = self.side * self.side
        self.g = torch.Generator(device=device).manual_seed(seed + 12345)

        # all Chebyshev offsets within radius, excluding (0,0)
        offs = [(dr, dc)
                for dr in range(-radius, radius + 1)
                for dc in range(-radius, radius + 1)
                if not (dr == 0 and dc == 0)]
        self.offs = torch.tensor(offs, device=device, dtype=torch.long)  # [O, 2]

    def _sample_positive(self, r, c):
        """Uniform over IN-BOUNDS neighbours, via rejection (matches the real buffer)."""
        shape = r.shape
        rr = torch.empty_like(r); cc = torch.empty_like(c)
        need = torch.ones(shape, dtype=torch.bool, device=self.device)
        for _ in range(16):
            if not need.any():
                break
            pick = torch.randint(len(self.offs), shape, generator=self.g, device=self.device)
            cand_r = r + self.offs[pick, 0]
            cand_c = c + self.offs[pick, 1]
            ok = (cand_r >= 0) & (cand_r < self.side) & (cand_c >= 0) & (cand_c < self.side)
            take = ok & need
            rr[take] = cand_r[take]; cc[take] = cand_c[take]
            need = need & ~ok
        if need.any():   # extremely rare fallback
            rr[need] = r[need].clamp(0, self.side - 1)
            cc[need] = c[need].clamp(0, self.side - 1)
        return rr, cc

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            X = self.src.sample_images(self.n_img)              # [I, P, D]
            I, P, D = X.shape
            shape = (I, self.pairs)

            a_idx = torch.randint(P, shape, generator=self.g, device=self.device)
            r, c = a_idx // self.side, a_idx % self.side
            pr, pc = self._sample_positive(r, c)
            p_idx = pr * self.side + pc

            # negative: any patch in the batch outside the anchor's positive region
            n_img_idx = torch.randint(I, shape, generator=self.g, device=self.device)
            n_pat_idx = torch.randint(P, shape, generator=self.g, device=self.device)
            img_ar = torch.arange(I, device=self.device).view(I, 1).expand(shape)
            for _ in range(8):
                nr, nc = n_pat_idx // self.side, n_pat_idx % self.side
                same_img = n_img_idx == img_ar
                near = (nr - r).abs().le(self.radius) & (nc - c).abs().le(self.radius)
                bad = same_img & near
                if not bad.any():
                    break
                n_img_idx = torch.where(bad, torch.randint(I, shape, generator=self.g, device=self.device), n_img_idx)
                n_pat_idx = torch.where(bad, torch.randint(P, shape, generator=self.g, device=self.device), n_pat_idx)

            anchor = X[img_ar.reshape(-1), a_idx.reshape(-1)]
            pos    = X[img_ar.reshape(-1), p_idx.reshape(-1)]
            neg    = X[n_img_idx.reshape(-1), n_pat_idx.reshape(-1)]
            yield torch.stack([anchor, pos, neg], dim=1).contiguous()   # [B, 3, D]


def parse_args():
    ap = argparse.ArgumentParser()
    # generative process
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--side", type=int, default=16)
    ap.add_argument("--k_smooth", type=int, default=32)
    ap.add_argument("--k_rough", type=int, default=32)
    ap.add_argument("--gp_sigma", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.15)
    ap.add_argument("--no_sparsify", dest="sparsify", action="store_false", default=True,
                    help="Keep coefficients dense signed Gaussians (identifiability control).")
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--data_seed", type=int, default=0)
    # SAE hyperparameters -- defaults are the swept best config
    ap.add_argument("--dict_size", type=int, default=16384)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--group_fractions", type=float, nargs="+", default=[0.2, 0.8])
    ap.add_argument("--group_weights", type=float, nargs="+", default=None)
    ap.add_argument("--temp_alpha", type=float, default=100.0)
    ap.add_argument("--auxk_alpha", type=float, default=1 / 32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--recon_weight", type=float, default=1.0)
    ap.add_argument("--contrastive_weight", type=float, default=1.0)
    ap.add_argument("--neighbor_weight", type=float, default=1.0)
    ap.add_argument("--normalize_contrastive", action="store_true", default=True)
    ap.add_argument("--no_normalize_contrastive", dest="normalize_contrastive", action="store_false")
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--decay_start", type=int, default=None)
    ap.add_argument("--threshold_beta", type=float, default=0.999)
    ap.add_argument("--threshold_start_step", type=int, default=1000)
    # loop
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size_images", type=int, default=8)
    ap.add_argument("--pairs_per_image", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    cfg = SynthConfig(d_model=args.d_model, side=args.side, k_smooth=args.k_smooth,
                      k_rough=args.k_rough, gp_sigma=args.gp_sigma, tau=args.tau,
                      sparsify=args.sparsify, noise_std=args.noise_std, seed=args.data_seed)
    src = SyntheticSpatialSource(cfg, device=args.device)

    # ground truth for the recovery analysis
    np.savez(save_dir / "ground_truth.npz",
             V=src.V.cpu().numpy(), U=src.U.cpu().numpy(),
             dictionary=src.dictionary.cpu().numpy(),
             is_smooth=src.is_smooth.cpu().numpy(), **{"cfg_json": json.dumps(cfg.to_json())})

    buffer = SyntheticTripletBuffer(src, n_img=args.batch_size_images,
                                    pairs_per_image=args.pairs_per_image,
                                    device=args.device, seed=args.seed)
    it = iter(buffer)
    first = next(it)
    activation_dim = first.shape[-1]
    print(f"[Data] triplet batch {tuple(first.shape)}  activation_dim={activation_dim}", flush=True)

    trainer = TemporalMatryoshkaBatchTopKTrainer(
        steps=args.steps, activation_dim=activation_dim, dict_size=args.dict_size,
        k=args.k, temporal=True, contrastive=True, layer=0, lm_name="synthetic",
        group_fractions=args.group_fractions, group_weights=args.group_weights,
        lr=args.lr, auxk_alpha=args.auxk_alpha, temp_alpha=args.temp_alpha,
        warmup_steps=args.warmup_steps, decay_start=args.decay_start,
        threshold_beta=args.threshold_beta, threshold_start_step=args.threshold_start_step,
        seed=args.seed, device=args.device, wandb_name="SynthSpatialSAE",
        submodule_name="synthetic", neighbor_recon=False,
        normalize_contrastive=args.normalize_contrastive,
        recon_weight=args.recon_weight, contrastive_weight=args.contrastive_weight,
        neighbor_weight=args.neighbor_weight,
    )
    print(f"[Config] dict_size={args.dict_size} k={args.k} gf={args.group_fractions} "
          f"temp_alpha={args.temp_alpha} contrastive_weight={args.contrastive_weight}", flush=True)

    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args) | {"activation_dim": activation_dim}, f, indent=2)

    loss_csv = save_dir / "loss_log.csv"
    fieldnames_ref = {}
    step = 0
    while step < args.steps:
        x = next(it)
        loss = trainer.update(step, x)
        if step % args.log_every == 0:
            row = {"step": step, **compute_diagnostics(trainer, x, loss)}
            append_metrics_csv(loss_csv, row, fieldnames_ref)
            print(f"[step {step:>6}] " + " ".join(
                f"{k}={v:.4f}" for k, v in row.items() if k != "step" and isinstance(v, float)),
                flush=True)
        if step > 0 and step % args.save_every == 0:
            save_latest_checkpoint(trainer, save_dir, step, args, activation_dim)
        step += 1

    save_latest_checkpoint(trainer, save_dir, max(step - 1, 0), args, activation_dim)
    print(f"[Done] {save_dir}", flush=True)


if __name__ == "__main__":
    main()
