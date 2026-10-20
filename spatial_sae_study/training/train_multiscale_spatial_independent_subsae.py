"""
train_multiscale_spatial_independent_subsae.py
================================================
Train the original distance-based multi-scale Spatial SAE as four independent
sub-SAEs instead of one Matryoshka / joint-loss SAE.

This implements the design:

  SubSAE-G0: reconstruction + contrastive to a far spatial neighbor
  SubSAE-G1: reconstruction + contrastive to a mid-distance spatial neighbor
  SubSAE-G2: reconstruction + contrastive to a local spatial neighbor
  SubSAE-G3: reconstruction only, if group_steps[3] == 0

Each sub-SAE has its own encoder, decoder, optimizer, TopK budget, and loss.
During one training iteration we reuse the same DINO patch-pair batch, but update
G0/G1/G2/G3 separately with separate backward passes and optimizer steps.

This is NOT Matryoshka anymore. It is an independent multi-branch / sub-SAE
version of the previous multi-scale spatial training script.

For compatibility with your previous analysis scripts, the script saves:

  1. sub_saes_final.pt
       A true checkpoint containing the four independent sub-SAE state_dicts.

  2. ae_final.pt
       A combined, analysis-compatible state_dict made by concatenating the
       four sub-SAE dictionaries into one SAE-like checkpoint with group_sizes.
       This is mainly for probing/analysis scripts that expect one SAE state.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in [SCRIPTS_DIR, REPO_ROOT, REPO_ROOT / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from new_vision_patch_pairs import (  # noqa: E402
    DINOFeatureExtractor,
    ImagePathDataset,
    ParquetImageDataset,
)
from dictionary_learning.trainers.temporal_sequence_top_k import (  # noqa: E402
    TemporalMatryoshkaBatchTopKTrainer,
)


# ─── Distance-based neighbor sampling, same as previous multiscale script ─────

def _neighbors_at_distance(h: int, w: int, r: int, c: int, d: int) -> list[tuple[int, int]]:
    out = []
    for dr in range(-d, d + 1):
        dc_abs = d - abs(dr)
        for dc in ([-dc_abs, dc_abs] if dc_abs > 0 else [0]):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _neighbors_within_distance(h: int, w: int, r: int, c: int, d: int) -> list[tuple[int, int]]:
    out = []
    for dist in range(1, d + 1):
        out.extend(_neighbors_at_distance(h, w, r, c, dist))
    return out


class MultiscalePatchPairBuffer:
    """
    Yields x with shape (B, n_slots, D):
      x[:, 0] = anchor patch token
      x[:, 1:] = one sampled neighbor for each group with group_steps[i] > 0

    The neighbor slot order follows the order of contrastive groups.
    Example group_steps=[3,2,1,0] gives slots:
      slot 0: anchor
      slot 1: G0 neighbor at distance 3
      slot 2: G1 neighbor at distance 2
      slot 3: G2 neighbor at distance 1
    """

    def __init__(
        self,
        image_paths: Optional[Sequence[Union[str, Path]]] = None,
        parquet_path: Optional[Union[str, Path]] = None,
        image_column: str = "image",
        dino_model_name: str = "dinov2_vitb14",
        dino_repo_path: Union[str, Path] = "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
        batch_size_images: int = 8,
        image_size: int = 224,
        pairs_per_image: int = 64,
        group_steps: List[int] = (3, 2, 1, 0),
        neighbor_sampling: str = "exact",
        device: str = "cuda",
        shuffle: bool = True,
        num_workers: int = 4,
        dino_block: Optional[int] = None,
    ):
        if parquet_path is None and image_paths is None:
            raise ValueError("Provide either parquet_path or image_paths.")
        if parquet_path is not None and image_paths is not None:
            raise ValueError("Provide only one of parquet_path or image_paths.")

        if parquet_path is not None:
            dataset = ParquetImageDataset(parquet_path, image_size=image_size, image_column=image_column)
        else:
            dataset = ImagePathDataset(image_paths or [], image_size=image_size)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size_images,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
            drop_last=False,
        )
        self.extractor = DINOFeatureExtractor(
            model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            device=device,
            dino_block=dino_block,
        )
        self.pairs_per_image = pairs_per_image
        self.group_steps = list(group_steps)
        self.neighbor_sampling = neighbor_sampling
        self.device = device

        self.contrastive_groups = [(i, s) for i, s in enumerate(group_steps) if s > 0]
        self.group_to_slot = {g: slot_idx + 1 for slot_idx, (g, _) in enumerate(self.contrastive_groups)}
        self.n_slots = 1 + len(self.contrastive_groups)
        print(
            f"[MultiscalePatchPairBuffer] group_steps={group_steps}, "
            f"contrastive_groups={self.contrastive_groups}, n_slots={self.n_slots}"
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        for images in self.loader:
            tokens = self.extractor.patch_tokens(images)  # (B_img, N, D)
            bsz, n_patches, d_model = tokens.shape
            side = int(math.sqrt(n_patches))
            assert side * side == n_patches, f"Expected square patch grid, got {n_patches} patches."
            grid = tokens.view(bsz, side, side, d_model)

            pair_list = []
            for b in range(bsz):
                for _ in range(self.pairs_per_image):
                    r = random.randrange(side)
                    c = random.randrange(side)
                    slots = [grid[b, r, c]]
                    valid = True

                    for _, step_d in self.contrastive_groups:
                        if self.neighbor_sampling == "exact":
                            nbrs = _neighbors_at_distance(side, side, r, c, step_d)
                        else:
                            nbrs = _neighbors_within_distance(side, side, r, c, step_d)
                        if not nbrs:
                            valid = False
                            break
                        rr, cc = random.choice(nbrs)
                        slots.append(grid[b, rr, cc])

                    if valid:
                        pair_list.append(torch.stack(slots, dim=0))

            if pair_list:
                yield torch.stack(pair_list, dim=0).to(self.device)


# ─── Independent sub-SAE trainer ──────────────────────────────────────────────

class IndependentSubSAESpatialTrainer(TemporalMatryoshkaBatchTopKTrainer):
    """
    A single independent sub-SAE.

    It receives x with shape:
      (B, 1, D) if contrastive_step == 0
      (B, 2, D) if contrastive_step > 0, where x[:,0] is anchor and x[:,1] is neighbor.

    It trains only this sub-SAE's parameters with its own optimizer and loss.
    """

    def __init__(self, contrastive_step: int, contrastive_alpha: float, **kwargs):
        kwargs["temporal"] = True
        kwargs["contrastive"] = False
        kwargs["group_fractions"] = [1.0]
        kwargs["group_weights"] = [1.0]
        super().__init__(**kwargs)
        self.contrastive_step = int(contrastive_step)
        self.contrastive_alpha = float(contrastive_alpha)

    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        import torch as t
        from collections import namedtuple

        anchor = x[:, 0]
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            anchor, return_active=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(f)

        # Single-group reconstruction.
        x_hat = f @ self.ae.W_dec + self.ae.b_dec
        l2_loss = (anchor - x_hat).pow(2).sum(dim=-1).mean()

        # Optional spatial contrastive loss for this sub-SAE.
        contrastive_loss = t.tensor(0.0, device=self.device)
        if self.contrastive_step > 0 and x.shape[1] >= 2 and self.contrastive_alpha != 0:
            nbr = x[:, 1]
            f_nbr, _, _ = self.ae.encode(nbr, return_active=True, use_threshold=False)
            logits = f @ f_nbr.T  # intentionally raw dot product, matching previous multiscale script
            labels = t.arange(logits.shape[0], device=self.device, dtype=t.long)
            contrastive_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        auxk_loss = self.get_auxiliary_loss((anchor - x_hat).detach(), post_relu_acts_BF)
        loss = l2_loss + self.auxk_alpha * auxk_loss + self.contrastive_alpha * contrastive_loss

        # Dead-feature tracking, same logic as original trainer override.
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        if not logging:
            return loss
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            anchor,
            x_hat,
            f,
            {
                "l2_loss": float(l2_loss.item()),
                "contrastive_loss": float(contrastive_loss.item()),
                "auxk_loss": float(auxk_loss.item()),
                "loss": float(loss.item()),
            },
        )


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _split_sizes(total: int, fractions: Sequence[float]) -> list[int]:
    raw = [int(round(total * f)) for f in fractions]
    diff = total - sum(raw)
    raw[-1] += diff
    if any(s <= 0 for s in raw):
        raise ValueError(f"Invalid split sizes from total={total}, fractions={fractions}: {raw}")
    return raw


def _default_group_ks(total_k: int, group_sizes: Sequence[int]) -> list[int]:
    total = sum(group_sizes)
    ks = [max(1, int(round(total_k * s / total))) for s in group_sizes]
    diff = total_k - sum(ks)
    ks[-1] += diff
    if any(k <= 0 for k in ks):
        raise ValueError(f"Invalid group ks from total_k={total_k}, group_sizes={group_sizes}: {ks}")
    return ks


def save_subsae_checkpoint(trainers: list[IndependentSubSAESpatialTrainer], path: Path, args: argparse.Namespace) -> None:
    payload = {
        "format": "independent_subsae_v1",
        "args": vars(args),
        "num_groups": len(trainers),
        "sub_saes": [t.ae.state_dict() for t in trainers],
        "group_dict_sizes": args.group_dict_sizes,
        "group_ks": args.group_ks,
        "group_steps": args.group_steps,
    }
    torch.save(payload, path)


def combined_state_dict_for_analysis(
    trainers: list[IndependentSubSAESpatialTrainer],
    group_dict_sizes: Sequence[int],
    total_k: int,
) -> dict:
    """
    Concatenate independent sub-SAE weights into one SAE-like state_dict.

    This is for compatibility with analysis scripts that expect one state_dict with
    W_enc, W_dec, b_enc, b_dec, group_sizes, and k. Because the sub-SAEs are truly
    independent, b_dec is not naturally shared; we use the mean b_dec.
    """
    states = [t.ae.state_dict() for t in trainers]
    out = {}
    first = states[0]

    for key in first.keys():
        vals = [s[key] for s in states if key in s]
        if len(vals) != len(states):
            continue
        if key == "W_enc":
            out[key] = torch.cat(vals, dim=1)
        elif key == "W_dec":
            out[key] = torch.cat(vals, dim=0)
        elif key in {"b_enc", "threshold", "num_tokens_since_fired"}:
            out[key] = torch.cat(vals, dim=0)
        elif key == "b_dec":
            out[key] = torch.stack(vals, dim=0).mean(dim=0)
        elif key == "group_sizes":
            out[key] = torch.tensor(list(group_dict_sizes), device=vals[0].device, dtype=vals[0].dtype)
        elif key == "k":
            out[key] = torch.tensor(total_k, device=vals[0].device, dtype=vals[0].dtype)
        else:
            # Scalars/metadata: keep the first value. This covers keys like active_groups if present.
            out[key] = vals[0]

    # Ensure these metadata keys exist even if not present as buffers in your SAE implementation.
    device = next(iter(first.values())).device
    out["group_sizes"] = torch.tensor(list(group_dict_sizes), device=device, dtype=torch.long)
    out["k"] = torch.tensor(int(total_k), device=device, dtype=torch.long)
    return out


def init_subsaes_from_joint_ckpt(
    trainers: list[IndependentSubSAESpatialTrainer],
    ckpt_path: str,
    group_dict_sizes: Sequence[int],
    device: str,
) -> None:
    """Warm-start independent sub-SAEs from slices of a previous combined SAE checkpoint."""
    joint = torch.load(ckpt_path, map_location=device)
    if isinstance(joint, dict) and "sub_saes" in joint:
        for i, trainer in enumerate(trainers):
            trainer.ae.load_state_dict(joint["sub_saes"][i], strict=False)
        print(f"[Init] Loaded independent sub-SAE checkpoint from {ckpt_path}")
        return

    bounds = [0]
    for s in group_dict_sizes:
        bounds.append(bounds[-1] + int(s))

    for i, trainer in enumerate(trainers):
        st = trainer.ae.state_dict()
        a, b = bounds[i], bounds[i + 1]
        if "W_enc" in joint and "W_enc" in st:
            st["W_enc"].copy_(joint["W_enc"][:, a:b])
        if "W_dec" in joint and "W_dec" in st:
            st["W_dec"].copy_(joint["W_dec"][a:b, :])
        if "b_enc" in joint and "b_enc" in st:
            st["b_enc"].copy_(joint["b_enc"][a:b])
        if "threshold" in joint and "threshold" in st:
            st["threshold"].copy_(joint["threshold"][a:b])
        if "b_dec" in joint and "b_dec" in st:
            st["b_dec"].copy_(joint["b_dec"])
        trainer.ae.load_state_dict(st, strict=False)
    print(f"[Init] Sliced previous joint checkpoint into independent sub-SAEs: {ckpt_path}")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train original distance-based multiscale Spatial SAE as independent sub-SAEs."
    )

    # Data
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--image_root", type=str, default=None)
    p.add_argument("--image_column", type=str, default="image")

    # DINOv2
    p.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    p.add_argument(
        "--dino_repo_path",
        type=str,
        default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache/hub/facebookresearch_dinov2_main",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--dino_block", type=int, default=None)

    # Patch pair sampling
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--batch_size_images", type=int, default=None)
    p.add_argument("--pairs_per_image", type=int, default=64)
    p.add_argument("--neighbor_sampling", type=str, default="exact", choices=["exact", "within"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--group_steps", type=int, nargs="+", default=[3, 2, 1, 0])

    # SAE / trainer defaults
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--dict_size", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--group_fractions", type=float, nargs="+", default=[0.25, 0.25, 0.25, 0.25])
    p.add_argument(
        "--group_ks",
        type=int,
        nargs="+",
        default=None,
        help="TopK budget per independent sub-SAE. Default: split --k proportional to group sizes, e.g. 8 8 8 8 for k=32.",
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--auxk_alpha", type=float, default=1 / 32)
    p.add_argument("--temp_alpha", type=float, default=0.1)
    p.add_argument(
        "--group_temp_alphas",
        type=float,
        nargs="+",
        default=None,
        help="Contrastive alpha per group. Default: temp_alpha for groups with step>0 and 0 for step=0.",
    )
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--decay_start", type=int, default=None)
    p.add_argument("--threshold_beta", type=float, default=0.999)
    p.add_argument("--threshold_start_step", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)

    # Trainer metadata
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--lm_name", type=str, default="dinov2_vitb14")
    p.add_argument("--wandb_name", type=str, default="IndependentMultiscaleSpatialSAE")
    p.add_argument("--submodule_name", type=str, default="x_norm_patchtokens")

    # Checkpoint / runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="results/multiscale_spatial_independent_subsae")
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None,
        help="Warm-start from a previous combined SAE checkpoint, or from this script's sub_saes checkpoint.",
    )

    args = p.parse_args()
    if args.data_path is None and args.image_root is None:
        p.error("Provide either --data_path or --image_root.")
    if args.data_path is not None and args.image_root is not None:
        p.error("Provide only one of --data_path or --image_root.")
    if len(args.group_steps) != len(args.group_fractions):
        p.error("--group_steps length must match --group_fractions length.")
    return args


def collect_image_paths(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise ValueError(f"No images found under: {root}")
    return paths


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.batch_size_images is None:
        args.batch_size_images = max(1, args.batch_size // max(1, args.pairs_per_image))

    args.group_dict_sizes = _split_sizes(args.dict_size, args.group_fractions)
    if args.group_ks is None:
        args.group_ks = _default_group_ks(args.k, args.group_dict_sizes)
    if len(args.group_ks) != len(args.group_dict_sizes):
        raise ValueError("--group_ks length must match number of groups.")

    if args.group_temp_alphas is None:
        args.group_temp_alphas = [args.temp_alpha if s > 0 else 0.0 for s in args.group_steps]
    if len(args.group_temp_alphas) != len(args.group_steps):
        raise ValueError("--group_temp_alphas length must match --group_steps length.")

    image_paths = None
    parquet_path = None
    if args.data_path is not None:
        parquet_path = args.data_path
    else:
        image_paths = collect_image_paths(args.image_root)

    print("Independent sub-SAE multiscale setup:")
    for i, (size, k_i, step_i, alpha_i) in enumerate(
        zip(args.group_dict_sizes, args.group_ks, args.group_steps, args.group_temp_alphas)
    ):
        task = f"recon + contrastive @ dist={step_i}" if step_i > 0 else "reconstruction only"
        print(f"  G{i}: dict_size={size}, k={k_i}, alpha={alpha_i}, task={task}")

    buffer = MultiscalePatchPairBuffer(
        image_paths=image_paths,
        parquet_path=parquet_path,
        image_column=args.image_column,
        dino_model_name=args.dino_model,
        dino_repo_path=args.dino_repo_path,
        batch_size_images=args.batch_size_images,
        image_size=args.image_size,
        pairs_per_image=args.pairs_per_image,
        group_steps=args.group_steps,
        neighbor_sampling=args.neighbor_sampling,
        device=args.device,
        shuffle=True,
        num_workers=args.num_workers,
        dino_block=args.dino_block,
    )

    first_batch = next(iter(buffer))
    activation_dim = first_batch.shape[-1]
    print(f"activation_dim={activation_dim}, first_batch shape={tuple(first_batch.shape)}")

    trainers: list[IndependentSubSAESpatialTrainer] = []
    for i, (dict_i, k_i, step_i, alpha_i) in enumerate(
        zip(args.group_dict_sizes, args.group_ks, args.group_steps, args.group_temp_alphas)
    ):
        trainer = IndependentSubSAESpatialTrainer(
            contrastive_step=step_i,
            contrastive_alpha=alpha_i,
            steps=args.steps,
            activation_dim=activation_dim,
            dict_size=dict_i,
            k=k_i,
            layer=args.layer,
            lm_name=args.lm_name,
            lr=args.lr,
            auxk_alpha=args.auxk_alpha,
            temp_alpha=0.0,  # handled by contrastive_alpha inside this class
            warmup_steps=args.warmup_steps,
            decay_start=args.decay_start,
            threshold_beta=args.threshold_beta,
            threshold_start_step=args.threshold_start_step,
            seed=args.seed + i,
            device=args.device,
            wandb_name=f"{args.wandb_name}_G{i}",
            submodule_name=args.submodule_name,
            neighbor_recon=False,
            normalize_contrastive=False,
        )
        trainers.append(trainer)

    if args.init_from_ckpt is not None:
        init_subsaes_from_joint_ckpt(trainers, args.init_from_ckpt, args.group_dict_sizes, args.device)

    # Save config.
    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args) | {"activation_dim": activation_dim}, f, indent=2)

    # Training loop.
    step = 0
    while step < args.steps:
        for x in buffer:
            loss_items = []
            for g, trainer in enumerate(trainers):
                step_d = args.group_steps[g]
                if step_d > 0:
                    slot = buffer.group_to_slot[g]
                    x_g = torch.stack([x[:, 0], x[:, slot]], dim=1)
                else:
                    x_g = x[:, 0:1]
                loss_val = trainer.update(step, x_g)
                loss_items.append(float(loss_val))

            if step % 10 == 0:
                loss_str = " ".join(f"G{i}={v:.4f}" for i, v in enumerate(loss_items))
                print(f"[step {step}] {loss_str}")

            if step > 0 and step % args.save_every == 0:
                sub_path = save_dir / f"sub_saes_step_{step}.pt"
                save_subsae_checkpoint(trainers, sub_path, args)

                combined = combined_state_dict_for_analysis(trainers, args.group_dict_sizes, args.k)
                combined_path = save_dir / f"checkpoint_step_{step}.pt"
                torch.save(combined, combined_path)
                print(f"Saved checkpoints: {sub_path} and {combined_path}")

            step += 1
            if step >= args.steps:
                break

    sub_final = save_dir / "sub_saes_final.pt"
    save_subsae_checkpoint(trainers, sub_final, args)

    combined_final = save_dir / "ae_final.pt"
    torch.save(combined_state_dict_for_analysis(trainers, args.group_dict_sizes, args.k), combined_final)

    print(f"Saved independent sub-SAE checkpoint: {sub_final}")
    print(f"Saved combined analysis-compatible checkpoint: {combined_final}")
    print("Note: ae_final.pt is a concatenated compatibility checkpoint; sub_saes_final.pt is the true model checkpoint.")


if __name__ == "__main__":
    main()
