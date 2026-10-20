from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Make imports robust whether this script is launched from repo root, scripts/, or elsewhere.
THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in [SCRIPTS_DIR, REPO_ROOT, REPO_ROOT / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from new_vision_patch_pairs import SpatialPatchPairBuffer  # noqa: E402
from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKTrainer


def collect_image_paths(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise ValueError(f"No images found under: {root}")
    return paths


def save_checkpoint(trainer: TemporalMatryoshkaBatchTopKTrainer, save_dir: Path, step: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "ae_state_dict": trainer.ae.state_dict(),
        "trainer_config": trainer.config,
    }
    torch.save(ckpt, save_dir / f"checkpoint_step_{step}.pt")
    torch.save(trainer.ae.state_dict(), save_dir / "ae_latest.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Spatial/Temporal SAE on DINOv2 patch pairs.")

    # Data. Use --data_path for your parquet file, or --image_root for a folder of images.
    parser.add_argument("--data_path", type=str, default=None, help="Parquet file containing an image column.")
    parser.add_argument("--image_root", type=str, default=None, help="Folder of images. Alternative to --data_path.")
    parser.add_argument("--image_column", type=str, default="image")

    # DINOv2 local torch.hub settings.
    parser.add_argument("--dino_model", "--model_name", dest="dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument(
        "--dino_repo_path",
        type=str,
        default="/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
        help="Local DINOv2 torch.hub repo/cache path.",
    )
    parser.add_argument("--image_size", type=int, default=224)

    # Patch pair sampling.
    parser.add_argument("--batch_size", type=int, default=512, help="Approximate patch-pair batch size.")
    parser.add_argument("--batch_size_images", type=int, default=None)
    parser.add_argument("--pairs_per_image", type=int, default=64)
    parser.add_argument("--neighbor_mode", type=str, default="8", choices=["4", "8"])
    parser.add_argument("--num_workers", type=int, default=4)

    # SAE/trainer settings.
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--dict_size", type=int, default=65536)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--group_fractions", type=float, nargs="+", default=[0.25, 0.25, 0.25, 0.25])
    parser.add_argument("--group_weights", type=float, nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--auxk_alpha", type=float, default=1 / 32)
    parser.add_argument("--temp_alpha", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--decay_start", type=int, default=None)
    parser.add_argument("--threshold_beta", type=float, default=0.999)
    parser.add_argument("--threshold_start_step", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    # Metadata expected by original trainer.
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--lm_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--wandb_name", type=str, default="SpatialTemporalMatryoshkaBatchTopKSAE")
    parser.add_argument("--submodule_name", type=str, default="x_norm_patchtokens")
    parser.add_argument("--dino_block", type=int, default=None,
                        help="If set, hook this intermediate DINOv2 block (1-indexed) instead of final output.")
    parser.add_argument("--no_contrastive", action="store_true",
                        help="Disable NT-Xent contrastive loss (trains a standard SAE, not spatial SAE).")

    # Experiment flags.
    parser.add_argument("--neighbor_recon", action="store_true",
                        help="Add neighbor patch reconstruction loss (Experiment 1)")
    parser.add_argument("--normalize_contrastive", action="store_true",
                        help="L2-normalize + temperature-scale contrastive logits (Experiment 2)")
    parser.add_argument("--init_from_ckpt", type=str, default=None,
                        help="Path to ae state_dict to warm-start from (e.g. v4 ae_final.pt)")

    # Runtime/checkpoints.
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="results/spatial_sae_run")
    parser.add_argument("--save_every", type=int, default=1000)

    args = parser.parse_args()
    if args.data_path is None and args.image_root is None:
        parser.error("Provide either --data_path <parquet> or --image_root <folder>.")
    if args.data_path is not None and args.image_root is not None:
        parser.error("Provide only one of --data_path or --image_root, not both.")
    return args


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size_images is None:
        args.batch_size_images = max(1, args.batch_size // max(1, args.pairs_per_image))

    image_paths: Optional[List[str]] = None
    parquet_path: Optional[str] = None
    if args.data_path is not None:
        parquet_path = args.data_path
    else:
        image_paths = collect_image_paths(args.image_root)

    buffer = SpatialPatchPairBuffer(
        image_paths=image_paths,
        parquet_path=parquet_path,
        image_column=args.image_column,
        dino_model_name=args.dino_model,
        dino_repo_path=args.dino_repo_path,
        batch_size_images=args.batch_size_images,
        image_size=args.image_size,
        pairs_per_image=args.pairs_per_image,
        neighbor_mode=args.neighbor_mode,
        device=args.device,
        shuffle=True,
        num_workers=args.num_workers,
        dino_block=args.dino_block,
    )

    # Infer activation_dim from the real DINOv2 patch tokens.
    first_batch = next(iter(buffer))
    activation_dim = first_batch.shape[-1]
    print(f"Inferred activation_dim={activation_dim}; first patch-pair batch shape={tuple(first_batch.shape)}")

    trainer = TemporalMatryoshkaBatchTopKTrainer(
        steps=args.steps,
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        temporal=True,
        contrastive=not args.no_contrastive,
        layer=args.layer,
        lm_name=args.lm_name,
        group_fractions=args.group_fractions,
        group_weights=args.group_weights,
        lr=args.lr,
        auxk_alpha=args.auxk_alpha,
        temp_alpha=args.temp_alpha,
        warmup_steps=args.warmup_steps,
        decay_start=args.decay_start,
        threshold_beta=args.threshold_beta,
        threshold_start_step=args.threshold_start_step,
        seed=args.seed,
        device=args.device,
        wandb_name=args.wandb_name,
        submodule_name=args.submodule_name,
        neighbor_recon=args.neighbor_recon,
        normalize_contrastive=args.normalize_contrastive,
    )

    if args.init_from_ckpt is not None:
        print(f"[Init] Loading weights from {args.init_from_ckpt}")
        state = torch.load(args.init_from_ckpt, map_location=args.device)
        trainer.ae.load_state_dict(state)
        print("[Init] Weights loaded.")

    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args) | {"activation_dim": activation_dim}, f, indent=2)

    step = 0
    while step < args.steps:
        for x in buffer:
            loss = trainer.update(step, x)
            if step % 10 == 0:
                print(f"[step {step}] loss={loss:.6f}")
            step += 1
            if step >= args.steps:
                break

    final_ckpt = save_dir / "ae_final.pt"
    torch.save(trainer.ae.state_dict(), final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main()
