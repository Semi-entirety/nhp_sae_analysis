"""The feature pipeline a checkpoint was trained on, rebuilt from the checkpoint.

WHY THIS MODULE EXISTS
----------------------
Every task script in this repo (accuracy_compare_2, sanity_check_hard,
analyze_transformation_invariance, sparsity_sweep_*, class_seg_l1, fit_pca_block)
carries its own copy-pasted `BlockDINO` that hardcodes

    dinov2_vitb14 . get_intermediate_layers(..., norm=True)      # ImageNet stats @224

That was the only convention that existed until job 41434687, which added two
more: DINOv2 block 0 with NO norm (and a global `input_scale`), and SigLIP2
block 0 raw (different weights, different preprocessing, its own input_scale).

An SAE trained on un-normed, input-scaled tokens will happily `encode()` normed,
unscaled tokens and return NOISE rather than raising. Nothing in the shapes
catches it -- every arm is 768-d. So the pipeline is never taken from the command
line. It is read from the checkpoint's own run_args.json and built from the
TRAINER'S OWN extractor classes, exactly as backbone_eval.py does.

USAGE (drop-in for the old BlockDINO + load_tensor pair)

    from arm_pipeline import pipeline_from_ckpt
    P = pipeline_from_ckpt(args.spatial_ckpt, device)     # .pt file or its dir
    tok = P.tokens_from_bytes(img_bytes)                  # [n_patch, D], scaled
    P.selftest(ae)                                        # fails loudly on mismatch

Old checkpoints predate every one of these run_args keys and their defaults ARE
the old behaviour, so `pipeline_from_ckpt` on a pre-existing checkpoint
reproduces `BlockDINO(norm=True)` + ImageNet-stats @224 exactly.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path

_CACHE = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache"
os.environ.setdefault("TORCH_HOME", _CACHE)
# SigLIP2 weights were pre-fetched to hf_cache; ADE20K lives in hf_datasets_cache.
# home is at 93% -- nothing may fall back to ~/.cache. Callers that also load a
# DATASET must set HF_HOME back to the model cache after importing this module.
_HF_MODELS = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/hf_cache"
os.environ["HF_HOME"] = _HF_MODELS

import torch
from PIL import Image

torch.hub.set_dir(_CACHE + "/hub")
THIS = Path(__file__).resolve()
for _p in [THIS.parent, THIS.parent.parent,
           THIS.parent.parent / "temporal-saes" / "dictionary_learning"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from channel_transform import transform_from_run_args  # noqa: E402
from new_spatial_train_triplet import (  # noqa: E402
    BlockAwareDINOFeatureExtractor,
    SigLIP2FeatureExtractor,
    build_transform,
)

DINO_REPO = _CACHE + "/hub/facebookresearch_dinov2_main"

# One backbone per distinct convention, shared by every arm that uses it --
# 18 arms would otherwise instantiate 18 copies of the same ViT.
_EXTRACTORS: dict = {}


def _resolve(ckpt) -> Path:
    """Accept either .../ae_latest.pt or the run directory containing it."""
    p = Path(ckpt)
    return p.parent if p.is_file() or p.suffix == ".pt" else p


def read_run_args(ckpt) -> dict:
    f = _resolve(ckpt) / "run_args.json"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found -- the feature pipeline cannot be recovered for this "
            f"checkpoint, and guessing it is exactly the failure this module exists "
            f"to prevent.")
    return json.loads(f.read_text())


class Pipeline:
    """Backbone + preprocessing + input scale, as one checkpoint saw them."""

    def __init__(self, ckpt, device):
        self.dir = _resolve(ckpt)
        a = read_run_args(self.dir)
        self.run_args = a
        # Defaults ARE the pre-job-41434687 behaviour; do not change them.
        self.backbone = a.get("backbone", "dinov2")
        self.norm = bool(a.get("backbone_norm", True))
        self.input_scale = float(a.get("input_scale", 1.0))
        self.image_size = int(a.get("image_size", 224))
        # A saved per-channel transform, verified against the sha256 recorded at
        # training time. Default "none" -> every pre-existing checkpoint is
        # unaffected. Built on CPU and moved per call: one Pipeline may be used
        # from several devices, and 768x768 float32 is 2.3 MB.
        self.channel = transform_from_run_args(a)
        self.channel_transform = self.channel.mode
        # mode alone is not enough: two arms could both be "diag" off DIFFERENT
        # stats files, which is exactly the unpairable case assert_same_pipeline exists for.
        self.channel_stats_sha = self.channel.sha
        self.block = (a.get("siglip_block", 0) if self.backbone == "siglip2"
                      else a.get("dino_block", 0))
        self.device = device
        self.T = build_transform(self.backbone, self.image_size)

        key = (self.backbone, self.block, self.norm, self.image_size)
        if key not in _EXTRACTORS:
            if self.backbone == "siglip2":
                _EXTRACTORS[key] = SigLIP2FeatureExtractor(
                    a.get("siglip_id", "google/siglip2-base-patch16-256"),
                    device, block=self.block, norm=self.norm)
            else:
                _EXTRACTORS[key] = BlockAwareDINOFeatureExtractor(
                    a.get("dino_model", "dinov2_vitb14"), DINO_REPO,
                    device, dino_block=self.block, norm=self.norm)
        self.ex = _EXTRACTORS[key]

    # ── naming ────────────────────────────────────────────────────────────
    @property
    def tag(self) -> str:
        n = "postLN" if self.norm else "raw"
        w = "" if self.channel_transform == "none" else f"_{self.channel_transform}"
        return f"{self.backbone}_b{self.block}_{n}{w}"

    def describe(self) -> str:
        return (f"{self.tag}  image_size={self.image_size}  norm={self.norm}  "
                f"input_scale={self.input_scale:.6g}  {self.channel.describe()}  "
                f"({self.dir.name})")

    # ── feature extraction ────────────────────────────────────────────────
    @torch.no_grad()
    def tokens(self, x) -> torch.Tensor:
        """[n_patch, D] on the checkpoint's own convention, input_scale included.

        Accepts a PIL image, raw image bytes, or an already-preprocessed tensor.
        Anything that is not already a tensor goes through THIS backbone's
        transform -- SigLIP2 is mean=std=0.5 @256, DINOv2 is ImageNet stats @224,
        and each is part of its model, not a confound to be equalised.
        """
        if isinstance(x, (bytes, bytearray)):
            x = Image.open(io.BytesIO(x)).convert("RGB")
        if isinstance(x, Image.Image):
            x = self.T(x)
        return self.patch_tokens(x)

    def tokens_from_bytes(self, b) -> torch.Tensor:
        return self.tokens(b)

    # ── drop-in replacements for the copy-pasted BlockDINO / load_tensor ──
    # Every task script does exactly `dino.patch_tokens(load_tensor(bytes, size))`,
    # so a Pipeline can substitute for BOTH with a two-line edit at the call site:
    #     P = pipeline_from_ckpt(args.arm_ckpt, args.device)
    #     dino, load_tensor, args.image_size = P, P.load_tensor, P.image_size

    @torch.no_grad()
    def patch_tokens(self, tensor) -> torch.Tensor:
        """Drop-in for BlockDINO.patch_tokens -- input_scale applied here, once."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        t = self.ex.patch_tokens(tensor)
        if t.ndim == 3:
            t = t.squeeze(0)
        # Same order as the trainer's buffer: channel transform, then input_scale.
        return self.channel.apply(t) * self.input_scale

    @torch.no_grad()
    def patch_tokens_batch(self, images) -> torch.Tensor:
        """[B, n_patch, D], batch dim ALWAYS kept, on CPU.

        Drop-in for sanity_check_hard's `BlockAwareDINO.forward`, which is called
        on a DataLoader batch and returns CPU tensors. `patch_tokens` above cannot
        serve here: it squeezes a leading 1, so a final batch of size 1 would
        silently come back as [n_patch, D] and be concatenated along the wrong
        axis.
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        t = self.ex.patch_tokens(images)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return (self.channel.apply(t) * self.input_scale).detach().cpu()

    def load_tensor(self, img_bytes, image_size=None) -> torch.Tensor:
        """Drop-in for the module-level load_tensor(img_bytes, image_size).

        `image_size` is accepted for signature compatibility and deliberately
        ignored: the size is a property of the checkpoint, not of the caller.
        """
        return self.preprocess(img_bytes)

    def preprocess(self, img) -> torch.Tensor:
        """[1, 3, S, S] -- for callers that must transform and extract separately
        (e.g. transformation invariance, which augments the PIL image first)."""
        if isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img)).convert("RGB")
        return self.T(img).unsqueeze(0)

    @property
    def normalize(self):
        """Just the Normalize step of this arm's transform.

        For callers that must BUILD their own transform pipelines (the
        transformation-invariance task rotates/pads/crops before ToTensor) and
        therefore cannot reuse `self.T` wholesale. Pulled out of the Compose
        rather than re-declared, so it cannot drift from the trainer's.
        """
        for t in self.T.transforms:
            if t.__class__.__name__ == "Normalize":
                return t
        raise RuntimeError(f"no Normalize in {self.tag}'s transform")

    def side(self, n_patch: int) -> int:
        s = int(round(math.sqrt(n_patch)))
        assert s * s == n_patch, f"{n_patch} patches is not a square grid"
        return s

    @property
    def patch_size(self) -> int:
        """DINOv2 is 224/14 and SigLIP2 is 256/16 -- both 16x16, but the patch
        size must follow the arm's own image size, never a constant."""
        return 14 if self.backbone == "dinov2" else 16

    # ── the check that makes a wrong convention fatal ─────────────────────
    @torch.no_grad()
    def selftest(self, ae, images, max_fvu: float = 0.95, label: str = "") -> float:
        """An arm fed the wrong feature convention returns noise, not an error.

        `images` MUST be a class-DIVERSE sample. FVU divides by the token
        variance, and this repo's probe parquet is class-clustered (the file is
        literally `...-cluster.parquet`), so `df.head(8)` is eight images of one
        class. That shrinks the denominator without touching the numerator and
        inflates FVU: measured 0.903 on head(8) versus 0.554 on a random 8 for
        the same no-norm checkpoint. Use `df.sample(...)`, not `df.head(...)`.
        """
        toks = torch.cat([self.tokens(im) for im in images], 0)
        xhat = ae.decode(ae.encode(toks, use_threshold=False))
        # paper convention (dictionary_learning/training.py:57-60), mean-subtracted
        fvu = float((toks - xhat).var(0).sum() / toks.var(0).sum())
        assert fvu < max_fvu, (
            f"{label or self.tag}: FVU {fvu:.3f} -- this SAE is not reconstructing "
            f"the features it is being fed. Either the pipeline does not match "
            f"training, or `images` ({len(images)} of them) is not class-diverse "
            f"enough to give a meaningful variance denominator (see docstring).")
        return fvu


def pipeline_from_ckpt(ckpt, device) -> Pipeline:
    return Pipeline(ckpt, device)


def assert_same_pipeline(*pipes) -> Pipeline:
    """Arms compared in one paired test must see identical features."""
    ref = pipes[0]
    for p in pipes[1:]:
        for f in ("backbone", "block", "norm", "input_scale", "image_size",
                  "channel_transform", "channel_stats_sha"):
            assert getattr(ref, f) == getattr(p, f), (
                f"{ref.dir.name} and {p.dir.name} disagree on {f}: "
                f"{getattr(ref, f)} vs {getattr(p, f)} -- these two checkpoints "
                f"cannot be compared in one paired run.")
    return ref
