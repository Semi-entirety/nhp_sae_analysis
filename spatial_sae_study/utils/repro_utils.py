"""
Reproducibility helpers.

Background: torch's CUDA RNG is NOT reproducible across GPU architectures. A run
trained on an A100 and re-analysed on an H100 regenerated a completely different
ground-truth dictionary (max abs difference 0.24, i.e. unrelated). That failure
was caught only because the analysis verified the regenerated dictionary against
a saved copy; without the check it would have looked like total recovery failure
and sent us debugging the model instead of the harness.

Two consequences, both handled here:

  * anything that must match across processes is drawn on the CPU
    (cpu_randn / cpu_generator), then moved to the device;
  * every run records the GPU it ran on (gpu_fingerprint), and analyses check
    that fingerprint and report the result rather than silently continuing.

The subtler hazard is arms that are supposed to be PAIRED. Runs sharing a
data_seed only see identical data if they also ran on the same GPU model, and
Slurm does not guarantee that. Mixed placement would silently break pairing with
no error and no visible symptom, so check_gpu_match is meant to be called by
default, not only when something looks wrong.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


def gpu_fingerprint(device=None) -> dict:
    """Identify the compute device precisely enough to detect an RNG mismatch."""
    out = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        i = 0 if device is None or str(device) == "cuda" else torch.device(device).index or 0
        p = torch.cuda.get_device_properties(i)
        out |= {"gpu_name": torch.cuda.get_device_name(i),
                "capability": f"{p.major}.{p.minor}",
                "cuda_runtime": torch.version.cuda}
    else:
        out |= {"gpu_name": "cpu", "capability": "cpu", "cuda_runtime": None}
    return out


def cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def cpu_randn(*shape, generator: torch.Generator, device="cpu", dtype=torch.float32):
    """Draw on the CPU, then move. Identical bits on any machine."""
    x = torch.randn(*shape, generator=generator, dtype=dtype)
    return x.to(device) if str(device) != "cpu" else x


def check_gpu_match(run_dirs: Sequence[str | Path], verbose: bool = True) -> dict:
    """Compare the GPU fingerprints recorded by a set of runs.

    Returns a report dict. `paired_safe` is True only when every run that should
    be comparable ran on the same GPU model, which is the condition under which a
    shared data_seed actually implies identical data.
    """
    seen, missing = {}, []
    for d in run_dirs:
        f = Path(d) / "run_args.json"
        if not f.exists():
            missing.append(str(d)); continue
        a = json.loads(f.read_text())
        fp = a.get("gpu_fingerprint")
        name = (fp or {}).get("gpu_name", "UNRECORDED")
        seen.setdefault(name, []).append(Path(d).name)
    cur = gpu_fingerprint()
    report = {
        "runs_by_gpu": seen,
        "n_distinct_gpu_models": len(seen),
        "unrecorded": "UNRECORDED" in seen,
        "missing_run_args": missing,
        "analysis_gpu": cur["gpu_name"],
        "analysis_matches_training": (len(seen) == 1 and cur["gpu_name"] in seen),
        "paired_safe": len(seen) == 1 and "UNRECORDED" not in seen,
    }
    if verbose:
        print("=" * 68)
        print("GPU REPRODUCIBILITY CHECK")
        for name, runs in seen.items():
            print(f"  training GPU: {name:28s} ({len(runs)} runs)")
        print(f"  analysis GPU: {cur['gpu_name']}")
        if missing:
            print(f"  !! {len(missing)} run(s) missing run_args.json")
        if report["unrecorded"]:
            print("  !! some runs predate fingerprinting -- GPU unknown, treat pairing as UNVERIFIED")
        if report["n_distinct_gpu_models"] > 1:
            print("  !! RUNS SPAN MULTIPLE GPU MODELS -- a shared data_seed does NOT imply")
            print("     identical data across them. Paired comparisons are INVALID.")
        elif report["paired_safe"]:
            print("  OK: all runs on one GPU model -> shared data_seed implies identical data")
        if not report["analysis_matches_training"] and report["n_distinct_gpu_models"] >= 1:
            print("  NOTE: analysis is on a different GPU than training. Safe ONLY because")
            print("        ground truth is loaded from disk, never regenerated from a seed.")
        print("=" * 68, flush=True)
    return report
