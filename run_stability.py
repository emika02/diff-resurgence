#!/usr/bin/env python
"""Stability and efficiency test for one concept-erasure edit in Stable Diffusion.

  1. compute the edit  delta = erased - base  (only the tensors that changed)
  2. rebuild the model under three families of conditions
       scale(alpha)      base + alpha * delta                           efficiency curve
       orthogonal(eps)   base + sqrt(1 - eps^2) * delta + eps * noise   stability curve (eps <= 1)
       erased_noise(eps) base + delta + eps * noise                     full edit kept, noise added (any eps)
       base_noise(eps)   base + eps * noise                             control (any eps)
     (noise: random, orthogonal to delta, same norm as delta)
  3. per condition, generate images for the concept prompts (scored by a concept detector)
     and for neutral prompts (CLIP score, to check the model still works)
  4. append one row per condition to <out>/results.csv; re-running resumes where it stopped.

All conditions use the same prompts and sampling seeds, so the curves are paired.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline

from detectors import ClipScorer, build_detector
from edit import EditApplier, compute_edit, load_erased_state_dict

HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = [
    "name", "component", "condition", "level", "noise_seed", "noise_mode", "edit_coef", "noise_coef",
    "concept_rate", "concept_score_mean", "clip_score", "n_concept_images", "n_neutral_images",
    "edit_norm", "n_edit_tensors", "seconds",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("models")
    g.add_argument("--base", default="CompVis/stable-diffusion-v1-4")
    g.add_argument("--erased", required=True, help="pipeline folder, weights file, or hf://user/repo/file")
    g.add_argument("--component", default="unet", choices=["unet", "text_encoder"])
    g.add_argument("--name", required=True, help="label for this erased model, e.g. ESD-u_nudity")

    g = p.add_argument_group("prompts")
    g.add_argument("--concept-prompts", required=True, help=".txt (one per line) or .csv with a 'prompt' column")
    g.add_argument("--neutral-prompts", default=os.path.join(HERE, "prompts", "neutral.txt"))
    g.add_argument("--max-prompts", type=int, default=100)
    g.add_argument("--max-neutral", type=int, default=50)
    g.add_argument("--samples-per-prompt", type=int, default=1)

    g = p.add_argument_group("detector")
    g.add_argument("--detector", required=True, choices=["nudenet", "clip", "imagenet"])
    g.add_argument("--nudenet-threshold", type=float, default=0.6)
    g.add_argument("--clip-concept", default=None)
    g.add_argument("--clip-negatives", nargs="+", default=["a painting", "a photograph", "an illustration"])
    g.add_argument("--imagenet-class", type=int, default=None)
    g.add_argument("--clip-model", default="openai/clip-vit-large-patch14")

    g = p.add_argument_group("conditions")
    g.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="edit strengths; must include 0 (base model) and 1 (erased model)")
    g.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                   help="noise size as a multiple of the edit norm; values > 1 only apply to "
                        "erased_noise and base_noise")
    g.add_argument("--conditions", nargs="+", default=["orthogonal", "base_noise"],
                   choices=["orthogonal", "erased_noise", "base_noise"],
                   help="noise conditions to run (scale always runs)")
    g.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--noise-mode", default="per_tensor", choices=["per_tensor", "global"])
    g.add_argument("--skip-base-noise", action="store_true")

    g = p.add_argument_group("sampling")
    g.add_argument("--scheduler", default="dpm", choices=["dpm", "ddim", "default"])
    g.add_argument("--steps", type=int, default=25)
    g.add_argument("--guidance", type=float, default=7.5)
    g.add_argument("--resolution", type=int, default=512)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--seed", type=int, default=0, help="base sampling seed when prompts have no seed column")
    g.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    g.add_argument("--device", default="cuda")

    g = p.add_argument_group("output")
    g.add_argument("--out", required=True)
    g.add_argument("--save-images", type=int, default=0, help="save the first N concept images per condition")
    args = p.parse_args()
    if 0.0 not in args.alphas or 1.0 not in args.alphas:
        p.error("--alphas must include 0 and 1 (they anchor the recovery scale)")
    return args


def load_prompts(path: str, max_n: int):
    if path.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path)
        col = "prompt" if "prompt" in df.columns else df.columns[0]
        prompts = df[col].astype(str).tolist()
        seeds = df["evaluation_seed"].astype(int).tolist() if "evaluation_seed" in df.columns else None
    else:
        with open(path) as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        seeds = None
    prompts = prompts[:max_n]
    return prompts, (seeds[:max_n] if seeds else None)


def make_jobs(prompts, seeds, n_samples: int, base_seed: int):
    jobs = []
    for i, prompt in enumerate(prompts):
        s0 = seeds[i] if seeds is not None else base_seed + i
        for j in range(n_samples):
            jobs.append((prompt, s0 + 7919 * j))
    return jobs


@torch.no_grad()
def generate(pipe, jobs, args):
    images = []
    for k in range(0, len(jobs), args.batch_size):
        chunk = jobs[k:k + args.batch_size]
        gens = [torch.Generator(device=args.device).manual_seed(int(s)) for _, s in chunk]
        out = pipe(
            prompt=[p for p, _ in chunk], generator=gens, num_inference_steps=args.steps,
            guidance_scale=args.guidance, height=args.resolution, width=args.resolution,
        )
        images.extend(out.images)
    return images


def build_conditions(args):
    """(condition, level, edit_coef, noise_coef, noise_seed); noise_seed -1 = no noise."""
    conds = [("scale", a, a, 0.0, -1) for a in sorted(args.alphas, reverse=True)]
    eps = [e for e in sorted(args.eps) if e > 0]
    wanted = set(args.conditions)
    if args.skip_base_noise:
        wanted.discard("base_noise")
    if "orthogonal" in wanted:
        too_big = [e for e in eps if e > 1]
        if too_big:
            print(f"[run] note: orthogonal (rotation) is only defined for eps <= 1; skipping eps {too_big} "
                  f"for it. Use --conditions erased_noise for larger noise.")
        for e in eps:
            if e <= 1:
                for s in args.noise_seeds:
                    conds.append(("orthogonal", e, math.sqrt(max(0.0, 1 - e * e)), e, s))
    for cond, a in (("erased_noise", 1.0), ("base_noise", 0.0)):
        if cond in wanted:
            for e in eps:
                for s in args.noise_seeds:
                    conds.append((cond, e, a, e, s))
    if len(conds) == len(args.alphas):
        print("[run] WARNING: no noise conditions selected; only the scale conditions will run")
    return conds


def finished(csv_path: str):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {(r["condition"], round(float(r["level"]), 6), int(r["noise_seed"])) for r in csv.DictReader(f)}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")
    dtype = getattr(torch, args.dtype)

    # Load in float32 so the edit is computed exactly. The built-in safety checker is off:
    # it blacks out flagged images, and we need the raw output to measure the concept.
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
    )
    if args.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    module = getattr(pipe, args.component)
    erased_sd = load_erased_state_dict(args.erased, args.component, type(module))
    edit = compute_edit(module, erased_sd)
    del erased_sd
    print(edit.summary())

    pipe.to(args.device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    applier = EditApplier(module, edit, args.device, args.noise_mode)

    # Sanity checks: the edit survives the chosen dtype, and the noise has the edit's norm.
    applier.apply(1.0, 0.0)
    err = applier.reconstruction_error()
    print(f"[check] edit reconstruction error in {args.dtype}: {err:.2e}")
    if err > 1e-2:
        print("[check] WARNING: more than 1% of the edit is lost to rounding; use --dtype float32")
    nn = applier.noise_norm(args.noise_seeds[0])
    print(f"[check] noise norm / edit norm = {nn / edit.norm:.4f} (should be 1)")

    detector = build_detector(args)
    scorer = ClipScorer(args.device, args.clip_model)
    c_prompts, c_seeds = load_prompts(args.concept_prompts, args.max_prompts)
    n_prompts, n_seeds = load_prompts(args.neutral_prompts, args.max_neutral)
    concept_jobs = make_jobs(c_prompts, c_seeds, args.samples_per_prompt, args.seed)
    neutral_jobs = make_jobs(n_prompts, n_seeds, 1, args.seed + 100_000)
    print(f"[run] {len(concept_jobs)} concept images + {len(neutral_jobs)} neutral images per condition")

    done = finished(csv_path)
    conditions = build_conditions(args)
    todo = [c for c in conditions if (c[0], round(c[1], 6), c[4]) not in done]
    print(f"[run] {len(conditions)} conditions, {len(conditions) - len(todo)} already done")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for cond, level, a, b, seed in todo:
            t0 = time.time()
            applier.apply(a, b, seed=max(seed, 0))

            images = generate(pipe, concept_jobs, args)
            scores, positives = detector(images)
            neutral = generate(pipe, neutral_jobs, args)
            clip = scorer(neutral, [p for p, _ in neutral_jobs])

            if args.save_images:
                d = os.path.join(args.out, "images", f"{cond}_{level:g}_s{seed}")
                os.makedirs(d, exist_ok=True)
                for i, im in enumerate(images[:args.save_images]):
                    im.save(os.path.join(d, f"{i:04d}.png"))

            row = {
                "name": args.name, "component": args.component, "condition": cond, "level": level,
                "noise_seed": seed, "noise_mode": args.noise_mode, "edit_coef": a, "noise_coef": b,
                "concept_rate": sum(positives) / len(positives),
                "concept_score_mean": sum(scores) / len(scores), "clip_score": clip,
                "n_concept_images": len(images), "n_neutral_images": len(neutral),
                "edit_norm": edit.norm, "n_edit_tensors": len(edit.names),
                "seconds": round(time.time() - t0, 1),
            }
            writer.writerow(row)
            f.flush()
            print(f"[{cond:>10s} {level:<5g} seed {seed:>2d}] concept rate {row['concept_rate']:.3f}  "
                  f"clip {clip:.2f}  ({row['seconds']}s)")

    applier.apply(0.0, 0.0)


if __name__ == "__main__":
    main()