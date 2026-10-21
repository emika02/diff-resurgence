#!/usr/bin/env python
"""Neutral-loss sharpness around the erased and the original model.

Question: does erasure move the model to a sharper spot? Push the weights in random
directions orthogonal to the edit and measure how much the ordinary denoising loss on
neutral (concept-free) data goes up, for the erased and the original model, using the
*same* random directions for both.

Conditions (y = random noise orthogonal to the edit, same norm as the edit):
    erased_noise(eps)  base + delta + eps*y                    sharpness around the erased model
    base_noise(eps)    base + eps*y                            sharpness around the original model
    orthogonal(eps)    base + sqrt(1-eps^2)*delta + eps*y      rotation, as in run_stability.py (eps <= 1)
    scale(alpha)       base + alpha*delta                      neutral-loss cost of the edit itself
  with --random-edits K (control: is the erasure special, or would any edit of this size do?):
    random_edit_r<k>   base + R_k                              a random edit: same weights, same norm as delta
    random_noise_r<k>(eps)  base + R_k + eps*y                 sharpness around that random edit
  (R_k is random, orthogonal to delta and as large as delta, like y but from its own seeds)

Everything is evaluated on one fixed batch: the same images, captions, timesteps and added
noise for every condition, so loss differences are paired and small changes are measurable.

Outputs
    <out>/loss_results.csv   one row per condition and timestep bin, plus an 'all' row
    <out>/losses/*.npy       per-sample losses, shape [n_images, n_bins], for your own analysis
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from PIL import Image

from edit import EditApplier, compute_edit, load_erased_state_dict

HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = [
    "name", "component", "condition", "level", "noise_seed", "noise_mode", "edit_coef", "noise_coef",
    "reference", "t_bin", "t_lo", "t_hi", "loss_mean", "delta_mean", "delta_se", "n_samples", "edit_norm",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("models")
    g.add_argument("--base", default="CompVis/stable-diffusion-v1-4")
    g.add_argument("--erased", required=True)
    g.add_argument("--component", default="unet", choices=["unet", "text_encoder"])
    g.add_argument("--name", required=True)

    g = p.add_argument_group("neutral data")
    g.add_argument("--neutral-data", default=None,
                   help="CSV with columns 'image' (path, relative to the CSV is fine) and 'caption'. "
                        "Real images (e.g. COCO) are preferred. If omitted, images are generated once "
                        "by the original model from --neutral-prompts and cached in <out>/neutral_data.")
    g.add_argument("--neutral-prompts", default=os.path.join(HERE, "prompts", "neutral.txt"))
    g.add_argument("--n-images", type=int, default=128)
    g.add_argument("--gen-steps", type=int, default=25, help="sampling steps when generating neutral data")
    g.add_argument("--resolution", type=int, default=512)
    g.add_argument("--data-seed", type=int, default=0, help="seed for data generation, timesteps and added noise")

    g = p.add_argument_group("conditions")
    g.add_argument("--conditions", nargs="+", default=["erased_noise", "base_noise", "orthogonal", "scale"],
                   choices=["erased_noise", "base_noise", "orthogonal", "scale"])
    g.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0],
                   help="noise size as a multiple of the edit's norm (values > 1 skipped for 'orthogonal')")
    g.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="extra edit strengths for 'scale'; 0 and 1 are always computed as references")
    g.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--noise-mode", default="per_tensor", choices=["per_tensor", "global"])
    g.add_argument("--random-edits", type=int, default=0,
                   help="number of random edits (same tensors, same norm as the erasure edit) to use as a "
                        "control; each adds 1 + len(eps) * len(noise-seeds) conditions")

    g = p.add_argument_group("evaluation")
    g.add_argument("--t-bins", type=int, default=10, help="timestep bins; one timestep per image per bin")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    g.add_argument("--device", default="cuda")
    g.add_argument("--out", required=True)
    return p.parse_args()


# --------------------------------------------------------------------------- data


def _load_image(path: str, res: int) -> torch.Tensor:
    """Resize the short side to `res`, centre-crop, scale to [-1, 1]."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = res / min(w, h)
    im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - res) // 2, (h - res) // 2
    im = im.crop((left, top, left + res, top + res))
    return torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1) / 127.5 - 1.0


def get_neutral_data(pipe, applier, args):
    """Returns (image paths, captions)."""
    if args.neutral_data:
        import pandas as pd

        df = pd.read_csv(args.neutral_data)
        img_col = "image" if "image" in df.columns else "file_name"
        root = os.path.dirname(os.path.abspath(args.neutral_data))
        paths = [p if os.path.isabs(p) else os.path.join(root, p) for p in df[img_col].astype(str)]
        caps = df["caption"].astype(str).tolist()
        return paths[:args.n_images], caps[:args.n_images]

    cache = os.path.join(args.out, "neutral_data")
    index = os.path.join(cache, "captions.csv")
    if os.path.exists(index):
        with open(index) as f:
            rows = list(csv.DictReader(f))
        if len(rows) >= args.n_images:
            rows = rows[:args.n_images]
            return [os.path.join(cache, r["image"]) for r in rows], [r["caption"] for r in rows]

    print(f"[data] generating {args.n_images} neutral images with the original model (cached for reruns)")
    with open(args.neutral_prompts) as f:
        prompts = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    os.makedirs(cache, exist_ok=True)
    applier.apply(0.0, 0.0)  # original weights
    caps = [prompts[i % len(prompts)] for i in range(args.n_images)]
    names = []
    for k in range(0, args.n_images, args.batch_size):
        idx = list(range(k, min(k + args.batch_size, args.n_images)))
        gens = [torch.Generator(device=args.device).manual_seed(args.data_seed + i) for i in idx]
        with torch.no_grad():
            ims = pipe(prompt=[caps[i] for i in idx], generator=gens, num_inference_steps=args.gen_steps,
                       height=args.resolution, width=args.resolution).images
        for i, im in zip(idx, ims):
            names.append(f"{i:05d}.png")
            im.save(os.path.join(cache, names[-1]))
    with open(index, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption"])
        w.writerows(zip(names, caps))
    return [os.path.join(cache, n) for n in names], caps


class FixedBatch:
    """Latents, token ids, timesteps and added noise, fixed once and reused for every condition."""

    def __init__(self, pipe, paths, captions, args):
        dev, dtype = args.device, pipe.unet.dtype
        n, B = len(paths), args.t_bins
        g = torch.Generator().manual_seed(args.data_seed + 12345)

        lat = []
        with torch.no_grad():
            for k in range(0, n, 8):
                x = torch.stack([_load_image(p, args.resolution) for p in paths[k:k + 8]]).to(dev, pipe.vae.dtype)
                lat.append((pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor).float().cpu())
        self.latents = torch.cat(lat)

        tok = pipe.tokenizer(captions, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt")
        self.input_ids = tok.input_ids

        T = pipe.scheduler.config.num_train_timesteps
        edges = np.linspace(0, T, B + 1).round().astype(int)
        self.bin_edges = edges
        lo = torch.tensor(edges[:-1]).expand(n, B)
        width = torch.tensor(edges[1:] - edges[:-1]).expand(n, B)
        self.timesteps = (lo + (torch.rand(n, B, generator=g) * width).long().clamp(max=width - 1)).long()
        self.noise = torch.randn((n, B) + tuple(self.latents.shape[1:]), generator=g)
        self.n, self.B, self.dtype = n, B, dtype


@torch.no_grad()
def encode_text(pipe, fixed, args):
    out = []
    for k in range(0, fixed.n, 32):
        out.append(pipe.text_encoder(fixed.input_ids[k:k + 32].to(args.device))[0])
    return torch.cat(out)


@torch.no_grad()
def denoising_losses(pipe, sched, fixed, text_emb, args) -> np.ndarray:
    """Per-sample, per-bin denoising MSE, shape [n_images, n_bins]."""
    pairs = [(i, b) for i in range(fixed.n) for b in range(fixed.B)]
    out = np.zeros((fixed.n, fixed.B), dtype=np.float64)
    pred_type = pipe.scheduler.config.get("prediction_type", "epsilon")
    for k in range(0, len(pairs), args.batch_size):
        ii = torch.tensor([p[0] for p in pairs[k:k + args.batch_size]])
        bb = torch.tensor([p[1] for p in pairs[k:k + args.batch_size]])
        lat = fixed.latents[ii].to(args.device, fixed.dtype)
        noise = fixed.noise[ii, bb].to(args.device, fixed.dtype)
        t = fixed.timesteps[ii, bb].to(args.device)
        noisy = sched.add_noise(lat, noise, t)
        pred = pipe.unet(noisy, t, encoder_hidden_states=text_emb[ii.to(args.device)]).sample
        target = noise if pred_type == "epsilon" else sched.get_velocity(lat, noise, t)
        loss = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3)).double().cpu().numpy()
        out[ii.numpy(), bb.numpy()] = loss
    return out


# --------------------------------------------------------------------------- main


def build_conditions(args):
    """List of dicts: condition, level, a (edit), b (noise), seed, r (random edit), r_seed,
    ref (whose losses the deltas are measured against), provides (reference key this condition
    defines, or None). References come first."""
    def c(cond, level, a=0.0, b=0.0, seed=-1, r=0.0, r_seed=0, ref="base", provides=None):
        return dict(cond=cond, level=float(level), a=a, b=b, seed=seed, r=r, r_seed=r_seed,
                    ref=ref, provides=provides)

    conds = [c("scale", 0.0, provides="base"), c("scale", 1.0, a=1.0, provides="erased")]
    for k in range(args.random_edits):
        conds.append(c(f"random_edit_r{k}", 1.0, r=1.0, r_seed=k, provides=f"random_r{k}"))
    eps = sorted(e for e in args.eps if e > 0)
    if "scale" in args.conditions:
        conds += [c("scale", a, a=a) for a in sorted(args.alphas) if a not in (0.0, 1.0)]
    for e in eps:
        for s_ in args.noise_seeds:
            if "erased_noise" in args.conditions:
                conds.append(c("erased_noise", e, a=1.0, b=e, seed=s_, ref="erased"))
            if "base_noise" in args.conditions:
                conds.append(c("base_noise", e, b=e, seed=s_, ref="base"))
            for k in range(args.random_edits):
                conds.append(c(f"random_noise_r{k}", e, b=e, seed=s_, r=1.0, r_seed=k, ref=f"random_r{k}"))
            if "orthogonal" in args.conditions and e <= 1:
                conds.append(c("orthogonal", e, a=math.sqrt(max(0.0, 1 - e * e)), b=e, seed=s_, ref="erased"))
    return conds


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out, "losses"), exist_ok=True)
    csv_path = os.path.join(args.out, "loss_results.csv")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
    )
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    module = getattr(pipe, args.component)
    erased_sd = load_erased_state_dict(args.erased, args.component, type(module))
    edit = compute_edit(module, erased_sd)
    del erased_sd
    print(edit.summary())

    pipe.to(args.device, dtype=getattr(torch, args.dtype))
    pipe.set_progress_bar_config(disable=True)
    applier = EditApplier(module, edit, args.device, args.noise_mode)
    applier.apply(1.0, 0.0)
    err = applier.reconstruction_error()
    print(f"[check] edit reconstruction error in {args.dtype}: {err:.2e}")
    if err > 1e-2:
        print("[check] WARNING: more than 1% of the edit is lost to rounding; use --dtype float32")

    paths, caps = get_neutral_data(pipe, applier, args)
    applier.apply(0.0, 0.0)
    fixed = FixedBatch(pipe, paths, caps, args)
    print(f"[data] {fixed.n} images x {fixed.B} timestep bins = {fixed.n * fixed.B} evaluations per condition")
    text_cache = encode_text(pipe, fixed, args) if args.component == "unet" else None

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done = {(r["condition"], round(float(r["level"]), 6), int(r["noise_seed"]))
                    for r in csv.DictReader(f) if r["t_bin"] == "all"}
    write_header = not os.path.exists(csv_path)

    refs = {}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for cd in build_conditions(args):
            cond, level, a, b, seed, ref = cd["cond"], cd["level"], cd["a"], cd["b"], cd["seed"], cd["ref"]
            key = (cond, round(level, 6), seed)
            npy = os.path.join(args.out, "losses", f"{cond}_{level:g}_s{seed}.npy")

            def evaluate():
                applier.apply(a, b, seed=max(seed, 0), r=cd["r"], r_seed=cd["r_seed"])
                emb = text_cache if text_cache is not None else encode_text(pipe, fixed, args)
                return denoising_losses(pipe, sched, fixed, emb, args)

            if key in done:
                if cd["provides"]:  # references are needed for the deltas of the remaining conditions
                    if os.path.exists(npy):
                        L = np.load(npy)
                    else:
                        L = evaluate()
                        np.save(npy, L)
                    refs[cd["provides"]] = L
                continue

            t0 = time.time()
            L = evaluate()
            np.save(npy, L)
            if cd["provides"]:
                refs[cd["provides"]] = L

            R = refs[ref]
            common = {"name": args.name, "component": args.component, "condition": cond, "level": level,
                      "noise_seed": seed, "noise_mode": args.noise_mode, "edit_coef": a, "noise_coef": b,
                      "reference": ref, "n_samples": fixed.n, "edit_norm": edit.norm}
            for bi in range(fixed.B):
                d = L[:, bi] - R[:, bi]
                writer.writerow({**common, "t_bin": bi, "t_lo": int(fixed.bin_edges[bi]),
                                 "t_hi": int(fixed.bin_edges[bi + 1]), "loss_mean": L[:, bi].mean(),
                                 "delta_mean": d.mean(), "delta_se": d.std(ddof=1) / math.sqrt(fixed.n)})
            d = L.mean(1) - R.mean(1)
            writer.writerow({**common, "t_bin": "all", "t_lo": 0, "t_hi": int(fixed.bin_edges[-1]),
                             "loss_mean": L.mean(), "delta_mean": d.mean(),
                             "delta_se": d.std(ddof=1) / math.sqrt(fixed.n)})
            f.flush()
            print(f"[{cond:>16s} {level:<5g} seed {seed:>2d}] loss {L.mean():.5f}  "
                  f"change vs {ref}: {d.mean():+.2e} ± {d.std(ddof=1) / math.sqrt(fixed.n):.1e}  "
                  f"({time.time() - t0:.0f}s)")

    applier.apply(0.0, 0.0)


if __name__ == "__main__":
    main()