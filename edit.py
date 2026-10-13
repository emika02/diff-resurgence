"""Weight-space edit utilities.

An erasure method changes some weights of one model component (usually the
U-Net, sometimes the text encoder). We call that change the edit:

    delta = erased_weights - base_weights

and rebuild the component as

    weights = base + a * delta + b * noise

where `noise` is random, orthogonal to `delta`, and has the same norm as
`delta`. Three settings of (a, b) cover every condition in the experiment:

    scale(alpha)       a = alpha,               b = 0     efficiency curve
    orthogonal(eps)    a = sqrt(1 - eps**2),    b = eps   stability curve (edit norm stays fixed)
    base_noise(eps)    a = 0,                   b = eps   control: same noise on the original model
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

# Prefixes some repos put in front of diffusers-style keys.
_PREFIXES = ("unet.", "text_encoder.", "module.", "model.")


# --------------------------------------------------------------------------- loading


def load_erased_state_dict(spec: str, component: str, component_cls) -> dict:
    """Load the erased weights for one component.

    `spec` can be
      - a diffusers pipeline folder (with a `<component>/` subfolder) or a component folder,
      - a .safetensors / .pt / .bin file holding a full or partial state dict,
      - hf://<user>/<repo>/<path/in/repo> for a single file on the Hugging Face Hub.
    Keys must be in diffusers / transformers format. Original CompVis (LDM) checkpoints
    need converting first (diffusers ships a conversion script for that).
    """
    if spec.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        parts = spec[len("hf://"):].split("/")
        spec = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))

    if os.path.isdir(spec):
        kwargs = {"subfolder": component} if os.path.isdir(os.path.join(spec, component)) else {}
        model = component_cls.from_pretrained(spec, torch_dtype=torch.float32, **kwargs)
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if spec.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(spec, device="cpu")
    else:
        sd = torch.load(spec, map_location="cpu", weights_only=False)

    # Unwrap containers like {"state_dict": {...}} or {"unet": {...}}.
    for _ in range(3):
        inner = None
        if isinstance(sd, dict):
            for key in ("state_dict", component, "model", "module"):
                if isinstance(sd.get(key), dict):
                    inner = sd[key]
                    break
        if inner is None:
            break
        sd = inner
    return sd


# --------------------------------------------------------------------------- the edit


@dataclass
class Edit:
    names: list            # tensor names that changed
    base: dict             # name -> fp32 CPU tensor (original weights)
    delta: dict            # name -> fp32 CPU tensor (erased - original)

    @property
    def norm(self) -> float:
        return math.sqrt(sum(float(d.double().pow(2).sum()) for d in self.delta.values()))

    def summary(self) -> str:
        total_sq = self.norm ** 2
        n_weights = sum(d.numel() for d in self.delta.values())
        groups: dict = {}
        for n in self.names:
            p = n.split(".")
            g = ".".join(p[:2]) if p[0] in ("down_blocks", "up_blocks") else p[0]
            groups[g] = groups.get(g, 0.0) + float(self.delta[n].double().pow(2).sum())
        kinds = sorted({n.split(".")[-2] for n in self.names if len(n.split(".")) > 1})
        lines = [f"edit: {len(self.names)} tensors, {n_weights:,} weights, norm {self.norm:.4f}"]
        for g, sq in sorted(groups.items()):
            lines.append(f"  {g:<24s}{100 * sq / total_sq:6.1f}% of squared norm")
        lines.append("  module types touched: " + ", ".join(kinds[:15]) + (" ..." if len(kinds) > 15 else ""))
        return "\n".join(lines)


def compute_edit(base_module: torch.nn.Module, erased_sd: dict, atol: float = 0.0) -> Edit:
    """Difference between erased and base weights, keeping only tensors that changed.

    `base_module` must still be in float32 when this is called.
    """
    base_sd = base_module.state_dict()
    matched, unmatched = {}, []
    for k, v in erased_sd.items():
        if not torch.is_tensor(v):
            continue
        key = k
        if key not in base_sd:
            for p in _PREFIXES:
                if key.startswith(p) and key[len(p):] in base_sd:
                    key = key[len(p):]
                    break
        if key in base_sd:
            matched[key] = v
        else:
            unmatched.append(k)

    if not matched:
        raise ValueError(
            f"No keys of the erased checkpoint match the base {type(base_module).__name__}. "
            f"First checkpoint keys: {list(erased_sd)[:5]}. If this is a CompVis/LDM "
            f"checkpoint, convert it to diffusers format first."
        )
    if unmatched:
        print(f"[edit] warning: {len(unmatched)} checkpoint keys not found in base, e.g. {unmatched[:3]}")

    names, base, delta = [], {}, {}
    for k, v in matched.items():
        b = base_sd[k]
        if not torch.is_floating_point(b):
            continue
        if v.shape != b.shape:
            raise ValueError(f"shape mismatch for {k}: erased {tuple(v.shape)} vs base {tuple(b.shape)}")
        b32 = b.detach().float().cpu()
        d = v.detach().float().cpu() - b32
        if d.abs().max().item() > atol:
            names.append(k)
            base[k] = b32.clone()
            delta[k] = d
    if not names:
        raise ValueError("Erased weights are identical to the base weights (wrong component or wrong file?).")
    return Edit(names, base, delta)


# --------------------------------------------------------------------------- applying


def _orthogonal_same_norm(y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Project y onto the complement of d and rescale it to ||d||."""
    yf, df = y.reshape(-1).double(), d.reshape(-1).double()
    dd = torch.dot(df, df)
    yf = yf - (torch.dot(yf, df) / dd) * df
    n = yf.norm()
    if yf.numel() < 2 or n < 1e-12:
        return torch.zeros_like(y)
    return (yf * (dd.sqrt() / n)).to(y.dtype).reshape(y.shape)


class EditApplier:
    """Writes base + a*delta + b*noise into the live model, tensor by tensor.

    noise_mode
      per_tensor  noise is orthogonal to the edit and has the edit's norm *in every tensor*
                  (keeps the layer-by-layer distribution of the edit; default)
      global      noise is orthogonal to the whole edit and has its total norm
                  (closest to the paper's single-adapter setup)
    Noise is regenerated from (seed, tensor index), so nothing large is stored.
    """

    def __init__(self, module: torch.nn.Module, edit: Edit, device: str, noise_mode: str = "per_tensor"):
        if noise_mode not in ("per_tensor", "global"):
            raise ValueError(noise_mode)
        self.edit, self.device, self.noise_mode = edit, device, noise_mode
        refs = dict(module.named_parameters())
        refs.update(dict(module.named_buffers()))
        missing = [n for n in edit.names if n not in refs]
        if missing:
            raise KeyError(f"tensors not found in live module: {missing[:3]}")
        self.targets = {n: refs[n] for n in edit.names}
        self._global_cache: dict = {}

    def _raw_noise(self, i: int, name: str, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + i)
        return torch.randn(self.edit.delta[name].shape, generator=g, device=self.device, dtype=torch.float32)

    def _global_coeffs(self, seed: int):
        """For global mode: noise = (y - c*delta) * s over the concatenated edit."""
        if seed not in self._global_cache:
            yd = dd = yy = 0.0
            for i, name in enumerate(self.edit.names):
                d = self.edit.delta[name].to(self.device).double()
                y = self._raw_noise(i, name, seed).double()
                yd += float((y * d).sum())
                dd += float((d * d).sum())
                yy += float((y * y).sum())
            c = yd / dd
            orth_norm = math.sqrt(max(yy - yd * yd / dd, 1e-24))
            self._global_cache[seed] = (c, math.sqrt(dd) / orth_norm)
        return self._global_cache[seed]

    @torch.no_grad()
    def apply(self, a: float = 1.0, b: float = 0.0, seed: int = 0):
        coeffs = self._global_coeffs(seed) if (b != 0.0 and self.noise_mode == "global") else None
        for i, name in enumerate(self.edit.names):
            base = self.edit.base[name].to(self.device)
            d = self.edit.delta[name].to(self.device)
            w = base + a * d
            if b != 0.0:
                y = self._raw_noise(i, name, seed)
                if coeffs is None:
                    y = _orthogonal_same_norm(y, d)
                else:
                    c, s = coeffs
                    y = (y - c * d) * s
                w = w + b * y
            t = self.targets[name]
            t.data.copy_(w.to(t.dtype))

    @torch.no_grad()
    def reconstruction_error(self) -> float:
        """After apply(1, 0): how much of the edit survives the model's dtype (relative error)."""
        err = 0.0
        for name in self.edit.names:
            got = self.targets[name].detach().float().cpu() - self.edit.base[name]
            err += float((got - self.edit.delta[name]).double().pow(2).sum())
        return math.sqrt(err) / self.edit.norm

    def noise_norm(self, seed: int) -> float:
        """Norm of the noise term for a given seed (should equal the edit norm)."""
        coeffs = self._global_coeffs(seed) if self.noise_mode == "global" else None
        sq = 0.0
        for i, name in enumerate(self.edit.names):
            d = self.edit.delta[name].to(self.device)
            y = self._raw_noise(i, name, seed)
            y = _orthogonal_same_norm(y, d) if coeffs is None else (y - coeffs[0] * d) * coeffs[1]
            sq += float(y.double().pow(2).sum())
        return math.sqrt(sq)