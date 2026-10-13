#!/usr/bin/env python
"""Plot neutral-loss sharpness for one or more erased models.

    python plot_losses.py results/*/loss_results.csv --out losses.png

Sharpness ratio = (loss increase around the erased model) / (loss increase around the original
model), same random directions, same data. Above 1: erasure moved the model to a sharper spot.
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sharpness_ratio(df: pd.DataFrame, bin_value="all") -> pd.DataFrame:
    """Per model and eps: mean loss increase (over seeds) for erased_noise divided by base_noise."""
    d = df[df.t_bin.astype(str) == str(bin_value)]
    m = d[d.condition.isin(["erased_noise", "base_noise"])].groupby(["name", "condition", "level"]).delta_mean.mean()
    m = m.unstack("condition")
    if not {"erased_noise", "base_noise"} <= set(m.columns):
        return pd.DataFrame()
    m["ratio"] = m["erased_noise"] / m["base_noise"]
    return m.reset_index()


def summarise(df: pd.DataFrame):
    rows = []
    allb = df[df.t_bin.astype(str) == "all"]
    r = sharpness_ratio(df)
    for name, g in allb.groupby("name"):
        cost = g[(g.condition == "scale") & np.isclose(g.level, 1.0)].delta_mean.mean()
        row = {"name": name, "edit_norm": g.edit_norm.iloc[0], "edit_cost_on_neutral": cost}
        if len(r):
            for _, x in r[r.name == name].iterrows():
                row[f"ratio@{x.level:g}"] = x.ratio
        rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str, ratio_eps: float | None):
    allb = df[df.t_bin.astype(str) == "all"]
    names = sorted(df.name.unique())
    colors = dict(zip(names, plt.cm.tab10.colors))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))

    for name in names:
        c = colors[name]
        for cond, style in (("erased_noise", "-o"), ("base_noise", "--s")):
            g = allb[(allb.name == name) & (allb.condition == cond)].groupby("level").delta_mean.agg(["mean", "std"])
            if len(g):
                ax[0].errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=style, color=c, capsize=3,
                               label=f"{name}: around {'erased' if cond == 'erased_noise' else 'original'}")
        s = allb[(allb.name == name) & (allb.condition == "scale")].sort_values("level")
        ax[2].plot(s.level, s.delta_mean, "-o", color=c, label=name)

    # per-timestep ratio at one eps
    binned = df[df.t_bin.astype(str) != "all"].copy()
    binned["t_bin"] = binned.t_bin.astype(int)
    levels = sorted(binned[binned.condition == "erased_noise"].level.unique())
    if levels:
        e = ratio_eps if ratio_eps is not None else max([l for l in levels if l <= 1] or levels)
        width = 0.8 / max(len(names), 1)
        for k, name in enumerate(names):
            g = binned[(binned.name == name) & np.isclose(binned.level, e)]
            m = g.groupby(["t_bin", "condition"]).delta_mean.mean().unstack()
            if {"erased_noise", "base_noise"} <= set(m.columns):
                ratio = m["erased_noise"] / m["base_noise"]
                ax[1].bar(ratio.index + k * width, ratio.values, width, color=colors[name], label=name)
        edges = binned.drop_duplicates("t_bin").sort_values("t_bin")
        ax[1].set_xticks(edges.t_bin + 0.4 - width / 2)
        ax[1].set_xticklabels([f"{lo}-{hi}" for lo, hi in zip(edges.t_lo, edges.t_hi)], rotation=45, fontsize=7)
        ax[1].axhline(1, color="grey", lw=0.8)
        ax[1].set(title=f"Sharpness ratio by timestep (ε = {e:g})", xlabel="timestep range (high = noisier)",
                  ylabel="loss increase: erased / original")

    ax[0].set(title="Neutral loss increase under orthogonal noise", xlabel="noise size ε (× edit norm)",
              ylabel="Δ denoising loss vs unperturbed", xscale="log")
    ax[2].set(title="Neutral-loss cost of the edit", xlabel="edit strength α", ylabel="Δ loss vs original")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[2].axhline(0, color="grey", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def _binned_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    b = df[(df.t_bin.astype(str) != "all") & df.condition.isin(["erased_noise", "base_noise"])].copy()
    b["t_bin"] = b.t_bin.astype(int)
    b["t_lo"] = b.t_lo.astype(int)
    b["t_hi"] = b.t_hi.astype(int)
    return b


def ratio_by_timestep(df: pd.DataFrame) -> pd.DataFrame:
    """Per model, noise scale and timestep range: mean loss increase around the erased model divided
    by the mean increase around the original (both averaged over noise seeds).
    'reliable' = the original's increase is more than 2 standard errors away from zero."""
    b = _binned_noise_rows(df)
    if b.empty:
        return pd.DataFrame()
    g = b.groupby(["name", "level", "t_bin", "t_lo", "t_hi", "condition"]).agg(
        mean=("delta_mean", "mean"),
        se=("delta_se", lambda s: float(np.sqrt((s ** 2).mean() / len(s)))),
    ).unstack("condition")
    if ("mean", "erased_noise") not in g.columns or ("mean", "base_noise") not in g.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "erased": g[("mean", "erased_noise")],
        "original": g[("mean", "base_noise")],
        "original_se": g[("se", "base_noise")],
    }).reset_index()
    out["ratio"] = out.erased / out.original
    out["reliable"] = out.original.abs() > 2 * out.original_se
    return out


def ratio_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Same ratio for each noise seed separately (both models got the same noise direction)."""
    b = _binned_noise_rows(df)
    if b.empty:
        return pd.DataFrame()
    p = b.pivot_table(index=["name", "level", "t_bin", "noise_seed"], columns="condition", values="delta_mean")
    if not {"erased_noise", "base_noise"} <= set(p.columns):
        return pd.DataFrame()
    p["ratio"] = p.erased_noise / p.base_noise
    return p.reset_index()


def print_ratio_table(tab: pd.DataFrame):
    t = tab.copy()
    t["range"] = t.t_lo.astype(str) + "-" + t.t_hi.astype(str)
    t["cell"] = [f"{r:.2f}" + ("" if ok else "?") for r, ok in zip(t.ratio, t.reliable)]
    order = t.drop_duplicates("t_bin").sort_values("t_bin")["range"].tolist()
    p = t.pivot_table(index=["name", "level"], columns="range", values="cell", aggfunc="first")[order]
    p.index = p.index.set_names(["model", "eps"])
    print("\nSharpness ratio (erased / original) by timestep range; '?' = original's increase within noise")
    print(p.to_string())


def _keep_levels(frame: pd.DataFrame, eps) -> pd.DataFrame:
    if frame.empty or not eps:
        return frame
    mask = np.zeros(len(frame), dtype=bool)
    for e in eps:
        mask |= np.isclose(frame.level.values, e)
    return frame[mask]


def plot_ratio_by_timestep(df: pd.DataFrame, path: str, eps=None):
    """One panel per model; one line per noise scale; x = timestep range.
    eps: list of noise scales to show (default: all available)."""
    tab, seeds = ratio_by_timestep(df), ratio_per_seed(df)
    if tab.empty:
        print("no erased_noise/base_noise pairs found; skipping the per-timestep ratio figure")
        return
    if eps:
        available = sorted(tab.level.unique())
        missing = [e for e in eps if not np.isclose(available, e).any()]
        if missing:
            print(f"note: eps {missing} not in the results (available: {[float(a) for a in available]})")
        tab, seeds = _keep_levels(tab, eps), _keep_levels(seeds, eps)
        if tab.empty:
            print("none of the requested eps values are in the results; skipping the per-timestep ratio figure")
            return
    print_ratio_table(tab)
    names = sorted(tab.name.unique())
    levels = sorted(tab.level.unique())
    colors = {l: plt.cm.viridis(i / max(len(levels) - 1, 1)) for i, l in enumerate(levels)}
    ncols = min(len(names), 3)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.4 * nrows), squeeze=False)
    ranges = tab.drop_duplicates("t_bin").sort_values("t_bin")

    for ax, name in zip(axes.flat, names):
        t, s = tab[tab.name == name], seeds[seeds.name == name] if len(seeds) else seeds
        shown = [1.0]
        for i, lvl in enumerate(levels):
            tl = t[np.isclose(t.level, lvl)].sort_values("t_bin")
            if tl.empty:
                continue
            c = colors[lvl]
            line = tl.ratio.where(tl.reliable)  # draw the line only through reliable points
            ax.plot(tl.t_bin, line, "-", color=c, lw=1.3, label=f"ε = {lvl:g}")
            ok, bad = tl[tl.reliable], tl[~tl.reliable]
            ax.plot(ok.t_bin, ok.ratio, "o", color=c, ms=5)
            ax.plot(bad.t_bin, bad.ratio, "o", mfc="none", mec=c, ms=5)
            if len(s):
                sl = s[np.isclose(s.level, lvl)]
                jitter = (i - (len(levels) - 1) / 2) * 0.06
                ax.plot(sl.t_bin + jitter, sl.ratio, ".", color=c, alpha=0.35, ms=3)
            shown += list(ok.ratio)
        lo, hi = min(shown), max(shown)
        pad = 0.15 * (hi - lo) + 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_xticks(ranges.t_bin)
        ax.set_xticklabels([f"{a}-{b}" for a, b in zip(ranges.t_lo, ranges.t_hi)], rotation=45, fontsize=7)
        ax.set(title=name, xlabel="timestep range (high = noisier)", ylabel="sharpness ratio (erased / original)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, title="hollow = unreliable\nsmall dots = single seeds", title_fontsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", default="losses.png")
    p.add_argument("--ratio-eps", type=float, default=None, help="eps for the per-timestep panel")
    p.add_argument("--ratio-out", default=None,
                   help="figure with the per-timestep ratio for every noise scale "
                        "(default: <out>_ratio_by_timestep.png)")
    p.add_argument("--show-eps", type=float, nargs="+", default=None,
                   help="noise scales to show in the per-timestep ratio figure and table, e.g. --show-eps 0.5 1 2 "
                        "(default: all)")
    args = p.parse_args()
    df = pd.concat([pd.read_csv(c, dtype={"t_bin": str}) for c in args.csvs], ignore_index=True)
    print(summarise(df).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    r = sharpness_ratio(df)
    if len(r) and ((r.base_noise.abs() < 3 * df[df.t_bin == "all"].delta_se.median()).any()):
        print("note: some original-model loss increases are within noise; ratios at those eps are unreliable")
    plot(df, args.out, args.ratio_eps)
    root, ext = os.path.splitext(args.out)
    plot_ratio_by_timestep(df, args.ratio_out or f"{root}_ratio_by_timestep{ext or '.png'}", args.show_eps)


if __name__ == "__main__":
    main()