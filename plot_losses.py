#!/usr/bin/env python
"""Plot neutral-loss sharpness for one or more erased models.

    python plot_losses.py results/*/loss_results.csv --out losses.png

Sharpness ratio = (loss increase around the erased model) / (loss increase around the original
model), same random directions, same data. Above 1: erasure moved the model to a sharper spot.

If the runs used --random-edits, the same ratio is also computed around random edits of the same
size (random_noise / base_noise). That is the control: if random edits give similar ratios, any
edit of this size makes the model sharper; if they stay near 1, the erasure is special.
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


NOISE_CONDS = ["erased_noise", "base_noise", "random_noise"]


def merge_random_edits(df: pd.DataFrame) -> pd.DataFrame:
    """random_noise_r0, random_noise_r1, ... -> random_noise (and random_edit_r* -> random_edit), with the
    edit index kept in 'random_id'. Averages then run over random edits as well as noise seeds."""
    df = df.copy()
    rid = df.condition.str.extract(r"^random_(?:noise|edit)_r(\d+)$")[0]
    df["random_id"] = pd.to_numeric(rid, errors="coerce")
    df.loc[df.condition.str.startswith("random_noise_r"), "condition"] = "random_noise"
    df.loc[df.condition.str.startswith("random_edit_r"), "condition"] = "random_edit"
    return df


def sharpness_ratio(df: pd.DataFrame, bin_value="all") -> pd.DataFrame:
    """Per model and eps: mean loss increase (over seeds, and over random edits) around the erased model
    and around random edits, each divided by the increase around the original model."""
    d = df[df.t_bin.astype(str) == str(bin_value)]
    m = d[d.condition.isin(NOISE_CONDS)].groupby(["name", "condition", "level"]).delta_mean.mean()
    m = m.unstack("condition")
    if not {"erased_noise", "base_noise"} <= set(m.columns):
        return pd.DataFrame()
    m["ratio"] = m["erased_noise"] / m["base_noise"]
    if "random_noise" in m.columns:
        m["random_ratio"] = m["random_noise"] / m["base_noise"]
    return m.reset_index()


def summarise(df: pd.DataFrame):
    rows = []
    allb = df[df.t_bin.astype(str) == "all"]
    r = sharpness_ratio(df)
    for name, g in allb.groupby("name"):
        cost = g[(g.condition == "scale") & np.isclose(g.level, 1.0)].delta_mean.mean()
        row = {"name": name, "edit_norm": g.edit_norm.iloc[0], "edit_cost_on_neutral": cost}
        rcost = g[g.condition == "random_edit"].delta_mean
        if len(rcost):
            row["random_edit_cost"] = rcost.mean()
        if len(r):
            for _, x in r[r.name == name].iterrows():
                row[f"ratio@{x.level:g}"] = x.ratio
                if "random_ratio" in x and pd.notna(x.random_ratio):
                    row[f"random_ratio@{x.level:g}"] = x.random_ratio
        rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str, ratio_eps: float | None, stat: str = "mean"):
    allb = df[df.t_bin.astype(str) == "all"]
    names = sorted(df.name.unique())
    colors = dict(zip(names, plt.cm.tab10.colors))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))

    for name in names:
        c = colors[name]
        where = {"erased_noise": "erased", "base_noise": "original", "random_noise": "random edit"}
        for cond, style in (("erased_noise", "-o"), ("base_noise", "--s"), ("random_noise", ":^")):
            g = allb[(allb.name == name) & (allb.condition == cond)].groupby("level").delta_mean.agg(["mean", "std"])
            if len(g):
                ax[0].errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=style, color=c, capsize=3,
                               label=f"{name}: around {where[cond]}")
        s = allb[(allb.name == name) & (allb.condition == "scale")].sort_values("level")
        ax[2].plot(s.level, s.delta_mean, "-o", color=c, label=name)
        rc = allb[(allb.name == name) & (allb.condition == "random_edit")].delta_mean
        if len(rc):
            ax[2].errorbar([1.0], [rc.mean()], yerr=[rc.std() if len(rc) > 1 else 0], fmt="x", color=c, ms=9,
                           capsize=3, label=f"{name}: random edit, same size")

    # per-timestep ratio at one eps; bins where the original's increase is within noise get a "?" not a bar
    tab = ratio_by_timestep(df, stat)
    if not tab.empty:
        levels = sorted(tab.level.unique())
        e = ratio_eps if ratio_eps is not None else max(levels)
        width = 0.8 / max(len(names), 1)
        shown = [1.0]
        for k, name in enumerate(names):
            t = tab[(tab.name == name) & np.isclose(tab.level, e)].sort_values("t_bin")
            ok, bad = t[t.reliable], t[~t.reliable]
            ax[1].bar(ok.t_bin + k * width, ok.ratio, width, color=colors[name], label=name)
            for x in bad.t_bin:
                ax[1].text(x + k * width, 1.0, "?", ha="center", va="bottom", color=colors[name], fontsize=9)
            shown += list(ok.ratio)
        lo, hi = min(shown + [0.0]), max(shown)
        ax[1].set_ylim(lo - 0.1, hi * 1.12 + 0.1)
        ranges = tab.drop_duplicates("t_bin").sort_values("t_bin")
        ax[1].set_xticks(ranges.t_bin + 0.4 - width / 2)
        ax[1].set_xticklabels([f"{a}-{b}" for a, b in zip(ranges.t_lo, ranges.t_hi)], rotation=45, fontsize=7)
        ax[1].axhline(1, color="grey", lw=0.8)
        ax[1].set(title=f"Sharpness ratio by timestep (ε = {e:g}, {stat} over seeds)\n? = original's increase within noise",
                  xlabel="timestep range (high = noisier)", ylabel="loss increase: erased / original")

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
    b = df[(df.t_bin.astype(str) != "all") & df.condition.isin(NOISE_CONDS)].copy()
    b["t_bin"] = b.t_bin.astype(int)
    b["t_lo"] = b.t_lo.astype(int)
    b["t_hi"] = b.t_hi.astype(int)
    return b


def ratio_by_timestep(df: pd.DataFrame, stat: str = "mean") -> pd.DataFrame:
    """Per model, noise scale and timestep range: loss increase around the erased model divided
    by the increase around the original, each summarised over noise seeds (and random edits) with
    `stat` ('mean' or 'median'; the median is less sensitive to one unusually sharp direction).
    'reliable' = the original's increase is more than 2 standard errors away from zero."""
    b = _binned_noise_rows(df)
    if b.empty:
        return pd.DataFrame()
    g = b.groupby(["name", "level", "t_bin", "t_lo", "t_hi", "condition"]).agg(
        mean=("delta_mean", stat),
        se=("delta_se", lambda s: float(np.sqrt((s ** 2).mean() / len(s)))),
    ).unstack("condition")
    if ("mean", "erased_noise") not in g.columns or ("mean", "base_noise") not in g.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "erased": g[("mean", "erased_noise")],
        "original": g[("mean", "base_noise")],
        "original_se": g[("se", "base_noise")],
    }).reset_index()
    out["erased_se"] = g[("se", "erased_noise")].values
    out["ratio"] = out.erased / out.original
    out["category"] = _categorise(out.erased, out.erased_se, out.original, out.original_se)
    if ("mean", "random_noise") in g.columns:
        out["random"] = g[("mean", "random_noise")].values
        out["random_se"] = g[("se", "random_noise")].values
        out["random_ratio"] = out.random / out.original
        out["random_category"] = _categorise(out.random, out.random_se, out.original, out.original_se)
    out["reliable"] = out.original.abs() > 2 * out.original_se
    return out


def _categorise(x, x_se, orig, orig_se, k: float = 2.0) -> pd.Series:
    """Sign-aware reading of one cell (x = loss change around the edited model, orig = around the original):
      ratio        original's change clearly positive: the ratio x / orig is meaningful
      contradicts  x clearly negative while the original's is clearly positive (edited point is flatter / downhill)
      supports     x clearly positive while the original's is not (within noise, or negative)
      both_down    both clearly negative: the push lowers the loss at both points
      unclear      everything else (mostly: both within noise)
    'Clearly' = more than k standard errors from zero."""
    xp, xn = x > k * x_se, x < -k * x_se
    op, on = orig > k * orig_se, orig < -k * orig_se
    cat = np.where(op & xn, "contradicts",
          np.where(op, "ratio",
          np.where(xp, "supports",
          np.where(xn & on, "both_down", "unclear"))))
    return pd.Series(cat, index=x.index)


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
    if "random_ratio" in t.columns and t.random_ratio.notna().any():
        t["rcell"] = [f"{r:.2f}" + ("" if ok else "?") for r, ok in zip(t.random_ratio, t.reliable)]
        q = t.pivot_table(index=["name", "level"], columns="range", values="rcell", aggfunc="first")[order]
        q.index = q.index.set_names(["model", "eps"])
        print("\nControl: sharpness ratio (random edit / original), averaged over random edits and seeds")
        print(q.to_string())


def _keep_levels(frame: pd.DataFrame, eps) -> pd.DataFrame:
    if frame.empty or not eps:
        return frame
    mask = np.zeros(len(frame), dtype=bool)
    for e in eps:
        mask |= np.isclose(frame.level.values, e)
    return frame[mask]


def plot_ratio_by_timestep(df: pd.DataFrame, path: str, eps=None, stat: str = "mean", seed_dots: bool = False):
    """One panel per model; one line per noise scale; x = timestep range.
    eps: list of noise scales to show (default: all available)."""
    tab = ratio_by_timestep(df, stat)
    seeds = ratio_per_seed(df) if seed_dots else pd.DataFrame()
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
        t = tab[tab.name == name]
        s = seeds[seeds.name == name] if len(seeds) else seeds
        shown = [1.0]
        for i, lvl in enumerate(levels):
            tl = t[np.isclose(t.level, lvl)].sort_values("t_bin")
            if tl.empty:
                continue
            c = colors[lvl]
            line = tl.ratio.where(tl.reliable)  # draw the line only through reliable points
            ax.plot(tl.t_bin, line, "-", color=c, lw=1.3, label=f"ε = {lvl:g}")
            if "random_ratio" in tl.columns and tl.random_ratio.notna().any():
                rline = tl.random_ratio.where(tl.reliable)
                ax.plot(tl.t_bin, rline, "--", color=c, lw=1.1, marker="x", ms=5)
                shown += list(tl.random_ratio[tl.reliable].dropna())
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
        if len(s):  # say how many single-seed points fall outside the plotted range
            vals = s.ratio.replace([np.inf, -np.inf], np.nan).dropna()
            above, below = int((vals > hi + pad).sum()), int((vals < lo - pad).sum())
            if above or below:
                ax.text(0.01, 0.01, f"single-seed points off the plot: {above} above, {below} below",
                        transform=ax.transAxes, fontsize=6.5, color="grey", va="bottom")
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_xticks(ranges.t_bin)
        ax.set_xticklabels([f"{a}-{b}" for a, b in zip(ranges.t_lo, ranges.t_hi)], rotation=45, fontsize=7)
        ax.set(title=name, xlabel="timestep range (high = noisier)", ylabel="sharpness ratio (erased / original)")
        ax.grid(alpha=0.3)
        has_random = "random_ratio" in t.columns and t.random_ratio.notna().any()
        title = "hollow = unreliable" + ("\nsmall dots = single seeds" if seed_dots else "")
        if has_random:
            title = "solid = erased, dashed × = random edit\n" + title
        ax.legend(fontsize=7, title=title, title_fontsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


CATEGORY_STYLE = {
    # category: (face colour, hatch, symbol, legend text)
    "supports": ("#b2182b", "///", "▲", "▲ edited model: loss rises; original: flat or falls"),
    "contradicts": ("#2166ac", "///", "▼", "▼ edited model: loss falls; original: rises"),
    "both_down": ("#bdbdbd", "...", "↓", "↓ both fall"),
    "unclear": ("#d9d9d9", None, "?", "? both within noise"),
}


def plot_ratio_heatmap(df: pd.DataFrame, path: str, eps=None, stat: str = "mean"):
    """Compact view: one row of panels per model, rows = noise size, columns = timestep range,
    colour = sharpness ratio on a log scale (white = 1, red = sharper than the original, blue = flatter).
    Cells where the ratio is not meaningful are marked by sign instead (see CATEGORY_STYLE).
    If random-edit runs exist, the control is shown in a second panel with the same scheme."""
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch, Rectangle

    tab = _keep_levels(ratio_by_timestep(df, stat), eps)
    if tab.empty:
        return
    names = sorted(tab.name.unique())
    has_random = "random_ratio" in tab.columns and tab.random_ratio.notna().any()
    kinds = [("ratio", "category", "erased / original")]
    if has_random:
        kinds.append(("random_ratio", "random_category", "random edit / original (control)"))
    ranges = tab.drop_duplicates("t_bin").sort_values("t_bin")
    xlabels = [f"{a}-{b}" for a, b in zip(ranges.t_lo, ranges.t_hi)]
    bins = list(ranges.t_bin)
    levels = sorted(tab.level.unique())
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.0)  # log2 ratio: 0.5 ... 1 ... 4
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(len(names), len(kinds),
                             figsize=(6.2 * len(kinds), 0.42 * len(levels) * len(names) + 1.6 * len(names)),
                             squeeze=False, constrained_layout=True)
    im = None
    print("\nHeatmap cell counts (ratio = normal coloured cell):")
    for r, name in enumerate(names):
        t = tab[tab.name == name]
        for c, (col, catcol, label) in enumerate(kinds):
            ax = axes[r, c]
            val = t.pivot_table(index="level", columns="t_bin", values=col, aggfunc="first").reindex(index=levels, columns=bins)
            cat = t.pivot_table(index="level", columns="t_bin", values=catcol, aggfunc="first").reindex(index=levels, columns=bins)
            show = (cat == "ratio") & (val > 0)
            im = ax.imshow(np.log2(val.where(show)).values, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(levels)):
                for j in range(len(bins)):
                    k, v = cat.values[i, j], val.values[i, j]
                    if k == "ratio" and v > 0:
                        dark = abs(np.log2(v)) > 1.2
                        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                                color="white" if dark else "black")
                        continue
                    if k == "ratio":  # original rises clearly, edited change ~0 or slightly negative
                        k = "contradicts" if v < 0 else "unclear"
                    face, hatch, sym, _ = CATEGORY_STYLE.get(k, CATEGORY_STYLE["unclear"])
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, hatch=hatch,
                                           edgecolor="white", lw=0))
                    ax.text(j, i, sym, ha="center", va="center", fontsize=8,
                            color="white" if k in ("supports", "contradicts") else "black")
            counts = cat.stack().value_counts().to_dict()
            print(f"  {name:28s} {label:34s} " + ", ".join(f"{k}: {n}" for k, n in sorted(counts.items())))
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels if r == len(names) - 1 else [], rotation=45, fontsize=7)
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([f"ε = {l:g}" for l in levels], fontsize=8)
            ax.set_title(f"{name}: {label}", fontsize=9)
            if r == len(names) - 1:
                ax.set_xlabel("timestep range (high = noisier)")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cb.set_ticks([-1, 0, 1, 2])
    cb.set_ticklabels(["0.5", "1", "2", "4"])
    cb.set_label(f"sharpness ratio ({stat} over seeds)", fontsize=8)
    handles = [Patch(facecolor=f, hatch=h, edgecolor="white", label=lab) for f, h, _, lab in CATEGORY_STYLE.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), fontsize=8,
               frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    p.add_argument("--stat", default="mean", choices=["mean", "median"],
                   help="how to combine noise seeds (and random edits) per condition; median is less "
                        "sensitive to one unusually sharp direction")
    p.add_argument("--seed-dots", action="store_true", help="show single-seed ratios as small dots in the line figure")
    args = p.parse_args()
    df = pd.concat([pd.read_csv(c, dtype={"t_bin": str}) for c in args.csvs], ignore_index=True)
    df = merge_random_edits(df)
    print(summarise(df).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    r = sharpness_ratio(df)
    if len(r) and ((r.base_noise.abs() < 3 * df[df.t_bin == "all"].delta_se.median()).any()):
        print("note: some original-model loss increases are within noise; ratios at those eps are unreliable")
    plot(df, args.out, args.ratio_eps, args.stat)
    root, ext = os.path.splitext(args.out)
    ext = ext or ".png"
    plot_ratio_by_timestep(df, args.ratio_out or f"{root}_ratio_by_timestep{ext}", args.show_eps,
                           args.stat, args.seed_dots)
    plot_ratio_heatmap(df, f"{root}_ratio_heatmap{ext}", args.show_eps, args.stat)


if __name__ == "__main__":
    main()