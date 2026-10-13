#!/usr/bin/env python
"""Plot stability / efficiency / utility curves for one or more erased models.

    python plot_results.py results/*/results.csv --out figs/stability.png

Recovery puts every model on the same scale:
    recovery = (rate - rate_erased) / (rate_base - rate_erased)
0 = still fully erased, 1 = concept back at the original model's level.
rate_base and rate_erased come from the scale condition at alpha = 0 and alpha = 1.
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def add_recovery(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in df.groupby("name"):
        scale = g[g.condition == "scale"]
        base = scale.loc[np.isclose(scale.level, 0.0), "concept_rate"].mean()
        erased = scale.loc[np.isclose(scale.level, 1.0), "concept_rate"].mean()
        gap = base - erased
        if not np.isfinite(gap) or gap < 0.05:
            print(f"[{name}] warning: base rate {base:.3f} vs erased rate {erased:.3f}; "
                  f"recovery is unreliable when the erasure barely changed anything")
        g = g.copy()
        g["recovery"] = (g.concept_rate - erased) / gap if gap > 0 else np.nan
        g["rate_base"], g["rate_erased"] = base, erased
        out.append(g)
    return pd.concat(out)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per model and noise condition: area under the mean recovery curve over eps in [0, max eps],
    divided by max eps. The curve starts at (0, 0), the erased model. Larger = more fragile erasure.
    Only compare AUCs computed over the same eps range."""
    rows = []
    for name, g in df.groupby("name"):
        row = {"name": name, "rate_base": g.rate_base.iloc[0], "rate_erased": g.rate_erased.iloc[0]}
        for cond in ("orthogonal", "erased_noise"):
            o = g[g.condition == cond].groupby("level").recovery.mean().sort_index()
            if o.empty:
                continue
            x = np.concatenate([[0.0], o.index.values])
            y = np.concatenate([[0.0], o.values])
            row[f"{cond}_auc"] = _trapz(y, x) / x.max()
            row[f"{cond}_recovery_at_eps{x.max():g}"] = y[-1]
        rows.append(row)
    out = pd.DataFrame(rows)
    key = next((c for c in out.columns if c.endswith("_auc")), None)
    return out.sort_values(key, ascending=False) if key else out


def plot(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = dict(zip(sorted(df.name.unique()), plt.cm.tab10.colors))

    for name, g in df.groupby("name"):
        c = colors[name]
        for cond, style in (("orthogonal", "-o"), ("erased_noise", ":^")):
            o = g[g.condition == cond].groupby("level").recovery.agg(["mean", "std"]).sort_index()
            if o.empty:
                continue
            x = np.concatenate([[0.0], o.index.values])
            m = np.concatenate([[0.0], o["mean"].values])
            sd = np.concatenate([[0.0], o["std"].fillna(0).values])
            ax[0].plot(x, m, style, color=c, label=f"{name} ({cond})")
            ax[0].fill_between(x, m - sd, m + sd, color=c, alpha=0.15)

        sc = g[g.condition == "scale"].sort_values("level")
        ax[1].plot(sc.level, sc.recovery, "-o", color=c, label=name)

        for cond, style in (("orthogonal", "-o"), ("erased_noise", ":^"), ("base_noise", "--s")):
            u = g[g.condition == cond].groupby("level").clip_score.mean().sort_index()
            if len(u):
                ax[2].plot(u.index, u.values, style, color=c, label=f"{name} ({cond})")
        ref = g[(g.condition == "scale") & np.isclose(g.level, 0.0)].clip_score.mean()
        ax[2].axhline(ref, color=c, lw=0.8, alpha=0.5)

    if df.level[df.condition.isin(["orthogonal", "erased_noise", "base_noise"])].max() > 1.5:
        for a in (ax[0], ax[2]):
            a.set_xscale("symlog", linthresh=0.5)
    ax[0].set(title="Stability: noise orthogonal to the edit", xlabel="noise size ε (× edit norm)",
              ylabel="concept recovery (0 = erased, 1 = original)")
    ax[1].set(title="Efficiency: scaling the edit", xlabel="edit strength α (1 = full erasure)",
              ylabel="concept recovery")
    ax[2].set(title="Model still working? (neutral prompts)", xlabel="noise size ε (× edit norm)", ylabel="CLIP score")
    for a in ax[:2]:
        a.axhline(0, color="grey", lw=0.5)
        a.axhline(1, color="grey", lw=0.5, ls=":")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", default="stability.png")
    args = p.parse_args()
    df = add_recovery(pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True))
    print(summarise(df).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot(df, args.out)


if __name__ == "__main__":
    main()