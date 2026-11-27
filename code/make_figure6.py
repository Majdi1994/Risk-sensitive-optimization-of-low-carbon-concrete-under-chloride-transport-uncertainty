#!/usr/bin/env python3
"""Render publication Figure 6 (two panels) from archived outputs.

(a) Production matched-seed Category B regret (Table 4): ~1.606 → 0.714 (55.5%).
(b) Residual-dependence relative reductions (Table F.9): 55.2% / 28.4% / 28.2%.

Run from package root:
  python code/make_figure6.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER04 = HERE.parent
# Prefer paper04/outputs; fall back to Zenodo-style layout (sibling outputs/)
OUT_CANDIDATES = [
    PAPER04 / "outputs",
    HERE.parent / "outputs",  # if script lives under Zenodo/code
]
FIGS_CANDIDATES = [
    PAPER04 / "figures",
    HERE.parent / "figures",
]


def first_dir(cands: list[Path]) -> Path:
    for p in cands:
        if p.exists():
            return p
    p = cands[0]
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    exports = first_dir(OUT_CANDIDATES)
    figs = first_dir(FIGS_CANDIDATES)

    seed_path = exports / "rev2_seedwise_decision_audits.csv"
    dep_path = exports / "rev2_dependence_sensitivity.json"
    if not seed_path.exists():
        raise SystemExit(f"missing {seed_path}")
    if not dep_path.exists():
        raise SystemExit(f"missing {dep_path}")

    sw = pd.read_csv(seed_path)
    wide = sw.pivot(
        index="optimization_seed",
        columns="pool",
        values="s3_expected_category_b_regret",
    ).sort_index()
    mean_v = wide["mean"].to_numpy(float)
    q10_v = wide["q10"].to_numpy(float)
    m_mean, m_q10 = float(np.mean(mean_v)), float(np.mean(q10_v))
    rel_prod = 100.0 * (1.0 - m_q10 / m_mean)

    dep = json.loads(dep_path.read_text(encoding="utf-8"))
    bars = []
    for key, lab in [
        ("shared", "Shared\nresidual"),
        ("independent", "Independent"),
        ("distance_correlated", "Distance-\ncorrelated"),
    ]:
        r = dep["structures"][key]["category_b_regret"]["relative_mean_reduction"]
        bars.append((lab, 100.0 * float(r)))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "figure.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300)

    ax = axes[0]
    x = np.array([0.0, 1.0])
    for ym, yq in zip(mean_v, q10_v):
        ax.plot(x, [ym, yq], color="#9bb7d4", lw=1.0, alpha=0.85, zorder=1)
    ax.plot(x, [m_mean, m_q10], color="#1f4e79", lw=2.8, zorder=3,
            marker="o", markersize=6, label="Across-seed mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mean-search\nshortlist", "Risk-aware\n($q_{10}$) shortlist"])
    ax.set_ylabel("Expected Category B ranking regret\n(kg CO$_2$-eq m$^{-3}$ y$^{-1}$)")
    ax.set_title(
        f"Matched seeds: shortlist ranking error\n"
        f"{m_mean:.3f} → {m_q10:.3f} ({rel_prod:.1f}% reduction)",
        pad=8,
    )
    ax.set_ylim(0, max(mean_v.max(), q10_v.max()) * 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.12, 1.06, "a", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    ax = axes[1]
    labs = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = ["#1f4e79", "#5b8fb9", "#c97b2b"]
    bp = ax.bar(np.arange(3), vals, color=colors, edgecolor="#222", width=0.72, zorder=2)
    for b, v in zip(bp, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labs)
    ax.set_ylabel("Relative Category B regret reduction (%)")
    ax.set_title("Direction under residual-dependence audits", pad=8)
    ax.set_ylim(0, 70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.12, 1.06, "b", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.2)
    out = figs / "Fig6_dependence_twopanel.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  production {m_mean:.3f}->{m_q10:.3f} ({rel_prod:.1f}%)  bars={vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
