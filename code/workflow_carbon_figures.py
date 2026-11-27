#!/usr/bin/env python3
"""Generate workflow figure, absolute-carbon figure, and refresh paired-S3 contrast."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from analysis_specification import EXPORTS, SPEC

REV = Path(__file__).resolve().parents[1]
FIGS = REV / "figures"


def make_workflow() -> Path:
    from plot_style import apply_plot_style
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.4, "ASTM C1556\narchive\n(540 / 25)"),
        (2.3, 1.4, "LOSO\nprediction\n(NGB)"),
        (4.3, 1.4, "NLL / conformal\ncalibration\n(λ ≈ 4.30)"),
        (6.3, 1.4, "Chloride\nmapping\n(t_eq, m)"),
        (8.3, 1.4, "Matched\nmean / q10\nNSGA-II"),
        (10.2, 1.4, "Decision\nlosses\n(regret, fail)"),
    ]
    for x, y, text in boxes:
        patch = FancyBboxPatch(
            (x, y), 1.55, 1.6,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor="#E8F1F8", edgecolor="#0B3D91", linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + 0.775, y + 0.8, text, ha="center", va="center", fontsize=11,
                color="#0B3D91", fontweight="bold", fontfamily="Times New Roman")

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 1.55
        x1 = boxes[i + 1][0]
        ax.annotate(
            "", xy=(x1 - 0.02, 2.2), xytext=(x0 + 0.02, 2.2),
            arrowprops=dict(arrowstyle="->", color="#1F6AA5", lw=1.3),
        )

    ax.text(
        6.0, 0.45,
        "Evidence chain: source-disjoint transport law → calibrated uncertainty → "
        "embedded SAA search → paired decision losses",
        ha="center", va="center", fontsize=11, color="#0B3D91",
        fontweight="bold", fontfamily="Times New Roman",
    )
    ax.set_title("Study workflow", fontsize=12, pad=6, fontweight="bold",
                 fontfamily="Times New Roman")
    out = FIGS / "rev2_workflow.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_carbon_figure() -> Path:
    """Front GWP against the system-boundary-consistent OPC empirical reference."""
    import json
    from plot_style import apply_plot_style, style_axes

    apply_plot_style()
    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    bench = json.loads(
        (EXPORTS / "rev2_opc_reference_benchmark.json").read_text(encoding="utf-8")
    )
    emp = bench["empirical_headline"]
    ref_gwp = float(emp["gwp_median"])
    ref_q25 = float(emp["gwp_q25"])
    ref_q75 = float(emp["gwp_q75"])
    model_opt = float(bench["model_optimal_conservative"]["gwp"])
    gwp = pools["gwp"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    ax = axes[0]
    for mode, color in (("mean", "#0B3D91"), ("q10", "#4C9BD6")):
        vals = pools.loc[pools["pool"] == mode, "gwp"].astype(float)
        ax.hist(vals, bins=30, alpha=0.55, label=f"{mode} arm", color=color, density=True)
    ax.axvspan(ref_q25, ref_q75, color="#7EB6D9", alpha=0.25, zorder=0,
               label=f"OPC empirical IQR ({ref_q25:.0f}–{ref_q75:.0f})")
    ax.axvline(ref_gwp, color="#084C61", lw=2.0, ls="--",
               label=f"OPC empirical median ({ref_gwp:.0f})")
    ax.axvline(model_opt, color="#1F6AA5", lw=1.4, ls="-.",
               label=f"OPC model-optimal ({model_opt:.0f})")
    ax.axvline(gwp.min(), color="#2E86AB", lw=1.4, ls=":",
               label=f"front low ({gwp.min():.0f})")
    ax.set_xlabel("Cradle-to-gate GWP (kg CO₂-eq m⁻³)")
    ax.set_ylabel("Density")
    ax.set_title("Front GWP against OPC reference (same boundary)")
    ax.legend(fontsize=11, frameon=False)
    style_axes(ax)

    ax = axes[1]
    for mode, color in (("mean", "#0B3D91"), ("q10", "#4C9BD6")):
        sub = pools.loc[pools["pool"] == mode].copy()
        b = sub["gwp"].astype(float) / np.maximum(sub["t_eq_mean"].astype(float), 1e-12)
        ax.hist(b, bins=30, alpha=0.55, label=f"{mode} arm", color=color, density=True)
    ax.set_xlabel("Point Category B (kg CO₂-eq m⁻³ y⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title("Corresponding Category B distribution")
    ax.legend(fontsize=11, frameon=False)
    style_axes(ax)

    fig.tight_layout()
    out = FIGS / "rev2_absolute_carbon.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "front_gwp_min": float(gwp.min()),
        "front_gwp_max": float(gwp.max()),
        "front_gwp_median": float(gwp.median()),
        "reference_opc_empirical_median_gwp": ref_gwp,
        "reference_opc_empirical_iqr": [ref_q25, ref_q75],
        "reference_opc_empirical_n": int(emp["n_strength_matched"]),
        "reference_opc_model_optimal_gwp": model_opt,
        "relative_to_empirical_at_low": float(
            bench["relative_reductions"]["front_low_vs_empirical_median"]
        ),
        "relative_to_empirical_at_median": float(
            bench["relative_reductions"]["front_median_vs_empirical_median"]
        ),
        "relative_to_model_optimal_at_low": float(
            bench["relative_reductions"]["front_low_vs_model_optimal"]
        ),
        "superseded_cement_only_gwp": float(
            bench["superseded_cement_only_reference"]["gwp"]
        ),
        "front_strength_median_mpa": float(bench["front"]["strength_median_mpa"]),
        "boundary": bench["boundary"],
    }
    (EXPORTS / "rev2_absolute_carbon.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return out


def refresh_paired_s3() -> Path:
    """Raise grey/red contrast on the paired seed-level figure."""
    audits = pd.read_csv(EXPORTS / "rev2_seedwise_decision_audits.csv")
    mean = audits[audits["pool"] == "mean"].sort_values("optimization_seed")
    q10 = audits[audits["pool"] == "q10"].sort_values("optimization_seed")
    assert list(mean["optimization_seed"]) == list(q10["optimization_seed"])

    panels = [
        ("s3_expected_category_b_regret", "Category B expected regret"),
        ("s3_top10_failure_category_b", "Category B top-10 failure"),
        ("s3_expected_J_regret", "J expected regret"),
        ("s3_top10_failure", "J top-10 failure"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, panels):
        y0 = mean[col].to_numpy(dtype=float)
        y1 = q10[col].to_numpy(dtype=float)
        for a, b in zip(y0, y1):
            ax.plot([0, 1], [a, b], color="#9AA0A6", lw=0.9, alpha=0.55, zorder=1)
        ax.plot([0, 1], [y0.mean(), y1.mean()], color="#C62828", lw=2.8, zorder=3,
                marker="o", markersize=7, label="Across-seed mean")
        ax.set_xticks([0, 1], ["mean", "q10"])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.15, 1.15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Paired seed-level S3 outcomes", fontsize=11)
    fig.tight_layout()
    out = FIGS / "rev2_paired_s3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    w = make_workflow()
    c = make_carbon_figure()
    p = refresh_paired_s3()
    print({"workflow": str(w), "carbon": str(c), "paired": str(p)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
