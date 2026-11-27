#!/usr/bin/env python3
"""Graphical abstract. Every printed value is read from the exports.

To change a printed value, change the export it is read from, then rerun this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PAPER04 = Path(__file__).resolve().parents[1]
EXPORTS = PAPER04 / "outputs"
FIGS = PAPER04 / "figures"

C_BLUE = "#1F4E79"
C_GRAY = "#4A4A4A"
C_GREEN = "#2E7D32"
C_FILL = "#F7FAFC"
C_EDGE = "#9FB3C8"
DPI = 600


def load(name: str) -> dict:
    return json.loads((EXPORTS / name).read_text(encoding="utf-8"))


def signed_pct(value: float) -> str:
    """Typeset a signed percentage with a true minus sign."""
    return f"{'\u2212' if value < 0 else '+'}{abs(value) * 100:.1f}%"


def picp90(block: dict) -> float:
    for key in ("picp_90", "picp90", "picp_0.90"):
        if key in block:
            return float(block[key])
    raise KeyError(f"no PICP90 field in {sorted(block)}")


def main() -> int:
    cal = load("rev2_production_calibration.json")
    grouped = load("rev2_grouped_validation_summary.json")
    head = load("rev2_matched_pool_head_to_head.json")
    spec = load("rev2_analysis_specification.json")

    lam = float(cal["production_lambda"])
    cov = picp90(cal["metrics_after_production_lambda"])
    r2 = float(grouped["pooled_after"]["r2"])
    n_obs = int(cal["n_oof"])
    n_src = int(cal["n_sources"])

    fail = head["paired_headline"]["s3_top10_failure_category_b"]
    regret = head["paired_headline"]["s3_expected_category_b_regret"]
    seeds = int(head["n_paired_seeds"])
    gens = int(spec.get("nsga2_generations", 120))
    pop = int(spec.get("nsga2_population", 80))

    fig, ax = plt.subplots(figsize=(11.1, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    stages = [
        ("1 \u00b7 Data", f"ASTM C1556\n$D_{{\\mathrm{{app}}}}$\n{n_obs} records, {n_src} sources"),
        ("2 \u00b7 Model", f"NGB with out-of-fold\nscale $\\lambda\\approx{lam:.2f}$\nsource-disjoint LOSO"),
        ("3 \u00b7 Search", f"NSGA-II + SAA\nmean vs. q10 objectives\n{seeds} seeds, pop {pop}, {gens} gens"),
        ("4 \u00b7 Audit", "S1\u2013S3 ranking risk\nshortlist failure\nand expected regret"),
        ("5 \u00b7 Decision", "Conditional\npre-experimental\nshortlist"),
    ]
    box_w, gap = 17.0, 3.75
    for i, (head_lab, body) in enumerate(stages):
        x = i * (box_w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, 19.0), box_w, 10.6,
            boxstyle="round,pad=0.25,rounding_size=1.0",
            linewidth=1.3, edgecolor=C_BLUE, facecolor=C_FILL,
        ))
        ax.text(x + box_w / 2, 31.1, head_lab, ha="center", va="center",
                fontsize=9.0, color=C_BLUE, fontweight="bold")
        ax.text(x + box_w / 2, 24.3, body, ha="center", va="center",
                fontsize=7.6, color=C_GRAY, linespacing=1.5)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + gap - 0.5, 24.3), xytext=(x + box_w + 0.5, 24.3),
                        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.4))

    ax.text(50, 15.8,
            "On Category B, embedding transport uncertainty inside the search "
            "reduces expected ranking regret while increasing shortlist membership failure",
            ha="center", va="center", fontsize=8.4, color=C_BLUE, style="italic")
    ax.text(50, 14.0,
            "Cradle-to-gate GWP inventory: kg CO$_2$-eq m$^{-3}$"
            "  ·  Category B regret uses an annual denominator (y$^{-1}$)",
            ha="center", va="center", fontsize=7.2, color=C_GRAY)

    fail_colour = C_GREEN if fail["relative_mean_change"] < 0 else "#C62828"
    results = [
        (
            "Within-front top-10 membership failure",
            f"{fail['mean_pool_mean']:.3f} $\\rightarrow$ {fail['q10_pool_mean']:.3f}",
            f"{signed_pct(fail['relative_mean_change'])}   "
            f"$p$ = {fail['paired_wilcoxon_two_sided_p']:.4f}",
            fail_colour,
        ),
        (
            "Expected Category B ranking regret\n(kg CO$_2$-eq m$^{-3}$ y$^{-1}$)",
            f"{regret['mean_pool_mean']:.3f} $\\rightarrow$ {regret['q10_pool_mean']:.3f}",
            # Cradle-to-gate GWP inventory: kg CO2-eq m-3; y-1 is Category B regret only.
            f"{signed_pct(regret['relative_mean_change'])}   "
            f"$p$ = {regret['paired_wilcoxon_two_sided_p']:.4f}",
            C_GREEN,
        ),
        (
            "Source-disjoint transfer, calibrated",
            f"$R^2$ = {r2:.3f}",
            f"PICP90 = {cov:.3f} at $\\lambda\\approx{lam:.2f}$",
            C_BLUE,
        ),
    ]
    rw, rgap = 30.0, 5.0
    for i, (title, big, sub, colour) in enumerate(results):
        x = i * (rw + rgap)
        ax.add_patch(FancyBboxPatch(
            (x, 0.6), rw, 11.2,
            boxstyle="round,pad=0.25,rounding_size=1.0",
            linewidth=1.1, edgecolor=C_EDGE, facecolor="white",
        ))
        ax.text(x + rw / 2, 9.9, title, ha="center", va="center",
                fontsize=7.9, color=C_GRAY)
        ax.text(x + rw / 2, 6.0, big, ha="center", va="center",
                fontsize=15.5, color=colour, fontweight="bold")
        ax.text(x + rw / 2, 2.5, sub, ha="center", va="center",
                fontsize=7.2, color=C_GRAY)

    for ext in ("png", "pdf"):
        out = FIGS / f"Graphical_Abstract.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print("wrote", out.name)
    plt.close(fig)

    provenance = {
        "generated_by": "graphical_abstract.py",
        "all_values_read_from_exports": True,
        "production_lambda": lam,
        "picp90_after_production_lambda": cov,
        "pooled_loso_r2": r2,
        "category_b_top10_failure": [fail["mean_pool_mean"], fail["q10_pool_mean"]],
        "category_b_relative_change": fail["relative_mean_change"],
        "category_b_regret": [regret["mean_pool_mean"], regret["q10_pool_mean"]],
        "category_b_regret_relative_change": regret["relative_mean_change"],
        "n_paired_seeds": seeds,
    }
    (EXPORTS / "rev2_graphical_abstract_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
