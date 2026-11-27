#!/usr/bin/env python3
"""Post-hoc strength feasibility retention under alternative strength treatments.

Production fronts were generated under the fitted Ridge strength model with the
active floor SPEC.active_strength_floor_mpa (engineering target + nested MAE;
41.53 MPa under the locked constants). This script reports what fraction of
Category B shortlisted candidates on the deposited thirty-seed fronts would
still clear that floor if strength were supplied by the legacy closed-form
comparator or by constant treatments. No optimization is re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_specification import EXPORTS, SPEC  # noqa: E402
from matched_search import legacy_surrogate_strength  # noqa: E402

TOP_K = SPEC.top_k
FLOOR = SPEC.active_strength_floor_mpa
MEDIAN_CS = 51.31


def shortlist_mask(pool: pd.DataFrame) -> np.ndarray:
    """Top-K by Category B point score (GWP / mean t_eq) within one arm-seed."""
    score = pool["gwp"].to_numpy(dtype=float) / np.maximum(
        pool["t_eq_mean"].to_numpy(dtype=float), 1e-12
    )
    order = np.argsort(score)
    mask = np.zeros(len(pool), dtype=bool)
    mask[order[: min(TOP_K, len(pool))]] = True
    return mask


def alternative_strengths(pool: pd.DataFrame) -> dict[str, np.ndarray]:
    fitted = pool["strength"].to_numpy(dtype=float)
    surrogate = np.array([
        legacy_surrogate_strength(
            float(r.w_b_ratio), float(r.scm_replacement), float(r.sf_frac)
        )
        for r in pool.itertuples()
    ])
    return {
        "fitted_ridge": fitted,
        "legacy_closed_form": surrogate,
        "median_constant": np.full(len(pool), MEDIAN_CS),
        "surrogate_mean_constant": np.full(len(pool), float(fitted.mean())),
    }


def retention(strengths: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sel = strengths[mask]
    all_ = strengths
    return {
        "shortlist_n": int(mask.sum()),
        "shortlist_retained": int((sel >= FLOOR).sum()),
        "shortlist_retention": float((sel >= FLOOR).mean()) if len(sel) else np.nan,
        "pool_retention": float((all_ >= FLOOR).mean()),
        "shortlist_min_mpa": float(sel.min()) if len(sel) else np.nan,
        "pool_min_mpa": float(all_.min()),
    }


def main() -> int:
    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    # Constant equal to the mean of deposited pool strengths (not a new fit).
    global_surrogate_mean = float(pools["strength"].mean())

    rows: list[dict[str, Any]] = []
    for seed in sorted(pools["optimization_seed"].unique()):
        for mode in ("mean", "q10"):
            pool = pools[
                (pools["optimization_seed"] == seed) & (pools["pool"] == mode)
            ].reset_index(drop=True)
            mask = shortlist_mask(pool)
            alts = alternative_strengths(pool)
            alts["surrogate_mean_constant"] = np.full(len(pool), global_surrogate_mean)
            for treatment, vals in alts.items():
                rec = retention(vals, mask)
                rows.append({
                    "optimization_seed": int(seed),
                    "pool": mode,
                    "treatment": treatment,
                    **rec,
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(EXPORTS / "rev2_strength_retention.csv", index=False)

    summary: dict[str, Any] = {
        "question": (
            "What fraction of Category B shortlisted candidates on the deposited "
            "thirty-seed fronts still clear the active strength floor under alternative "
            "strength treatments?"
        ),
        "floor_mpa": FLOOR,
        "top_k": TOP_K,
        "n_paired_seeds": int(pools["optimization_seed"].nunique()),
        "shortlist_definition": "top-K by Category B point score GWP/mean(t_eq) within each arm-seed",
        "treatments": {},
        "note": (
            "Post-hoc feasibility retention on candidates already generated under "
            "the fitted Ridge model. Companion to the SI strength-retention audit; "
            "not a replacement for the production Table 4 decision contrast."
        ),
    }
    for treatment in frame["treatment"].unique():
        sub = frame[frame["treatment"] == treatment]
        summary["treatments"][treatment] = {
            "mean_shortlist_retention": float(sub["shortlist_retention"].mean()),
            "min_shortlist_retention": float(sub["shortlist_retention"].min()),
            "mean_pool_retention": float(sub["pool_retention"].mean()),
            "min_pool_retention": float(sub["pool_retention"].min()),
            "worst_shortlist_min_mpa": float(sub["shortlist_min_mpa"].min()),
            "n_seeds_full_shortlist_retention": int(
                (sub.groupby("optimization_seed")["shortlist_retention"].min() >= 1.0 - 1e-12).sum()
            ),
        }

    (EXPORTS / "rev2_strength_retention.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
