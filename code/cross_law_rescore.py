#!/usr/bin/env python3
"""Cross-law Category B regret rescoring of deposited NSGA fronts.

Laws:
  1) production — deposited mu/sigma + production λ (sanity vs 1.606→0.714)
  2) constant_sigma — deposited mu + constant raw σ (pool median) + production λ

Does NOT re-run NSGA. Writes outputs/_cross_law_rescore/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dependence_sensitivity import build_problems, genotype_from_pool  # noqa: E402
from analysis_specification import EXPORTS, SPEC  # noqa: E402
from matched_search import (  # noqa: E402
    DATA_PATH,
    GWP,
    align_X,
    derive_support,
    impact_from_mix,
    mk_mean_from_workbook,
    paired_metric_inference,
    s3_draw_metrics,
    sample_ccrit,
    sample_cs,
    t_eq,
    t_init_years,
    union_draw_baseline,
)

OUT = EXPORTS / "_cross_law_rescore"
OUT.mkdir(parents=True, exist_ok=True)

MC_N = 2000
COVER_M = 0.050
PROD_LAM = 4.3007



def spearman_sigma_error() -> dict:
    preds = pd.read_csv(EXPORTS / "rev2_grouped_validation_predictions.csv")
    y = preds["y_log10_Da"].to_numpy() if "y_log10_Da" in preds.columns else preds["y"].to_numpy()
    mu = preds["mu"].to_numpy() if "mu" in preds.columns else preds["mu_log10"].to_numpy()
    sig_cal = preds["sigma_calibrated"].to_numpy()
    # raw/production-proportional: if raw sigma present use it; else invert fold scales
    if "sigma" in preds.columns:
        sig_raw = preds["sigma"].to_numpy()
    elif "sigma_raw" in preds.columns:
        sig_raw = preds["sigma_raw"].to_numpy()
    else:
        # sigma_calibrated / fold scale ≈ raw; use production λ ranking via raw proxy
        sig_raw = sig_cal  # fallback; still report both if identical
    err = np.abs(y - mu)
    rho_fold, p_fold = stats.spearmanr(sig_cal, err)
    rho_prod, p_prod = stats.spearmanr(sig_raw, err)
    # production-λ scaled ranks equal raw ranks if λ constant
    rho_lam, p_lam = stats.spearmanr(PROD_LAM * sig_raw, err)
    return {
        "fold_local_calibrated_sigma": {"spearman_rho": float(rho_fold), "p": float(p_fold)},
        "raw_or_production_proportional_sigma": {
            "spearman_rho": float(rho_prod),
            "p": float(p_prod),
            "note": "ranks equal under global production λ",
        },
        "production_lambda_times_raw": {"spearman_rho": float(rho_lam), "p": float(p_lam)},
        "n": int(len(err)),
    }


def score_from_arrays(
    pools: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    m: np.ndarray,
    gwp: np.ndarray,
    lam: float,
    seeds: list[int],
) -> tuple[pd.DataFrame, dict]:
    """Score using precomputed arrays aligned to pools row order."""
    rows = []
    for seed in seeds:
        audit_seed = 92_000 + int(seed)
        scored = {}
        for mode in ("mean", "q10"):
            mask = (pools["pool"] == mode) & (pools["optimization_seed"] == seed)
            idx = pools.index[mask]
            # preserve local_id order
            order = pools.loc[idx].sort_values("local_id").index.to_numpy()
            pos = np.array([pools.index.get_loc(i) for i in order])
            n = len(pos)
            rng = np.random.default_rng(audit_seed)
            z = rng.standard_normal(MC_N)
            z = np.broadcast_to(z, (n, MC_N)).copy()
            cs, cc = sample_cs(MC_N, rng), sample_ccrit(MC_N, rng)
            logd = mu[pos, None] + lam * sigma[pos, None] * z
            d = 10.0 ** logd * 1e-12
            te = t_eq(t_init_years(d, m[pos, None], cs, cc))
            scored[mode] = gwp[pos, None] / np.maximum(te, 1e-12)
        base_b = union_draw_baseline(scored["mean"], scored["q10"])
        for mode in ("mean", "q10"):
            fail, regret, _, mean_b = s3_draw_metrics(scored[mode], 10, baseline=base_b)
            rank = np.argsort(mean_b)
            ktop = min(10, len(mean_b))
            rows.append({
                "optimization_seed": int(seed),
                "pool": mode,
                "s3_expected_category_b_regret": float(np.mean((regret / MC_N)[rank[:ktop]])),
                "s3_top10_failure_category_b": float(np.mean((fail / MC_N)[rank[:ktop]])),
            })
    frame = pd.DataFrame(rows)
    return frame, paired_metric_inference(frame, "s3_expected_category_b_regret")



def main() -> int:
    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    seeds = sorted(int(s) for s in pools["optimization_seed"].unique())
    print(f"pools={len(pools)} seeds={len(seeds)} mc={MC_N}")

    mu_dep = pools["ngb_mu_log10"].to_numpy(float)
    sig_dep = pools["ngb_sigma_log10"].to_numpy(float)
    m_dep = pools["m"].to_numpy(float)
    gwp_dep = pools["gwp"].to_numpy(float)

    results = {"mc_draws": MC_N, "cover_m": COVER_M, "n_seeds": len(seeds)}

    print("1) production deposited law…")
    frame_p, inf_p = score_from_arrays(pools, mu_dep, sig_dep, m_dep, gwp_dep, PROD_LAM, seeds)
    frame_p.to_csv(OUT / "seedwise_production.csv", index=False)
    results["production"] = {
        "lambda": PROD_LAM,
        "mean_pool_mean": inf_p.get("mean_pool_mean"),
        "q10_pool_mean": inf_p.get("q10_pool_mean"),
        "relative_mean_change": inf_p.get("relative_mean_change"),
        "materially_reduces": inf_p.get("materially_reduces"),
        "paired_bootstrap_mean_delta_95ci": inf_p.get("paired_bootstrap_mean_delta_95ci"),
    }
    print(" production", results["production"])

    const_sig = float(np.median(sig_dep))
    print(f"2) constant_sigma={const_sig:.6f}…")
    sig_const = np.full_like(sig_dep, const_sig)
    frame_c, inf_c = score_from_arrays(pools, mu_dep, sig_const, m_dep, gwp_dep, PROD_LAM, seeds)
    frame_c.to_csv(OUT / "seedwise_constant_sigma.csv", index=False)
    results["constant_sigma"] = {
        "lambda": PROD_LAM,
        "constant_sigma_value": const_sig,
        "mean_pool_mean": inf_c.get("mean_pool_mean"),
        "q10_pool_mean": inf_c.get("q10_pool_mean"),
        "relative_mean_change": inf_c.get("relative_mean_change"),
        "materially_reduces": inf_c.get("materially_reduces"),
        "paired_bootstrap_mean_delta_95ci": inf_c.get("paired_bootstrap_mean_delta_95ci"),
    }
    print(" constant_sigma", results["constant_sigma"])

    print("3) spearman sigma vs |error|…")
    results["spearman_sigma_vs_abs_error"] = spearman_sigma_error()
    print(results["spearman_sigma_vs_abs_error"])

    # Decision helper
    def rel(x):
        return float(x["relative_mean_change"]) if x.get("relative_mean_change") is not None else None

    results["interpretation"] = {
        "production_relative_change": rel(results["production"]),
        "constant_sigma_relative_change": rel(results["constant_sigma"]),
        "q10_advantage_survives_constant_sigma": bool(
            results["constant_sigma"].get("materially_reduces")
            and (rel(results["constant_sigma"]) or 0) < 0
        ),
    }
    (OUT / "cross_law_rescore_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("WROTE", OUT / "cross_law_rescore_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
