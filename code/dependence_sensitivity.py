#!/usr/bin/env python3
"""Post-hoc residual-dependence sensitivity for Category B / J S3 audits.

Production uses a shared standardized residual across candidates (CRN). This
diagnostic recomputes the seedwise S3 endpoints under three structures on the
deposited fronts, without re-running NSGA-II:

1. shared — identical z across candidates (production)
2. independent — independent z per candidate
3. distance_correlated — per-draw MVN with corr = exp(-d_ij / median_d)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analysis_specification import EXPORTS, SPEC, write_specification
from matched_search import (
    DATA_PATH,
    GWP,
    ReconstructionProblem,
    TOP_K,
    aggregate_seedwise,
    align_X,
    derive_support,
    flatten_seed_summary,
    impact_from_mix,
    load_lambda,
    mk_mean_from_workbook,
    paired_metric_inference,
    s3_draw_metrics,
    sample_ccrit,
    sample_cs,
    t_eq,
    t_init_years,
    union_draw_baseline,
    write_json,
)


def genotype_from_pool(pool: pd.DataFrame, support: dict[str, Any]) -> np.ndarray:
    s = support
    return np.column_stack([
        (pool["w_b_ratio"] - s["w_b_min"]) / (s["w_b_max"] - s["w_b_min"]),
        pool["scm_replacement"] / SPEC.max_scm_total_ratio,
        (pool["binder_content"] - s["binder_min"]) / (s["binder_max"] - s["binder_min"]),
        (pool["fine_agg_ratio"] - 0.30) / 0.40,
        pool["fa_frac"], pool["slag_frac"], pool["sf_frac"], pool["mk_frac"],
        (pool["SP"] - s["sp_min"]) / (s["sp_max"] - s["sp_min"]),
    ])


def make_residuals(
    structure: str,
    n_cand: int,
    mc_n: int,
    rng: np.random.Generator,
    feature_dist: np.ndarray | None,
) -> np.ndarray:
    """Return z with shape (n_cand, mc_n)."""
    if structure == "shared":
        z = rng.standard_normal(mc_n)
        return np.broadcast_to(z, (n_cand, mc_n)).copy()
    if structure == "independent":
        return rng.standard_normal((n_cand, mc_n))
    if structure != "distance_correlated":
        raise ValueError(structure)
    if feature_dist is None:
        raise ValueError("feature_dist required for distance_correlated")
    med = float(np.median(feature_dist[np.triu_indices(n_cand, k=1)]))
    scale = max(med, 1e-9)
    corr = np.exp(-feature_dist / scale)
    # Numerical PSD guard
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    jitter = 1e-8 * np.eye(n_cand)
    try:
        chol = np.linalg.cholesky(corr + jitter)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr + jitter)
        eigvals = np.clip(eigvals, 1e-10, None)
        chol = eigvecs @ np.diag(np.sqrt(eigvals))
    white = rng.standard_normal((n_cand, mc_n))
    return chol @ white


def pairwise_feature_distance(aligned: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    z = scaler.transform(aligned)
    # Euclidean distances in the same standardized space used for OOD.
    diff = z[:, None, :] - z[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def score_arm(
    pool: pd.DataFrame,
    problem: ReconstructionProblem,
    structure: str,
    mc_n: int,
    audit_seed: int,
    train_scaler: StandardScaler,
) -> dict[str, np.ndarray]:
    X = genotype_from_pool(pool, problem.support)
    base = problem.evaluate_details(X)
    n_cand = len(pool)
    rng = np.random.default_rng(audit_seed)
    # Keep exposure draws shared; only the diffusivity residual structure varies.
    cs, cc = sample_cs(mc_n, rng), sample_ccrit(mc_n, rng)
    feat_dist = None
    if structure == "distance_correlated":
        feat_dist = pairwise_feature_distance(
            align_X(base["records"], problem.all_names), train_scaler
        )
    z = make_residuals(structure, n_cand, mc_n, rng, feat_dist)
    logd = base["mu"][:, None] + problem.lam * base["sigma"][:, None] * z
    d = 10**logd * 1e-12
    te = t_eq(t_init_years(d, base["m"][:, None], cs, cc))
    gwp = np.asarray([impact_from_mix(m, GWP) for m in base["mixes"]], dtype=float)
    b = gwp[:, None] / np.maximum(te, 1e-12)
    j = gwp[:, None] * (d / 1e-12)
    return {"b": b, "j": j}


def summarize_scores(
    scores_b: np.ndarray,
    scores_j: np.ndarray,
    baseline_b: np.ndarray,
    baseline_j: np.ndarray,
    pool_name: str,
    seed: int,
) -> dict[str, Any]:
    mc_n = scores_b.shape[1]
    k = min(TOP_K, scores_b.shape[0])
    fail_b, regret_b, _, mean_b = s3_draw_metrics(scores_b, TOP_K, baseline=baseline_b)
    fail_j, regret_j, _, mean_j = s3_draw_metrics(scores_j, TOP_K, baseline=baseline_j)
    rank_b = np.argsort(mean_b)
    rank_j = np.argsort(mean_j)
    return {
        "optimization_seed": seed,
        "pool": pool_name,
        "n": int(scores_b.shape[0]),
        "production_lambda_used": None,
        "S1": {"kendall_gwp_vs_category_b": None, "topk_overlap_gwp_vs_category_b": None},
        "S2": {
            "kendall_J_vs_category_b": None,
            "topk_overlap_J_vs_category_b": None,
            "J_definition": "GWP * [Dapp/(1e-12 m2/s)]",
        },
        "S3": {
            "mean_topk_failure_among_meanB_topk": float(
                np.mean((fail_b / mc_n)[rank_b[:k]])
            ),
            "mean_expected_category_b_regret_among_meanB_topk": float(
                np.mean((regret_b / mc_n)[rank_b[:k]])
            ),
            "mean_within_shortlist_category_b_dispersion_among_meanB_topk": None,
            "mean_topk_failure_among_meanJ_topk": float(
                np.mean((fail_j / mc_n)[rank_j[:k]])
            ),
            "mean_expected_J_regret_among_meanJ_topk": float(
                np.mean((regret_j / mc_n)[rank_j[:k]])
            ),
            "mean_within_shortlist_J_dispersion_among_meanJ_topk": None,
            "regret_baseline": "matched_seed_union",
            "mc_draws": mc_n,
            "common_random_numbers": False,
            "mc_convergence_category_b": [],
            "mc_convergence_J": [],
        },
        "mean_vs_q10": {"kendall": None, "topk_overlap": None},
        "freeze25_sensitivity": {"kendall_primary_vs_freeze": None, "topk_overlap": None},
        "category_b_vs_inverse_t_eq": {"kendall": None},
        "permutation_null_S1": {
            "tau_two_sided_p": None, "overlap_upper_p": None,
        },
        "permutation_null_S2": {
            "tau_two_sided_p": None, "overlap_upper_p": None,
        },
    }


def score_union_arm(
    subsets: dict[str, pd.DataFrame],
    problems: dict[tuple[str, int], ReconstructionProblem],
    seed: int,
    structure: str,
    mc_n: int,
    audit_seed: int,
    train_scaler: StandardScaler,
) -> dict[str, dict[str, np.ndarray]]:
    """Draw residual/exposure noise once over the seed's 160-candidate union.

    Production previously restarted the RNG for each 80-row pool, so independent
    and distance-correlated arms accidentally reused the same residual row for
    different mixtures at the same local_id. The union draw removes that artifact.
    """
    modes = ("mean", "q10")
    bases = {}
    records = []
    lengths = {}
    for mode in modes:
        subset = subsets[mode]
        problem = problems[(mode, seed)]
        X = genotype_from_pool(subset, problem.support)
        base = problem.evaluate_details(X)
        bases[mode] = base
        lengths[mode] = len(subset)
        records.append(align_X(base["records"], problem.all_names))
    n_union = sum(lengths.values())
    aligned_union = np.vstack(records)
    rng = np.random.default_rng(audit_seed)
    # Shared exposure draws for the seed; only diffusivity residual structure varies.
    cs, cc = sample_cs(mc_n, rng), sample_ccrit(mc_n, rng)
    feat_dist = None
    if structure == "distance_correlated":
        feat_dist = pairwise_feature_distance(aligned_union, train_scaler)
    z_union = make_residuals(structure, n_union, mc_n, rng, feat_dist)

    scored: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    for mode in modes:
        n = lengths[mode]
        z = z_union[offset : offset + n]
        offset += n
        base = bases[mode]
        problem = problems[(mode, seed)]
        logd = base["mu"][:, None] + problem.lam * base["sigma"][:, None] * z
        d = 10**logd * 1e-12
        te = t_eq(t_init_years(d, base["m"][:, None], cs, cc))
        gwp = np.asarray([impact_from_mix(m, GWP) for m in base["mixes"]], dtype=float)
        b = gwp[:, None] / np.maximum(te, 1e-12)
        j = gwp[:, None] * (d / 1e-12)
        scored[mode] = {"b": b, "j": j}
    return scored


def run_structure(
    pools: pd.DataFrame,
    problems: dict[tuple[str, int], ReconstructionProblem],
    structure: str,
    mc_n: int,
    train_scaler: StandardScaler,
    seeds: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for seed in seeds:
        audit_seed = 91_000 + int(seed)
        subsets = {}
        for mode in ("mean", "q10"):
            subset = pools[
                (pools["pool"] == mode) & (pools["optimization_seed"] == seed)
            ].sort_values("local_id").reset_index(drop=True)
            if len(subset) < 2:
                subsets = {}
                break
            subsets[mode] = subset
        if len(subsets) < 2:
            continue
        scored = score_union_arm(
            subsets, problems, int(seed), structure, mc_n, audit_seed, train_scaler,
        )
        base_b = union_draw_baseline(scored["mean"]["b"], scored["q10"]["b"])
        base_j = union_draw_baseline(scored["mean"]["j"], scored["q10"]["j"])
        for mode in ("mean", "q10"):
            summary = summarize_scores(
                scored[mode]["b"], scored[mode]["j"], base_b, base_j, mode, seed,
            )
            summary["production_lambda_used"] = problems[(mode, seed)].lam
            summary["S3"]["residual_structure"] = structure
            summary["S3"]["common_random_numbers"] = structure == "shared"
            summary["S3"]["union_residual_draw"] = True
            flat = flatten_seed_summary(summary)
            flat["residual_structure"] = structure
            rows.append(flat)
    seedwise = pd.DataFrame(rows)
    if not len(seedwise):
        raise RuntimeError(f"no seedwise rows for structure={structure}")
    head = {
        "residual_structure": structure,
        "category_b_regret": paired_metric_inference(
            seedwise, "s3_expected_category_b_regret"
        ),
        "category_b_membership_failure": paired_metric_inference(
            seedwise, "s3_top10_failure_category_b"
        ),
        "J_regret": paired_metric_inference(seedwise, "s3_expected_J_regret"),
        "J_membership_failure": paired_metric_inference(seedwise, "s3_top10_failure"),
        "seedwise_descriptive": aggregate_seedwise(seedwise).get("seedwise_descriptive"),
        "union_residual_draw": True,
    }
    return seedwise, head


def build_problems(
    pools: pd.DataFrame,
    ckpt: dict[str, Any],
    all_names: list[str],
    support: dict[str, Any],
    scaler: StandardScaler,
    knn: NearestNeighbors,
    ood_limit: float,
    m_mk: float,
    lam: float,
    selected_n: int,
) -> dict[tuple[str, int], ReconstructionProblem]:
    problems = {}
    for seed in sorted(pools["optimization_seed"].unique()):
        for mode in ("mean", "q10"):
            problems[(mode, int(seed))] = ReconstructionProblem(
                mode, selected_n, int(seed), ckpt, all_names, support,
                scaler, knn, ood_limit, m_mk, lam,
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    args = parser.parse_args()
    write_specification()

    pools_path = EXPORTS / "rev2_matched_nsga2_pools.csv"
    if not pools_path.exists():
        raise FileNotFoundError(f"missing deposited fronts: {pools_path}")
    pools = pd.read_csv(pools_path)
    if "pool" not in pools.columns or not len(pools):
        raise RuntimeError("pools file is empty or incomplete; finish the re-run first")

    production = joblib.load(EXPORTS / "rev2_production_ngb.joblib")
    ckpt = {
        "ensemble": production["model"],
        "preprocess_pipeline": production["preprocessing"],
        "all_feature_names": production["all_feature_names"],
    }
    all_names = list(ckpt["all_feature_names"])
    raw = pd.read_csv(DATA_PATH)
    training = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    support = derive_support(training)
    m_mk = mk_mean_from_workbook()
    lam, _, _ = load_lambda()
    X_train = align_X(training, all_names)
    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    k = max(1, min(SPEC.ood_knn_neighbors, len(training) - 1))
    self_dist = NearestNeighbors(n_neighbors=k + 1).fit(Z_train).kneighbors(Z_train)[0][:, k]
    ood_limit = float(np.quantile(self_dist, SPEC.ood_training_quantile))
    knn = NearestNeighbors(n_neighbors=k).fit(Z_train)
    support["ood"] = {
        "neighbor_rank": k, "distance_limit": ood_limit, "active_constraint": True,
    }

    saa_path = EXPORTS / "rev2_saa_convergence.csv"
    selected_n = (
        int(pd.read_csv(saa_path)["selected_production_n"].iloc[0])
        if saa_path.exists() else SPEC.saa_production_n
    )
    seeds = sorted(int(s) for s in pools["optimization_seed"].unique())
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    mc_n = args.mc or SPEC.mc_draws
    problems = build_problems(
        pools, ckpt, all_names, support, scaler, knn, ood_limit, m_mk, lam, selected_n,
    )

    structures = ("shared", "independent", "distance_correlated")
    seedwise_frames = []
    heads = {}
    for structure in structures:
        seedwise, head = run_structure(
            pools, problems, structure, mc_n, scaler, seeds,
        )
        seedwise_frames.append(seedwise)
        heads[structure] = head
        print(
            f"{structure}: CatB regret rel red "
            f"{head['category_b_regret']['relative_mean_reduction']:.3f}; "
            f"membership "
            f"{head['category_b_membership_failure']['relative_mean_reduction']:.3f}"
        )

    all_seedwise = pd.concat(seedwise_frames, ignore_index=True)
    all_seedwise.to_csv(EXPORTS / "rev2_dependence_sensitivity_seedwise.csv", index=False)
    payload = {
        "purpose": (
            "Post-hoc sensitivity of Category B / J expected-regret and membership "
            "contrasts to cross-candidate residual dependence on deposited fronts."
        ),
        "primary_claim_remains": "shared",
        "mc_draws": mc_n,
        "n_seeds": len(seeds),
        "structures": heads,
        "distance_correlation": (
            "corr_ij = exp(-d_ij / median_pairwise_d) in standardized feature space"
        ),
    }
    write_json(EXPORTS / "rev2_dependence_sensitivity.json", payload)
    print(f"wrote rev2_dependence_sensitivity.json ({len(seeds)} seeds, mc={mc_n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
