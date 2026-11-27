#!/usr/bin/env python3
"""Post-hoc 5-NN support restriction of the deposited matched fronts.

Production OOD uses 5-NN at the training 95th percentile (SPEC.ood_knn_neighbors).
This script restricts each seed's fronts to the 5-NN passers and recomputes the
four paired S3 contrasts without re-running any optimization:

  - within-front top-10 membership failure on Category B and J
  - union-referenced expected ranking regret on Category B and J

Seeds that retain fewer than ten passers on either arm are reported and
skipped for that arm's shortlist summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chloride_pipeline import DATA_PATH, GWP, align_X, impact_from_mix, mk_mean_from_workbook  # noqa: E402
from grouped_loso_validation import aligned_design  # noqa: E402
from analysis_specification import EXPORTS, SPEC  # noqa: E402
from matched_search import (  # noqa: E402
    ReconstructionProblem,
    derive_support,
    feature_record,
    load_lambda,
    s3_draw_metrics,
    sample_ccrit,
    sample_cs,
    t_eq,
    t_init_years,
    union_draw_baseline,
)

TOP_K = SPEC.top_k
QUANTILE = 0.95


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    out = [0.0] * len(pvals)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = min(1.0, pvals[idx] * (len(pvals) - rank + 1))
        running = max(running, adj)
        out[idx] = running
    return out


def candidate_matrix(pools: pd.DataFrame, all_names: list[str]) -> np.ndarray:
    records = []
    for row in pools.itertuples(index=False):
        mix = {
            "OPC": float(row.OPC), "FA": float(row.FA), "GGBFS": float(row.GGBFS),
            "SF": float(row.SF), "MK": float(row.MK), "Water": float(row.Water),
            "NFA": float(row.NFA), "NCA": float(row.NCA), "SP": float(row.SP),
        }
        records.append(feature_record(mix, cs_strength=float(row.strength)))
    return align_X(pd.DataFrame(records), all_names)


def score_arm(
    pool: pd.DataFrame,
    problem: ReconstructionProblem,
    mc_n: int,
    audit_seed: int,
) -> dict[str, np.ndarray]:
    s = problem.support
    X = np.column_stack([
        (pool["w_b_ratio"] - s["w_b_min"]) / (s["w_b_max"] - s["w_b_min"]),
        pool["scm_replacement"] / SPEC.max_scm_total_ratio,
        (pool["binder_content"] - s["binder_min"]) / (s["binder_max"] - s["binder_min"]),
        (pool["fine_agg_ratio"] - 0.30) / 0.40,
        pool["fa_frac"], pool["slag_frac"], pool["sf_frac"], pool["mk_frac"],
        (pool["SP"] - s["sp_min"]) / (s["sp_max"] - s["sp_min"]),
    ])
    base = problem.evaluate_details(X)
    rng = np.random.default_rng(audit_seed)
    z = rng.standard_normal(mc_n)
    cs, cc = sample_cs(mc_n, rng), sample_ccrit(mc_n, rng)
    gwp = np.asarray([impact_from_mix(m, GWP) for m in base["mixes"]])
    logd = base["mu"][:, None] + problem.lam * base["sigma"][:, None] * z
    d = 10**logd * 1e-12
    ti = t_init_years(d, base["m"][:, None], cs, cc)
    te = t_eq(ti)
    b = gwp[:, None] / np.maximum(te, 1e-12)
    j = gwp[:, None] * (d / 1e-12)
    return {
        "b": b,
        "j": j,
        "b_mean": b.mean(axis=1),
        "j_mean": j.mean(axis=1),
    }


def front_metrics(
    scores: np.ndarray,
    mean_score: np.ndarray,
    baseline: np.ndarray | None,
) -> tuple[float, float]:
    """Return (within-front top-k failure, expected regret among point shortlist)."""
    if len(mean_score) < TOP_K:
        return float("nan"), float("nan")
    fail, regret, _, _ = s3_draw_metrics(scores, TOP_K, baseline=baseline)
    n = scores.shape[1]
    idx = np.argsort(mean_score)[:TOP_K]
    return float(np.mean((fail / n)[idx])), float(np.mean((regret / n)[idx]))


def paired_summary(delta: np.ndarray, mean_pool: float) -> dict[str, Any]:
    rng = np.random.default_rng(20260801)
    boots = np.array([
        delta[rng.integers(0, len(delta), size=len(delta))].mean()
        for _ in range(10_000)
    ])
    # Drop exact zeros for Wilcoxon; if all zero, report p=1.
    nz = delta[np.abs(delta) > 0]
    if len(nz) == 0:
        p = 1.0
    else:
        p = float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
    return {
        "mean_delta": float(delta.mean()),
        "relative_change": float(delta.mean() / max(abs(mean_pool), 1e-12)),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "wilcoxon_p": p,
        "n_seeds_favouring_q10": int((delta < 0).sum()),
        "n_seeds_used": int(len(delta)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    args = parser.parse_args()
    started = time.perf_counter()

    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    production = joblib.load(EXPORTS / "rev2_production_ngb.joblib")
    all_names = list(production["all_feature_names"])
    ckpt = {
        "ensemble": production["model"],
        "preprocess_pipeline": production["preprocessing"],
        "all_feature_names": all_names,
    }
    raw = pd.read_csv(DATA_PATH)
    training = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    support = derive_support(training)
    m_mk = mk_mean_from_workbook()
    lam, _, _ = load_lambda()
    X_train = aligned_design(training, all_names)
    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    k1 = max(1, min(SPEC.ood_knn_neighbors, len(training) - 1))
    self_1 = NearestNeighbors(n_neighbors=k1 + 1).fit(Z_train).kneighbors(
        Z_train, return_distance=True
    )[0][:, k1]
    ood_limit = float(np.quantile(self_1, SPEC.ood_training_quantile))
    knn = NearestNeighbors(n_neighbors=k1).fit(Z_train)

    # 5-NN like-for-like threshold on the full candidate table.
    train_5 = NearestNeighbors(n_neighbors=6).fit(Z_train).kneighbors(
        Z_train, return_distance=True
    )[0][:, 5]
    limit_5nn = float(np.quantile(train_5, QUANTILE))
    X_cand = candidate_matrix(pools, all_names)
    Z_cand = scaler.transform(X_cand)
    cand_5 = NearestNeighbors(n_neighbors=5).fit(Z_train).kneighbors(
        Z_cand, return_distance=True
    )[0][:, 4]
    pools = pools.copy()
    pools["pass_5nn"] = cand_5 <= limit_5nn

    seeds = sorted(pools["optimization_seed"].unique().tolist())
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    mc_n = args.mc or SPEC.mc_draws
    saa_n = 50
    saa_path = EXPORTS / "rev2_saa_convergence.csv"
    if saa_path.exists():
        saa_n = int(pd.read_csv(saa_path)["selected_production_n"].iloc[0])

    rows = []
    skipped = []
    for seed in seeds:
        scores = {}
        retained = {}
        for mode in ("mean", "q10"):
            pool_all = pools[
                (pools["optimization_seed"] == seed) & (pools["pool"] == mode)
            ].reset_index(drop=True)
            pool = pool_all[pool_all["pass_5nn"]].reset_index(drop=True)
            retained[mode] = int(len(pool))
            if len(pool) < TOP_K:
                skipped.append({"seed": int(seed), "pool": mode, "n_passers": int(len(pool))})
                scores[mode] = None
                continue
            problem = ReconstructionProblem(
                mode, saa_n, int(seed), ckpt, all_names, support, scaler, knn,
                ood_limit, m_mk, lam,
            )
            scores[mode] = score_arm(pool, problem, mc_n, audit_seed=int(seed))

        if scores["mean"] is None or scores["q10"] is None:
            continue

        base_b = union_draw_baseline(scores["mean"]["b"], scores["q10"]["b"])
        base_j = union_draw_baseline(scores["mean"]["j"], scores["q10"]["j"])
        row: dict[str, Any] = {
            "optimization_seed": int(seed),
            "mean_n_passers": retained["mean"],
            "q10_n_passers": retained["q10"],
        }
        for mode in ("mean", "q10"):
            fail_b, reg_b = front_metrics(
                scores[mode]["b"], scores[mode]["b_mean"], baseline=base_b,
            )
            fail_j, reg_j = front_metrics(
                scores[mode]["j"], scores[mode]["j_mean"], baseline=base_j,
            )
            # Membership is within the restricted front (arm-local among passers).
            fail_b_local, _ = front_metrics(
                scores[mode]["b"], scores[mode]["b_mean"], baseline=None,
            )
            fail_j_local, _ = front_metrics(
                scores[mode]["j"], scores[mode]["j_mean"], baseline=None,
            )
            row[f"{mode}_B_fail"] = fail_b_local
            row[f"{mode}_B_regret"] = reg_b
            row[f"{mode}_J_fail"] = fail_j_local
            row[f"{mode}_J_regret"] = reg_j
        rows.append(row)
        print(json.dumps({
            "seed": seed,
            "mean_n": retained["mean"],
            "q10_n": retained["q10"],
            "dB_fail": round(row["q10_B_fail"] - row["mean_B_fail"], 4),
            "dB_reg": round(row["q10_B_regret"] - row["mean_B_regret"], 4),
            "dJ_fail": round(row["q10_J_fail"] - row["mean_J_fail"], 4),
            "dJ_reg": round(row["q10_J_regret"] - row["mean_J_regret"], 4),
        }))

    frame = pd.DataFrame(rows)
    frame.to_csv(EXPORTS / "rev2_ood_threshold_sensitivity.csv", index=False)

    contrasts: dict[str, Any] = {}
    keys = [
        ("B_fail", "B_top10_failure"),
        ("B_regret", "B_expected_regret"),
        ("J_fail", "J_top10_failure"),
        ("J_regret", "J_expected_regret"),
    ]
    pvals = []
    for short, name in keys:
        mean_col = f"mean_{short}"
        q10_col = f"q10_{short}"
        d = (frame[q10_col] - frame[mean_col]).to_numpy(dtype=float)
        block = {
            "mean_pool": float(frame[mean_col].mean()),
            "q10_pool": float(frame[q10_col].mean()),
            **paired_summary(d, float(frame[mean_col].mean())),
        }
        contrasts[name] = block
        pvals.append(block["wilcoxon_p"])
    for name, hp in zip([k[1] for k in keys], holm(pvals)):
        contrasts[name]["holm_p"] = hp

    out = {
        "question": (
            "Do the four paired S3 contrasts survive when each arm's front is "
            "restricted post hoc to candidates that also pass the like-for-like "
            "5-NN support threshold?"
        ),
        "active_gate": "1-NN at training 95th percentile (unchanged)",
        "restriction": "post_hoc_5nn_passers_only",
        "limit_5nn": limit_5nn,
        "fraction_passers_global": float(pools["pass_5nn"].mean()),
        "n_passers_global": int(pools["pass_5nn"].sum()),
        "n_paired_seeds_used": int(len(frame)),
        "n_seeds_skipped_sparse": len(skipped),
        "skipped": skipped,
        "mc_draws": mc_n,
        "post_hoc_rescoring_only": True,
        "membership_reference": "within_restricted_front",
        "regret_baseline": "matched_seed_union_of_restricted_fronts",
        "contrasts": contrasts,
        "mean_passers_per_arm": {
            "mean": float(frame["mean_n_passers"].mean()) if len(frame) else None,
            "q10": float(frame["q10_n_passers"].mean()) if len(frame) else None,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (EXPORTS / "rev2_ood_threshold_sensitivity.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in {
            "n_seeds": out["n_paired_seeds_used"],
            "frac_pass": out["fraction_passers_global"],
            **{
                name: {
                    "rel": round(100 * contrasts[name]["relative_change"], 1),
                    "holm_p": contrasts[name]["holm_p"],
                }
                for name in contrasts
            },
        }.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
