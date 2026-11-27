#!/usr/bin/env python3
"""Post-hoc risk-aware reranking of the mean arm as a third control.

Compares three shortlists on the deposited matched fronts without re-running
optimization:

  Arm A  mean search, shortlist ranked by E[GWP/t_eq]     (production)
  Arm B  q10  search, shortlist ranked by E[GWP/t_eq]     (production)
  Arm C  mean search, shortlist ranked by GWP/q10(t_eq)   (control)

All three shortlists are scored against the same matched-seed union baseline
under shared CRN draws. Recovered fraction of the search effect is
(mean - control) / (mean - q10).
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
from chloride_pipeline import DATA_PATH, GWP, impact_from_mix, mk_mean_from_workbook  # noqa: E402
from grouped_loso_validation import aligned_design  # noqa: E402
from analysis_specification import EXPORTS, SPEC  # noqa: E402
from matched_search import (  # noqa: E402
    ReconstructionProblem,
    derive_support,
    load_lambda,
    s3_draw_metrics,
    sample_ccrit,
    sample_cs,
    t_eq,
    t_init_years,
    union_draw_baseline,
)

TOP_K = SPEC.top_k


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    out = [0.0] * len(pvals)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = min(1.0, pvals[idx] * (len(pvals) - rank + 1))
        running = max(running, adj)
        out[idx] = running
    return out


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
    gwp = np.asarray([impact_from_mix(m, GWP) for m in base["mixes"]], dtype=float)
    logd = base["mu"][:, None] + problem.lam * base["sigma"][:, None] * z
    d = 10**logd * 1e-12
    ti = t_init_years(d, base["m"][:, None], cs, cc)
    te = t_eq(ti)
    b = gwp[:, None] / np.maximum(te, 1e-12)
    j = gwp[:, None] * (d / 1e-12)
    q10_te = np.quantile(te, 0.10, axis=1)
    return {
        "b": b,
        "j": j,
        "te": te,
        "gwp": gwp,
        "b_mean": b.mean(axis=1),
        "j_mean": j.mean(axis=1),
        "gwp_over_q10_te": gwp / np.maximum(q10_te, 1e-12),
        "j_q10": gwp * (np.quantile(d, 0.90, axis=1) / 1e-12),
    }


def shortlist_metrics(
    scores: np.ndarray,
    rank_key: np.ndarray,
    baseline: np.ndarray | None,
) -> dict[str, float]:
    """Return within/union failure, expected regret, and shortlist mean level."""
    fail, regret, _, _ = s3_draw_metrics(scores, TOP_K, baseline=baseline)
    n = scores.shape[1]
    idx = np.argsort(rank_key)[:TOP_K]
    level = float(np.mean(scores.mean(axis=1)[idx]))
    return {
        "fail": float(np.mean((fail / n)[idx])),
        "regret": float(np.mean((regret / n)[idx])),
        "level": level,
    }


def paired_summary(delta: np.ndarray, mean_pool: float) -> dict[str, Any]:
    rng = np.random.default_rng(20260801)
    boots = np.array([
        delta[rng.integers(0, len(delta), size=len(delta))].mean()
        for _ in range(10_000)
    ])
    nz = delta[np.abs(delta) > 0]
    p = 1.0 if len(nz) == 0 else float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
    return {
        "mean_delta": float(delta.mean()),
        "relative_change": float(delta.mean() / max(abs(mean_pool), 1e-12)),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "wilcoxon_p": p,
        "n_seeds_favouring_treatment": int((delta < 0).sum()),
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

    seeds = sorted(pools["optimization_seed"].unique().tolist())
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    mc_n = args.mc or SPEC.mc_draws
    saa_n = 50
    saa_path = EXPORTS / "rev2_saa_convergence.csv"
    if saa_path.exists():
        saa_n = int(pd.read_csv(saa_path)["selected_production_n"].iloc[0])

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        scores: dict[str, dict[str, np.ndarray]] = {}
        for mode in ("mean", "q10"):
            pool = pools[
                (pools["optimization_seed"] == seed) & (pools["pool"] == mode)
            ].reset_index(drop=True)
            problem = ReconstructionProblem(
                mode, saa_n, int(seed), ckpt, all_names, support, scaler, knn,
                ood_limit, m_mk, lam,
            )
            scores[mode] = score_arm(pool, problem, mc_n, audit_seed=int(seed))

        base_b = union_draw_baseline(scores["mean"]["b"], scores["q10"]["b"])
        base_j = union_draw_baseline(scores["mean"]["j"], scores["q10"]["j"])

        mean_b = shortlist_metrics(scores["mean"]["b"], scores["mean"]["b_mean"], base_b)
        q10_b = shortlist_metrics(scores["q10"]["b"], scores["q10"]["b_mean"], base_b)
        # Control: same mean-arm candidates, shortlisted by GWP/q10(t_eq).
        ctrl_b = shortlist_metrics(
            scores["mean"]["b"], scores["mean"]["gwp_over_q10_te"], base_b,
        )
        mean_j = shortlist_metrics(scores["mean"]["j"], scores["mean"]["j_mean"], base_j)
        q10_j = shortlist_metrics(scores["q10"]["j"], scores["q10"]["j_mean"], base_j)
        ctrl_j = shortlist_metrics(
            scores["mean"]["j"], scores["mean"]["j_q10"], base_j,
        )

        # Within-front membership for the control uses the same mean-arm scores.
        mean_b_local = shortlist_metrics(
            scores["mean"]["b"], scores["mean"]["b_mean"], None,
        )
        ctrl_b_local = shortlist_metrics(
            scores["mean"]["b"], scores["mean"]["gwp_over_q10_te"], None,
        )
        q10_b_local = shortlist_metrics(
            scores["q10"]["b"], scores["q10"]["b_mean"], None,
        )
        mean_j_local = shortlist_metrics(
            scores["mean"]["j"], scores["mean"]["j_mean"], None,
        )
        ctrl_j_local = shortlist_metrics(
            scores["mean"]["j"], scores["mean"]["j_q10"], None,
        )
        q10_j_local = shortlist_metrics(
            scores["q10"]["j"], scores["q10"]["j_mean"], None,
        )

        row = {
            "optimization_seed": int(seed),
            "mean_B_regret": mean_b["regret"],
            "q10_B_regret": q10_b["regret"],
            "ctrl_B_regret": ctrl_b["regret"],
            "mean_B_level": mean_b["level"],
            "q10_B_level": q10_b["level"],
            "ctrl_B_level": ctrl_b["level"],
            "mean_B_fail": mean_b_local["fail"],
            "q10_B_fail": q10_b_local["fail"],
            "ctrl_B_fail": ctrl_b_local["fail"],
            "mean_J_regret": mean_j["regret"],
            "q10_J_regret": q10_j["regret"],
            "ctrl_J_regret": ctrl_j["regret"],
            "mean_J_fail": mean_j_local["fail"],
            "q10_J_fail": q10_j_local["fail"],
            "ctrl_J_fail": ctrl_j_local["fail"],
        }
        rows.append(row)
        search_effect = mean_b["regret"] - q10_b["regret"]
        recovered = mean_b["regret"] - ctrl_b["regret"]
        frac = recovered / search_effect if abs(search_effect) > 1e-12 else float("nan")
        print(json.dumps({
            "seed": int(seed),
            "mean": round(mean_b["regret"], 4),
            "q10": round(q10_b["regret"], 4),
            "ctrl": round(ctrl_b["regret"], 4),
            "recovered_frac": round(frac, 3),
        }))

    frame = pd.DataFrame(rows)
    frame.to_csv(EXPORTS / "rev2_posthoc_rerank_control.csv", index=False)

    def contrast(a: str, b: str, label: str) -> dict[str, Any]:
        d = (frame[b] - frame[a]).to_numpy(dtype=float)
        return {
            "label": label,
            "arm_a": a,
            "arm_b": b,
            "mean_a": float(frame[a].mean()),
            "mean_b": float(frame[b].mean()),
            **paired_summary(d, float(frame[a].mean())),
        }

    contrasts = {
        "B_regret_q10_vs_mean": contrast("mean_B_regret", "q10_B_regret", "q10 search vs mean search"),
        "B_regret_ctrl_vs_mean": contrast("mean_B_regret", "ctrl_B_regret", "post-hoc rerank vs mean search"),
        "B_regret_q10_vs_ctrl": contrast("ctrl_B_regret", "q10_B_regret", "q10 search vs post-hoc rerank"),
        "J_regret_q10_vs_mean": contrast("mean_J_regret", "q10_J_regret", "q10 search vs mean search"),
        "J_regret_ctrl_vs_mean": contrast("mean_J_regret", "ctrl_J_regret", "post-hoc rerank vs mean search"),
        "B_fail_ctrl_vs_mean": contrast("mean_B_fail", "ctrl_B_fail", "membership: post-hoc vs mean"),
        "B_fail_q10_vs_mean": contrast("mean_B_fail", "q10_B_fail", "membership: q10 vs mean"),
    }
    pvals = [contrasts[k]["wilcoxon_p"] for k in contrasts]
    for name, hp in zip(contrasts, holm(pvals)):
        contrasts[name]["holm_p"] = hp

    search_effect = float(frame["mean_B_regret"].mean() - frame["q10_B_regret"].mean())
    recovered = float(frame["mean_B_regret"].mean() - frame["ctrl_B_regret"].mean())
    recovered_frac = recovered / search_effect if abs(search_effect) > 1e-12 else float("nan")
    seed_fracs = []
    for _, r in frame.iterrows():
        se = r["mean_B_regret"] - r["q10_B_regret"]
        re = r["mean_B_regret"] - r["ctrl_B_regret"]
        seed_fracs.append(re / se if abs(se) > 1e-12 else float("nan"))
    seed_fracs_arr = np.asarray(seed_fracs, dtype=float)

    out = {
        "question": (
            "Does risk-aware post-hoc reranking of the mean arm's archived "
            "candidates recover the Category B expected-regret reduction "
            "achieved by embedding q10 inside the search?"
        ),
        "arms": {
            "mean": "mean search, shortlist by E[GWP/t_eq] (production ranking)",
            "q10": "q10 search, shortlist by E[GWP/t_eq] (production ranking)",
            "control": "mean search, shortlist by GWP/q10(t_eq) (post-hoc rerank)",
        },
        "production_shortlist_already_uses": "E[GWP/t_eq] per-draw mean (not GWP/E[t_eq])",
        "regret_baseline": "matched_seed_union under shared CRN",
        "membership_reference": "within_front",
        "mc_draws": mc_n,
        "n_seeds": int(len(frame)),
        "post_hoc_rescoring_only": True,
        "shortlist_B_level": {
            "mean": float(frame["mean_B_level"].mean()),
            "q10": float(frame["q10_B_level"].mean()),
            "control": float(frame["ctrl_B_level"].mean()),
        },
        "category_B_regret": {
            "mean": float(frame["mean_B_regret"].mean()),
            "q10": float(frame["q10_B_regret"].mean()),
            "control": float(frame["ctrl_B_regret"].mean()),
            "search_effect": search_effect,
            "recovered_by_posthoc": recovered,
            "recovered_fraction": recovered_frac,
            "median_seed_recovered_fraction": float(np.nanmedian(seed_fracs_arr)),
        },
        "contrasts": contrasts,
        "runtime_seconds": time.perf_counter() - started,
    }
    (EXPORTS / "rev2_posthoc_rerank_control.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n_seeds": out["n_seeds"],
        "mean_B_regret": round(out["category_B_regret"]["mean"], 4),
        "q10_B_regret": round(out["category_B_regret"]["q10"], 4),
        "ctrl_B_regret": round(out["category_B_regret"]["control"], 4),
        "recovered_fraction": round(out["category_B_regret"]["recovered_fraction"], 4),
        "shortlist_B_level": {
            k: round(v, 3) for k, v in out["shortlist_B_level"].items()
        },
        "runtime_s": round(out["runtime_seconds"], 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
