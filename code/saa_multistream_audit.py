#!/usr/bin/env python3
"""Multi-stream SAA stability audit for N in {50,100,250,500}.

Replicates the nested-CRN multimetric gates across independent residual streams.
Escalation: production N=100 rerun recommended only if p10 top-10 overlap fails
on a majority of streams. Writes rev2_saa_multistream_audit.json/.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_specification import EXPORTS, SPEC, write_specification  # noqa: E402
from matched_search import (  # noqa: E402
    DATA_PATH,
    ReconstructionProblem,
    align_X,
    derive_support,
    load_lambda,
    mk_mean_from_workbook,
    t_eq,
    t_init_years,
    write_json,
)

GRID = (50, 100, 250, 500)
GATES = {
    "median_kendall": 0.90,
    "p10_kendall": 0.80,
    "median_nmae": 0.05,
    "median_top10_overlap": 0.80,
    "p10_top10_overlap": 0.60,
}


def stream_nested_crn(seed: int, n: int, stream_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent CRN stream; stream 0 matches production nested_crn base."""
    from matched_search import sample_ccrit, sample_cs

    rng = np.random.default_rng(8_000_003 + int(seed) + 1_000_003 * int(stream_id))
    return rng.standard_normal(n), sample_cs(n, rng), sample_ccrit(n, rng)


def evaluate_stream(
    problem: ReconstructionProblem,
    X: np.ndarray,
    seeds: tuple[int, ...],
    stream_id: int,
    max_n: int,
    grid: tuple[int, ...],
) -> pd.DataFrame:
    base = problem.evaluate_details(X)
    rows = []
    top_k = 10
    for seed in seeds:
        z, cs, cc = stream_nested_crn(seed, max_n, stream_id)
        log_d = base["mu"][:, None] + problem.lam * base["sigma"][:, None] * z
        ti = t_eq(t_init_years(10**log_d * 1e-12, base["m"][:, None], cs, cc))
        ref = np.quantile(ti, 0.10, axis=1)
        ref_rank = stats.rankdata(-ref)
        ref_top = set(np.argsort(-ref)[:top_k])
        scale = max(float(np.ptp(ref)), 1e-9)
        for n in grid:
            q = np.quantile(ti[:, :n], 0.10, axis=1)
            top_overlap = len(ref_top & set(np.argsort(-q)[:top_k])) / top_k
            rows.append({
                "stream_id": stream_id,
                "seed": seed,
                "saa_n": n,
                "kendall_vs_max": stats.kendalltau(ref_rank, stats.rankdata(-q)).correlation,
                "nmae_vs_max": float(np.mean(np.abs(q - ref)) / scale),
                "top10_overlap_vs_max": top_overlap,
            })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby(["stream_id", "saa_n"], as_index=False).agg(
        median_kendall=("kendall_vs_max", "median"),
        p10_kendall=("kendall_vs_max", lambda x: float(x.quantile(0.10))),
        median_nmae=("nmae_vs_max", "median"),
        median_top10_overlap=("top10_overlap_vs_max", "median"),
        p10_top10_overlap=("top10_overlap_vs_max", lambda x: float(x.quantile(0.10))),
    )
    summary["pass_median_kendall"] = summary["median_kendall"] >= GATES["median_kendall"]
    summary["pass_p10_kendall"] = summary["p10_kendall"] >= GATES["p10_kendall"]
    summary["pass_median_nmae"] = summary["median_nmae"] <= GATES["median_nmae"]
    summary["pass_median_top10"] = summary["median_top10_overlap"] >= GATES["median_top10_overlap"]
    summary["pass_p10_top10"] = summary["p10_top10_overlap"] >= GATES["p10_top10_overlap"]
    summary["passes_all_gates"] = (
        summary["pass_median_kendall"]
        & summary["pass_p10_kendall"]
        & summary["pass_median_nmae"]
        & summary["pass_median_top10"]
        & summary["pass_p10_top10"]
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-streams", type=int, default=5)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    write_specification()

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
    support["ood"] = {"neighbor_rank": k, "distance_limit": ood_limit, "active_constraint": True}

    max_n = max(GRID)
    problem = ReconstructionProblem(
        "q10", max_n, SPEC.optimization_seeds[0], ckpt, all_names,
        support, scaler, knn, ood_limit, m_mk, lam,
    )
    rng = np.random.default_rng(20260730)
    n_probe = 48 if args.smoke else 256
    probe_X = rng.random((n_probe, 9))
    seeds = SPEC.optimization_seeds[:2] if args.smoke else SPEC.optimization_seeds
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    n_streams = 2 if args.smoke else args.n_streams

    frames = []
    for stream_id in range(n_streams):
        print(f"stream {stream_id + 1}/{n_streams} ...")
        frames.append(evaluate_stream(problem, probe_X, seeds, stream_id, max_n, GRID))
    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)

    n100 = summary[summary["saa_n"] == 100]
    n_fail = int((~n100["pass_p10_top10"]).sum())
    n_streams_eff = int(n100["stream_id"].nunique())
    repeatedly_fails = n_fail > (n_streams_eff / 2.0)
    escalate = bool(repeatedly_fails)

    payload = {
        "purpose": (
            "Multi-stream nested-CRN SAA multimetric audit; benchmark is N=max(grid)=500"
        ),
        "grid": list(GRID),
        "n_streams": n_streams_eff,
        "n_seeds": len(seeds),
        "gates": GATES,
        "benchmark": "N=500 nested prefix (stream-specific)",
        "production_n": SPEC.saa_production_n,
        "n100_p10_top10_by_stream": n100[["stream_id", "p10_top10_overlap", "pass_p10_top10"]].to_dict(
            orient="records"
        ),
        "n100_p10_fail_count": n_fail,
        "n100_repeatedly_fails_p10_gate": repeatedly_fails,
        "escalate_saa_production_rerun": escalate,
        "escalation_rule": (
            "Escalate production SAA rerun iff N=100 fails p10 top-10 overlap "
            "on a majority of independent streams"
        ),
        "stream_summaries": summary.to_dict(orient="records"),
    }
    detail.to_csv(EXPORTS / "rev2_saa_multistream_audit.csv", index=False)
    summary.to_csv(EXPORTS / "rev2_saa_multistream_summary.csv", index=False)
    write_json(EXPORTS / "rev2_saa_multistream_audit.json", payload)
    print(
        f"N=100 p10 fails on {n_fail}/{n_streams_eff} streams; "
        f"escalate={escalate}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
