#!/usr/bin/env python3
"""Post hoc applicability-domain support audit for the exported candidate pools.

The active gate screens each candidate on its distance to the single nearest
training neighbour against the 95th percentile of training 1-NN self-distances
(3.537).  A 1-NN rule can pass a synthetic mixture that sits next to one
isolated literature point, so this script also reports the fraction of the
4,800 terminal candidates that would satisfy a like-for-like 5-NN support
threshold.  No optimization is re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chloride_pipeline import align_X  # noqa: E402
from grouped_loso_validation import DATA_PATH, EXPORTS, aligned_design  # noqa: E402
from matched_search import feature_record  # noqa: E402

QUANTILE = 0.95


def _training_matrix(all_names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    raw = pd.read_csv(DATA_PATH)
    training = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    return aligned_design(training, all_names), training


def _candidate_matrix(pools: pd.DataFrame, all_names: list[str]) -> np.ndarray:
    records = []
    for row in pools.itertuples(index=False):
        mix = {
            "OPC": float(row.OPC),
            "FA": float(row.FA),
            "GGBFS": float(row.GGBFS),
            "SF": float(row.SF),
            "MK": float(row.MK),
            "Water": float(row.Water),
            "NFA": float(row.NFA),
            "NCA": float(row.NCA),
            "SP": float(row.SP),
        }
        records.append(feature_record(mix, cs_strength=float(row.strength)))
    # Candidate records are already numeric feature dictionaries; align_X is safe.
    return align_X(pd.DataFrame(records), all_names)


def main() -> int:
    production_path = EXPORTS / "rev2_production_ngb.joblib"
    pools_path = EXPORTS / "rev2_matched_nsga2_pools.csv"
    if not production_path.exists():
        raise FileNotFoundError(production_path)
    if not pools_path.exists():
        raise FileNotFoundError(pools_path)

    production = joblib.load(production_path)
    all_names = list(production["all_feature_names"])
    X_train, training = _training_matrix(all_names)
    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)

    pools = pd.read_csv(pools_path)
    X_cand = _candidate_matrix(pools, all_names)
    Z_cand = scaler.transform(X_cand)

    # Training self-distances: column 0 is the record itself.
    nn5 = NearestNeighbors(n_neighbors=6).fit(Z_train)
    train_d = nn5.kneighbors(Z_train, return_distance=True)[0]
    train_1nn = train_d[:, 1]
    train_5nn = train_d[:, 5]
    limit_1nn = float(np.quantile(train_1nn, QUANTILE))
    limit_5nn = float(np.quantile(train_5nn, QUANTILE))

    # Candidate distances into the training corpus.
    cand_1nn = NearestNeighbors(n_neighbors=1).fit(Z_train).kneighbors(
        Z_cand, return_distance=True
    )[0][:, 0]
    cand_5nn = NearestNeighbors(n_neighbors=5).fit(Z_train).kneighbors(
        Z_cand, return_distance=True
    )[0][:, 4]

    mk_positive = pools["MK"].astype(float).to_numpy() > 0.0
    pass_1nn = cand_1nn <= limit_1nn
    pass_5nn = cand_5nn <= limit_5nn

    # Consistency check against the distance archived during optimization
    # (production matched_search screens on the k-th neighbour, k=5).
    archived = pools["ood_knn_distance"].astype(float).to_numpy()
    max_abs_diff = float(np.max(np.abs(archived - cand_5nn)))

    out = {
        "question": (
            "What fraction of the 4,800 terminal candidates satisfy the production "
            "like-for-like 5-NN support threshold built as the 95th percentile "
            "of training 5-NN self-distances?"
        ),
        "n_training": int(len(training)),
        "n_candidates": int(len(pools)),
        "active_gate": {
            "neighbor_rank": 5,
            "threshold_quantile": QUANTILE,
            "threshold": limit_5nn,
            "threshold_rounded_in_paper": 3.537,
            "fraction_passing": float(pass_5nn.mean()),
            "n_passing": int(pass_5nn.sum()),
            "candidate_5nn_median": float(np.median(cand_5nn)),
            "candidate_5nn_p90": float(np.quantile(cand_5nn, 0.90)),
            "candidate_5nn_max": float(cand_5nn.max()),
            "mk_positive_fraction_among_passers": float(mk_positive[pass_5nn].mean())
            if pass_5nn.any()
            else None,
        },
        "post_hoc_1nn_support": {
            "neighbor_rank": 1,
            "threshold_quantile": QUANTILE,
            "threshold": limit_1nn,
            "fraction_passing": float(pass_1nn.mean()),
            "n_passing": int(pass_1nn.sum()),
            "candidate_1nn_max": float(cand_1nn.max()),
            "candidate_1nn_mean": float(cand_1nn.mean()),
            "mk_positive_fraction_among_passers": float(mk_positive[pass_1nn].mean())
            if pass_1nn.any()
            else None,
        },
        "mk_positive_fraction_all_candidates": float(mk_positive.mean()),
        "training_self_distance": {
            "1nn_p50": float(np.median(train_1nn)),
            "1nn_p95": limit_1nn,
            "5nn_p50": float(np.median(train_5nn)),
            "5nn_p95": limit_5nn,
        },
        "archived_5nn_consistency": {
            "max_abs_diff_vs_pool_ood_knn_distance": max_abs_diff,
            "agree_within_1e-6": bool(max_abs_diff < 1e-6),
        },
        "interpretation": (
            "The active production gate is 5-NN at the training 5-NN 95th percentile "
            "(SPEC.ood_knn_neighbors). The 1-NN fraction is a post hoc companion "
            "check on the same exported fronts and does not replace the active constraint."
        ),
    }
    # Prefer the exact archived limit from the feasibility support if present.
    support_path = EXPORTS / "rev2_feasibility_support.json"
    if support_path.exists():
        support = json.loads(support_path.read_text(encoding="utf-8"))
        archived_limit = support.get("ood", {}).get("distance_limit")
        if archived_limit is not None:
            out["active_gate"]["threshold"] = float(archived_limit)
            out["active_gate"]["threshold_source"] = "rev2_feasibility_support.json"
            pass_5nn = cand_5nn <= float(archived_limit)
            out["active_gate"]["fraction_passing"] = float(pass_5nn.mean())
            out["active_gate"]["n_passing"] = int(pass_5nn.sum())

    out_path = EXPORTS / "rev2_ood_threshold_audit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Also keep a clearly named twin so the package is self-documenting.
    (EXPORTS / "rev2_ood_support_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "active_5nn_pass_fraction": out["active_gate"]["fraction_passing"],
        "post_hoc_1nn_pass_fraction": out["post_hoc_1nn_support"]["fraction_passing"],
        "mk_positive_fraction": out["mk_positive_fraction_all_candidates"],
        "limit_5nn": out["active_gate"]["threshold"],
        "limit_1nn": out["post_hoc_1nn_support"]["threshold"],
        "wrote": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
