#!/usr/bin/env python3
"""Record-level within-domain validation using the frozen pipeline.

Interpolation benchmark only (not a transfer claim). Production LOSO remains
R2=0.316. The checkpoint filename R2 is not reused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_loso_validation import (  # noqa: E402
    DATA_PATH,
    EXPORTS,
    NGB_PATH,
    SEED,
    aligned_design,
    fit_nll_scale,
    jsonable,
    metric_row,
)
from sklearn.base import clone

OUT = EXPORTS / "rev2_within_domain_split.json"


def main() -> int:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    frozen = joblib.load(NGB_PATH)
    all_feature_names = list(frozen["all_feature_names"])
    x = aligned_design(df, all_feature_names)
    y = np.log10(np.maximum(df["Da"].to_numpy(dtype=float), 1e-12))
    groups = df["source_id"].to_numpy(dtype=int)

    # Stratify on source_id where possible so each source appears in both splits.
    # Sources with a single record cannot be stratified; fall back to unstratified.
    counts = pd.Series(groups).value_counts()
    can_stratify = (counts.min() >= 2) and (counts >= 2).all()
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.20, random_state=SEED,
        stratify=groups if can_stratify else None,
    )
    # Hold out 20% of training sources' records as a calibration set for NLL scale
    # (record-level, not source-disjoint — this is the interpolation regime).
    train_only, cal_idx = train_test_split(
        train_idx, test_size=0.20, random_state=SEED + 1,
    )

    preprocessing = clone(frozen["preprocess_pipeline"])
    np.random.seed(SEED)
    x_train = preprocessing.fit_transform(x[train_only], y[train_only])
    x_cal = preprocessing.transform(x[cal_idx])
    x_test = preprocessing.transform(x[test_idx])

    model = clone(frozen["ensemble"])
    model.set_params(random_state=np.random.RandomState(SEED), verbose=False)
    model.fit(x_train, y[train_only])

    mu_train = np.asarray(model.predict(x_train), dtype=float)
    sig_train = np.maximum(
        np.asarray(model.pred_dist(x_train).scale, dtype=float).ravel(), 1e-6,
    )
    mu_cal = np.asarray(model.predict(x_cal), dtype=float)
    sig_cal = np.maximum(
        np.asarray(model.pred_dist(x_cal).scale, dtype=float).ravel(), 1e-6,
    )
    mu_test = np.asarray(model.predict(x_test), dtype=float)
    sig_test = np.maximum(
        np.asarray(model.pred_dist(x_test).scale, dtype=float).ravel(), 1e-6,
    )
    scale = fit_nll_scale(y[cal_idx], mu_cal, sig_cal)
    raw = metric_row(y[test_idx], mu_test, sig_test, "raw")
    calibrated = metric_row(y[test_idx], mu_test, scale * sig_test, "calibrated")

    # Fitted-data and calibration-partition point accuracy, so the optimism gap
    # between fitted and held-out records is auditable alongside the transfer gap.
    train_fit = metric_row(y[train_only], mu_train, sig_train, "train")
    cal_fit = metric_row(y[cal_idx], mu_cal, sig_cal, "cal")

    def point_only(row: dict) -> dict:
        return {k: row.get(k) for k in ("r2", "rmse", "mae")}

    # Also report the frozen production lambda consequence without re-selecting it.
    cal_path = EXPORTS / "rev2_production_calibration.json"
    production_lambda = float(
        json.loads(cal_path.read_text(encoding="utf-8"))["production_lambda"]
    )
    prod = metric_row(
        y[test_idx], mu_test, production_lambda * sig_test, "production_lambda",
    )

    summary = {
        "regime": "record_level_within_domain_interpolation",
        "interpretation": (
            "Within-domain interpolation benchmark using the exact frozen "
            "preprocess+NGB architecture. Not a transfer claim."
        ),
        "n_total": int(len(df)),
        "n_train": int(len(train_only)),
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "test_fraction": 0.20,
        "stratified_by_source": bool(can_stratify),
        "seed": SEED,
        "record_level_nll_scale": scale,
        "frozen_production_lambda": production_lambda,
        "train_fit_point": point_only(train_fit),
        "calibration_partition_point": point_only(cal_fit),
        "optimism_gap_r2_train_minus_test": (
            float(train_fit["r2"]) - float(raw["r2"])
        ),
        "raw": {
            "r2": raw.get("r2"), "rmse": raw.get("rmse"), "mae": raw.get("mae"),
            "crps": raw.get("crps"), "nll": raw.get("nll"),
            "mean_sigma": raw.get("mean_sigma"),
            **{k: raw[k] for k in raw if k.startswith("picp_")},
        },
        "calibrated_record_level": {
            "r2": calibrated.get("r2"), "rmse": calibrated.get("rmse"),
            "mae": calibrated.get("mae"), "crps": calibrated.get("crps"),
            "nll": calibrated.get("nll"), "mean_sigma": calibrated.get("mean_sigma"),
            **{k: calibrated[k] for k in calibrated if k.startswith("picp_")},
        },
        "production_lambda_on_same_split": {
            "r2": prod.get("r2"), "rmse": prod.get("rmse"), "mae": prod.get("mae"),
            "crps": prod.get("crps"), "nll": prod.get("nll"),
            "mean_sigma": prod.get("mean_sigma"),
            **{k: prod[k] for k in prod if k.startswith("picp_")},
        },
        "headline_r2": raw.get("r2"),
        "do_not_confuse_with": {
            "loso_25_source_r2": 0.316,
            "dominant_source_r2": 0.056,
            "historical_checkpoint_filename_r2": 0.8397,
        },
    }
    OUT.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
