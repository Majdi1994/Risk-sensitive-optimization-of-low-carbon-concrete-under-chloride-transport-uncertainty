#!/usr/bin/env python3
"""Leave-one-source-out on the 24 non-dominant sources only.

The production lambda from the full 25-source out-of-fold calibration is frozen
and is not re-derived from this subset. This script writes only the rev2_loso24_*
artifacts. The pooled R² here is about 0.220. That is not the 25-source result
(about 0.316) and not the 18-source strength-provenance refit (about 0.010).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_loso_validation import (  # noqa: E402
    DATA_PATH,
    EXPORTS,
    LEVELS,
    MIN_SIGMA,
    NGB_PATH,
    SEED,
    aligned_design,
    calibration_split,
    conformal_q90,
    fit_fold_model,
    fit_nll_scale,
    gaussian_nll,
    jsonable,
    metric_row,
)

OUT_PRED = EXPORTS / "rev2_loso24_predictions.csv"
OUT_FOLDS = EXPORTS / "rev2_loso24_folds.csv"
OUT_SUMMARY = EXPORTS / "rev2_loso24_summary.json"


def main() -> int:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    frozen = joblib.load(NGB_PATH)
    all_feature_names = list(frozen["all_feature_names"])
    x = aligned_design(df, all_feature_names)
    y = np.log10(np.maximum(df["Da"].to_numpy(dtype=float), 1e-12))
    groups = df["source_id"].to_numpy(dtype=int)
    source_counts = df.groupby("source_id").size().sort_values(ascending=False)
    dominant_source = int(source_counts.index[0])

    # Drop the dominant source entirely from the analysis corpus.
    keep = groups != dominant_source
    df = df.loc[keep].reset_index(drop=True)
    x = x[keep]
    y = y[keep]
    groups = groups[keep]
    fold_sources = np.array(sorted(set(groups.tolist())), dtype=int)

    # Frozen production lambda from the full 25-source OOF calibration.
    cal_path = EXPORTS / "rev2_production_calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(f"Missing frozen production calibration: {cal_path}")
    production_lambda = float(json.loads(cal_path.read_text(encoding="utf-8"))["production_lambda"])

    predictions: list[dict] = []
    fold_rows: list[dict] = []
    for fold, held_source in enumerate(fold_sources):
        test_idx = np.where(groups == held_source)[0]
        candidate_idx = np.where(groups != held_source)[0]
        train_idx, cal_idx = calibration_split(candidate_idx, groups, SEED + fold)
        mu_cal, sig_cal, mu_test, sig_test, model_type, error = fit_fold_model(
            frozen, x[train_idx], y[train_idx], x[cal_idx], x[test_idx],
            SEED + fold, smoke=False,
        )
        if error:
            print(f"WARNING fold {fold}: fallback {error}")
        scale = fit_nll_scale(y[cal_idx], mu_cal, sig_cal)
        q90 = conformal_q90(y[cal_idx], mu_cal, sig_cal)
        sig_test_cal = scale * sig_test
        sig_test_prod = production_lambda * sig_test
        raw = metric_row(y[test_idx], mu_test, sig_test, "before_nll_calibration")
        fold_local = metric_row(y[test_idx], mu_test, sig_test_cal, "after_nll_calibration")
        prod = metric_row(
            y[test_idx], mu_test, sig_test_prod, "after_global_oof_nll_calibration",
        )
        label = str(df.iloc[test_idx[0]]["source_label"])
        fold_rows.append({
            "fold": fold,
            "held_source_id": int(held_source),
            "held_source_label": label,
            "n_train": int(len(train_idx)),
            "n_cal": int(len(cal_idx)),
            "n_test": int(len(test_idx)),
            "model_type": model_type,
            "fold_local_nll_scale": scale,
            "frozen_production_lambda": production_lambda,
            **{f"raw_{k}": v for k, v in raw.items() if k != "stage"},
            **{f"fold_local_{k}": v for k, v in fold_local.items() if k != "stage"},
            **{f"production_{k}": v for k, v in prod.items() if k != "stage"},
            "conformal_q90": q90,
        })
        for local, row_idx in enumerate(test_idx):
            predictions.append({
                "row_index": int(row_idx),
                "fold": fold,
                "source_id": int(held_source),
                "source_label": label,
                "y_log10_Da": float(y[row_idx]),
                "mu": float(mu_test[local]),
                "sigma_raw": float(sig_test[local]),
                "sigma_fold_local": float(sig_test_cal[local]),
                "sigma_production": float(sig_test_prod[local]),
            })
        print(
            f"loso24 fold {fold + 1}/{len(fold_sources)} source={held_source} "
            f"n={len(test_idx)} R2={raw['r2']:.3f}"
        )

    pred_df = pd.DataFrame(predictions)
    folds_df = pd.DataFrame(fold_rows)
    y_pool = pred_df["y_log10_Da"].to_numpy()
    mu_pool = pred_df["mu"].to_numpy()
    pooled_raw = metric_row(y_pool, mu_pool, pred_df["sigma_raw"].to_numpy(), "raw")
    pooled_fold = metric_row(
        y_pool, mu_pool, pred_df["sigma_fold_local"].to_numpy(), "fold_local",
    )
    pooled_prod = metric_row(
        y_pool, mu_pool, pred_df["sigma_production"].to_numpy(), "production",
    )
    summary = {
        "n_records": int(len(df)),
        "n_sources": int(len(fold_sources)),
        "excluded_dominant_source_id": dominant_source,
        "excluded_dominant_n": int(source_counts.loc[dominant_source]),
        "frozen_production_lambda": production_lambda,
        "lambda_policy": "frozen from full 25-source OOF NLL; not re-derived",
        "pooled_raw": pooled_raw,
        "pooled_fold_local_calibrated": pooled_fold,
        "pooled_production_lambda": pooled_prod,
        "picp90_raw": pooled_raw.get("picp_0.9"),
        "picp90_production": pooled_prod.get("picp_0.9"),
    }
    # metric_row may use different key names; normalize headline fields.
    for block_name, block in (
        ("pooled_raw", pooled_raw),
        ("pooled_fold_local_calibrated", pooled_fold),
        ("pooled_production_lambda", pooled_prod),
    ):
        summary[block_name] = {
            "r2": block.get("r2"),
            "rmse": block.get("rmse"),
            "mae": block.get("mae"),
            "crps": block.get("crps"),
            "nll": block.get("nll"),
            "mean_sigma": block.get("mean_sigma"),
            **{
                k: block[k] for k in block
                if k.startswith("picp_") or k.startswith("mean_width_")
            },
        }

    pred_df.to_csv(OUT_PRED, index=False)
    folds_df.to_csv(OUT_FOLDS, index=False)
    OUT_SUMMARY.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
