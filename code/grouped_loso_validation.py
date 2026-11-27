#!/usr/bin/env python3
"""Source-disjoint grouped validation of the production ASTM C1556 model.

The frozen production checkpoint is used only as an architecture/parameter
template.  Every LOSO fold refits preprocessing and the probabilistic learner
on training sources; Gaussian scale and conformal q90 are fit on distinct
calibration sources.  Held-out sources are never used for fitting or
calibration.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)

REV = Path(__file__).resolve().parents[1]
EXPORTS = REV / "outputs"
DATA_PATH = REV / "data" / "astm_with_source.csv"
NGB_PATH = REV / "data" / "ngb_checkpoint.pkl"
SEED = 42
LEVELS = (0.50, 0.80, 0.90, 0.95)
COVERAGE_GRID = tuple(np.round(np.arange(0.05, 1.00, 0.05), 2))
MIN_SIGMA = 1e-6


def jsonable(value: Any) -> Any:
    """Convert numpy/non-finite values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def create_advanced_features(df_subset: pd.DataFrame) -> pd.DataFrame:
    """Mirror the production feature construction without source leakage."""
    df_eng = df_subset.copy()
    drop = [c for c in df_eng if c.startswith("Has_")]
    drop += [c for c in ("SCM_Count",) if c in df_eng]
    df_eng = df_eng.drop(columns=drop, errors="ignore")
    if {"NFA", "NCA"}.issubset(df_eng):
        total_agg = df_eng["NFA"] + df_eng["NCA"]
        df_eng["Fine_Aggregate_Ratio"] = df_eng["NFA"] / (total_agg + 1e-6)
        df_eng["Coarse_Aggregate_Ratio"] = df_eng["NCA"] / (total_agg + 1e-6)
        df_eng["Aggregate_Total"] = total_agg
    if {"Total_Binder", "Aggregate_Total"}.issubset(df_eng):
        df_eng["Binder_Aggregate_Ratio"] = (
            df_eng["Total_Binder"] / (df_eng["Aggregate_Total"] + 1e-6)
        )
    if {"Water", "Total_Binder"}.issubset(df_eng):
        df_eng["Paste_Content"] = df_eng["Water"] + df_eng["Total_Binder"]
        if "Aggregate_Total" in df_eng:
            total_mass = df_eng["Paste_Content"] + df_eng["Aggregate_Total"]
            df_eng["Paste_Volume_Ratio"] = df_eng["Paste_Content"] / (total_mass + 1e-6)
    pairs = (
        ("w_b_ratio", "SCM_Replacement_Ratio", "w_b_x_SCM_Replacement"),
        ("w_b_ratio", "Exposure_Duration", "w_b_x_Exposure"),
        ("SCM_Replacement_Ratio", "Exposure_Duration", "SCM_x_Exposure"),
        ("CS", "w_b_ratio", "CS_x_w_b"),
        ("Binder_Aggregate_Ratio", "Fine_Aggregate_Ratio", "Binder_x_FineAgg"),
        ("Paste_Volume_Ratio", "w_b_ratio", "Paste_x_w_b"),
        ("CS", "Exposure_Duration", "CS_x_Exposure"),
        ("w_b_ratio", "Fine_Aggregate_Ratio", "w_b_x_FineAgg"),
        ("SCM_Replacement_Ratio", "Binder_Aggregate_Ratio", "SCM_x_BinderAgg"),
        ("Paste_Volume_Ratio", "Fine_Aggregate_Ratio", "Paste_x_FineAgg"),
    )
    for left, right, name in pairs:
        if left in df_eng and right in df_eng:
            df_eng[name] = df_eng[left] * df_eng[right]
    if {"CS", "w_b_ratio", "SCM_Replacement_Ratio"}.issubset(df_eng):
        df_eng["CS_x_w_b_x_SCM"] = (
            df_eng["CS"] * df_eng["w_b_ratio"] * df_eng["SCM_Replacement_Ratio"]
        )
    for feature in ("w_b_ratio", "SCM_Replacement_Ratio", "Exposure_Duration"):
        if feature in df_eng:
            df_eng[f"{feature}_squared"] = df_eng[feature] ** 2
            # Production logic creates cubics only for variables whose domain is <10.
            if feature != "Exposure_Duration":
                df_eng[f"{feature}_cubed"] = df_eng[feature] ** 3
    non_features = {
        "Da", "Test_Method", "Test_Method_Encoded", "Reference", "source_id",
        "source_label", "Index",
    }
    return df_eng.drop(
        columns=[c for c in df_eng if c in non_features or c.startswith("Unnamed")],
        errors="ignore",
    )


def aligned_design(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    engineered = create_advanced_features(df)
    for name in feature_names:
        if name not in engineered:
            engineered[name] = 0.0
    return (
        engineered[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(np.asarray(sigma, dtype=float), MIN_SIGMA)
    z = (np.asarray(y) - np.asarray(mu)) / sigma
    return float(np.mean(0.5 * np.log(2.0 * np.pi) + np.log(sigma) + 0.5 * z**2))


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(np.asarray(sigma, dtype=float), MIN_SIGMA)
    z = (np.asarray(y) - np.asarray(mu)) / sigma
    values = sigma * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(values))


def fit_nll_scale(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form positive scale minimizing Gaussian NLL on calibration data."""
    standardized_sq = ((y - mu) / np.maximum(sigma, MIN_SIGMA)) ** 2
    return float(max(np.sqrt(np.mean(standardized_sq)), MIN_SIGMA))


def conformal_q90(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    scores = np.abs(y - mu) / np.maximum(sigma, MIN_SIGMA)
    n = len(scores)
    probability = min(1.0, math.ceil((n + 1) * 0.90) / n)
    return float(np.quantile(scores, probability, method="higher"))


def safe_r2(y: np.ndarray, mu: np.ndarray) -> float:
    return float(r2_score(y, mu)) if len(y) >= 2 and np.ptp(y) > 0 else float("nan")


def metric_row(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, stage: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": stage,
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, mu))),
        "mae": float(mean_absolute_error(y, mu)),
        "r2": safe_r2(y, mu),
        "crps": gaussian_crps(y, mu, sigma),
        "nll": gaussian_nll(y, mu, sigma),
        "mean_sigma": float(np.mean(sigma)),
    }
    for level in LEVELS:
        z = stats.norm.ppf(0.5 + level / 2.0)
        result[f"picp_{int(level * 100)}"] = float(
            np.mean(np.abs(y - mu) <= z * sigma)
        )
        result[f"mean_width_{int(level * 100)}"] = float(np.mean(2.0 * z * sigma))
    return result


def calibration_split(
    candidate_idx: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups[candidate_idx])
    if len(unique_groups) < 2:
        raise ValueError("At least two non-test sources are required for train/calibration.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    local_train, local_cal = next(
        splitter.split(candidate_idx, groups=groups[candidate_idx])
    )
    return candidate_idx[local_train], candidate_idx[local_cal]


def fit_fold_model(
    frozen: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str | None]:
    """Fit cloned production NGB, with a documented closest-model fallback."""
    preprocessing = clone(frozen["preprocess_pipeline"])
    np.random.seed(seed)  # stabilizes mutual_info_regression used by SelectKBest
    x_train_p = preprocessing.fit_transform(x_train, y_train)
    x_cal_p = preprocessing.transform(x_cal)
    x_test_p = preprocessing.transform(x_test)
    try:
        model = clone(frozen["ensemble"])
        # This archived ngboost version calls ``choice`` directly on its
        # random_state, so preserve the checkpoint's RandomState interface.
        params: dict[str, Any] = {
            "random_state": np.random.RandomState(seed),
            "verbose": False,
        }
        if smoke:
            params["n_estimators"] = min(40, int(model.get_params()["n_estimators"]))
        model.set_params(**params)
        model.fit(x_train_p, y_train)
        mu_cal = np.asarray(model.predict(x_cal_p), dtype=float)
        mu_test = np.asarray(model.predict(x_test_p), dtype=float)
        sig_cal = np.maximum(
            np.asarray(model.pred_dist(x_cal_p).scale, dtype=float).ravel(), MIN_SIGMA
        )
        sig_test = np.maximum(
            np.asarray(model.pred_dist(x_test_p).scale, dtype=float).ravel(), MIN_SIGMA
        )
        return mu_cal, sig_cal, mu_test, sig_test, "NGBRegressor", None
    except Exception as exc:
        # Closest available probabilistic surrogate: same fold-local preprocessing,
        # boosted conditional mean, and a training-residual homoscedastic Gaussian.
        fallback = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=float(frozen["ensemble"].get_params()["learning_rate"]),
            max_iter=int(frozen["ensemble"].get_params()["n_estimators"]),
            random_state=seed,
        )
        fallback.fit(x_train_p, y_train)
        mu_train = fallback.predict(x_train_p)
        train_sigma = max(float(np.sqrt(np.mean((y_train - mu_train) ** 2))), MIN_SIGMA)
        mu_cal = fallback.predict(x_cal_p)
        mu_test = fallback.predict(x_test_p)
        return (
            np.asarray(mu_cal),
            np.full(len(mu_cal), train_sigma),
            np.asarray(mu_test),
            np.full(len(mu_test), train_sigma),
            "HistGradientBoostingRegressor_homoscedastic_Gaussian",
            f"{type(exc).__name__}: {exc}",
        )


def fit_and_save_production_model(
    frozen: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    all_feature_names: list[str],
    production_lambda: float,
    validation_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Fit the production architecture on all records, save, reload, and verify."""
    preprocessing = clone(frozen["preprocess_pipeline"])
    np.random.seed(SEED)
    x_production = preprocessing.fit_transform(x, y)
    model = clone(frozen["ensemble"])
    model.set_params(random_state=np.random.RandomState(SEED), verbose=False)
    model.fit(x_production, y)
    artifact_path = EXPORTS / "rev2_production_ngb.joblib"
    artifact = {
        "model": model,
        "preprocessing": preprocessing,
        "all_feature_names": all_feature_names,
        "production_lambda": float(production_lambda),
        "reference_age_days": 35.0,
        "validation_provenance": validation_provenance,
    }
    joblib.dump(artifact, artifact_path)

    reloaded = joblib.load(artifact_path)
    check_x = reloaded["preprocessing"].transform(x[: min(10, len(x))])
    check_mu = np.asarray(reloaded["model"].predict(check_x), dtype=float)
    check_sigma = np.asarray(
        reloaded["model"].pred_dist(check_x).scale, dtype=float
    ).ravel()
    required_keys = {
        "model", "preprocessing", "all_feature_names", "production_lambda",
        "reference_age_days", "validation_provenance",
    }
    if not required_keys.issubset(reloaded):
        raise RuntimeError("Reloaded production artifact is missing required keys.")
    if not np.isfinite(check_mu).all() or not np.isfinite(check_sigma).all():
        raise RuntimeError("Reloaded production artifact returned non-finite predictions.")
    return {
        "path": str(artifact_path),
        "reload_verified": True,
        "verification_n": int(len(check_mu)),
        "verification_mu_min": float(check_mu.min()),
        "verification_mu_max": float(check_mu.max()),
        "verification_sigma_min": float(check_sigma.min()),
        "verification_sigma_max": float(check_sigma.max()),
        "required_keys_verified": sorted(required_keys),
    }


def duration_audit(
    fold: int,
    held_source: int,
    split_name: str,
    idx: np.ndarray,
    durations: np.ndarray,
    train_min: float,
    train_max: float,
) -> dict[str, Any]:
    values = durations[idx]
    return {
        "fold": fold,
        "held_source_id": held_source,
        "split": split_name,
        "n": int(len(values)),
        "duration_min_days": float(np.min(values)),
        "duration_q25_days": float(np.quantile(values, 0.25)),
        "duration_median_days": float(np.median(values)),
        "duration_q75_days": float(np.quantile(values, 0.75)),
        "duration_max_days": float(np.max(values)),
        "n_below_train_range": int(np.sum(values < train_min)),
        "n_above_train_range": int(np.sum(values > train_max)),
        "fraction_outside_train_range": float(
            np.mean((values < train_min) | (values > train_max))
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run three representative LOSO folds with at most 40 NGB estimators.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional deterministic limit for diagnostic runs; default is all sources.",
    )
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    EXPORTS.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required labeled corpus not found: {DATA_PATH}")
    if not NGB_PATH.exists():
        raise FileNotFoundError(f"Frozen NGB checkpoint not found: {NGB_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"Da", "source_id", "source_label", "Exposure_Duration"}
    missing = required - set(df)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if df["source_id"].isna().any():
        raise ValueError("source_id contains missing values.")

    frozen = joblib.load(NGB_PATH)
    all_feature_names = list(frozen["all_feature_names"])
    x = aligned_design(df, all_feature_names)
    y = np.log10(np.maximum(df["Da"].to_numpy(dtype=float), 1e-12))
    groups = df["source_id"].to_numpy(dtype=int)
    durations = df["Exposure_Duration"].to_numpy(dtype=float)
    source_counts = df.groupby("source_id").size().sort_values(ascending=False)
    dominant_source = int(source_counts.index[0])
    all_sources = source_counts.index.to_numpy(dtype=int)

    if args.smoke:
        smallest = int(source_counts.index[-1])
        median_source = int(source_counts.index[len(source_counts) // 2])
        fold_sources = np.array(
            list(dict.fromkeys([dominant_source, median_source, smallest])), dtype=int
        )
    else:
        fold_sources = all_sources.copy()
    if args.max_folds is not None:
        fold_sources = fold_sources[: max(1, args.max_folds)]

    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    duration_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    fallback_errors: list[dict[str, Any]] = []

    for fold, held_source in enumerate(fold_sources):
        test_idx = np.where(groups == held_source)[0]
        candidate_idx = np.where(groups != held_source)[0]
        train_idx, cal_idx = calibration_split(candidate_idx, groups, SEED + fold)
        train_sources = set(groups[train_idx])
        cal_sources = set(groups[cal_idx])
        test_sources = set(groups[test_idx])
        if train_sources & cal_sources or train_sources & test_sources or cal_sources & test_sources:
            raise RuntimeError(f"Source leakage detected in fold {fold}.")

        mu_cal, sig_cal, mu_test, sig_test, model_type, error = fit_fold_model(
            frozen,
            x[train_idx],
            y[train_idx],
            x[cal_idx],
            x[test_idx],
            SEED + fold,
            args.smoke,
        )
        scale = fit_nll_scale(y[cal_idx], mu_cal, sig_cal)
        q90 = conformal_q90(y[cal_idx], mu_cal, sig_cal)
        sig_test_cal = scale * sig_test
        raw = metric_row(y[test_idx], mu_test, sig_test, "before_nll_calibration")
        calibrated = metric_row(
            y[test_idx], mu_test, sig_test_cal, "after_nll_calibration"
        )
        conformal_covered = np.abs(y[test_idx] - mu_test) <= q90 * sig_test

        reference = str(df.iloc[test_idx[0]]["source_label"])
        fold_row: dict[str, Any] = {
            "fold": fold,
            "held_source_id": int(held_source),
            "held_source_label": reference,
            "fold_type": (
                "dominant_source_holdout"
                if held_source == dominant_source
                else "LOSO"
            ),
            "n_train": int(len(train_idx)),
            "n_cal": int(len(cal_idx)),
            "n_test": int(len(test_idx)),
            "n_train_sources": int(len(train_sources)),
            "n_cal_sources": int(len(cal_sources)),
            "n_test_sources": 1,
            "model_type": model_type,
            "nll_sigma_scale": scale,
            "cal_nll_before": gaussian_nll(y[cal_idx], mu_cal, sig_cal),
            "cal_nll_after": gaussian_nll(y[cal_idx], mu_cal, scale * sig_cal),
            "conformal_q90": q90,
            "conformal_q90_picp_test": float(np.mean(conformal_covered)),
            "conformal_q90_mean_width_test": float(np.mean(2.0 * q90 * sig_test)),
        }
        fold_row.update({f"before_{k}": v for k, v in raw.items() if k != "stage"})
        fold_row.update(
            {f"after_{k}": v for k, v in calibrated.items() if k != "stage"}
        )
        fold_rows.append(fold_row)
        if error:
            fallback_errors.append(
                {"fold": fold, "held_source_id": int(held_source), "error": error}
            )

        split_rows.extend(
            {
                "fold": fold,
                "held_source_id": int(held_source),
                "source_id": int(source),
                "split": split_name,
                "n_records": int(np.sum(groups[idx] == source)),
            }
            for split_name, idx, sources in (
                ("train", train_idx, train_sources),
                ("calibration", cal_idx, cal_sources),
                ("test", test_idx, test_sources),
            )
            for source in sorted(sources)
        )
        train_min, train_max = float(durations[train_idx].min()), float(durations[train_idx].max())
        for split_name, idx in (
            ("train", train_idx),
            ("calibration", cal_idx),
            ("test", test_idx),
        ):
            duration_rows.append(
                duration_audit(
                    fold, int(held_source), split_name, idx, durations, train_min, train_max
                )
            )

        for local, row_idx in enumerate(test_idx):
            predictions.append(
                {
                    "row_index": int(row_idx),
                    "fold": fold,
                    "source_id": int(held_source),
                    "source_label": reference,
                    "is_dominant_source": bool(held_source == dominant_source),
                    "y_log10_Da": float(y[row_idx]),
                    "mu": float(mu_test[local]),
                    "sigma_raw": float(sig_test[local]),
                    "nll_sigma_scale": scale,
                    "sigma_calibrated": float(sig_test_cal[local]),
                    "conformal_q90": q90,
                    "conformal_q90_lower": float(mu_test[local] - q90 * sig_test[local]),
                    "conformal_q90_upper": float(mu_test[local] + q90 * sig_test[local]),
                    "Exposure_Duration": float(durations[row_idx]),
                }
            )
        for stage, sigma in (
            ("before_nll_calibration", sig_test),
            ("after_nll_calibration", sig_test_cal),
        ):
            for level in COVERAGE_GRID:
                z = stats.norm.ppf(0.5 + level / 2.0)
                coverage_rows.append(
                    {
                        "scope": "fold",
                        "fold": fold,
                        "held_source_id": int(held_source),
                        "stage": stage,
                        "nominal_coverage": level,
                        "empirical_coverage": float(
                            np.mean(np.abs(y[test_idx] - mu_test) <= z * sigma)
                        ),
                        "mean_interval_width": float(np.mean(2.0 * z * sigma)),
                    }
                )
        print(
            f"fold {fold + 1}/{len(fold_sources)} source={held_source} "
            f"n={len(test_idx)} model={model_type} scale={scale:.3f}"
        )

    pred_df = pd.DataFrame(predictions).sort_values("row_index")
    folds_df = pd.DataFrame(fold_rows)
    split_df = pd.DataFrame(split_rows)
    duration_df = pd.DataFrame(duration_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    y_pool = pred_df["y_log10_Da"].to_numpy()
    mu_pool = pred_df["mu"].to_numpy()
    sig_raw_pool = pred_df["sigma_raw"].to_numpy()
    sig_cal_pool = pred_df["sigma_calibrated"].to_numpy()
    pooled_before = metric_row(y_pool, mu_pool, sig_raw_pool, "before_nll_calibration")
    pooled_after = metric_row(y_pool, mu_pool, sig_cal_pool, "after_nll_calibration")
    production_lambda = fit_nll_scale(y_pool, mu_pool, sig_raw_pool)
    sig_production_pool = production_lambda * sig_raw_pool
    pooled_production = metric_row(
        y_pool,
        mu_pool,
        sig_production_pool,
        "after_global_oof_nll_calibration",
    )

    pooled_coverage = []
    for stage, sigma in (
        ("before_nll_calibration", sig_raw_pool),
        ("after_nll_calibration", sig_cal_pool),
        ("after_global_oof_nll_calibration", sig_production_pool),
    ):
        for level in COVERAGE_GRID:
            z = stats.norm.ppf(0.5 + level / 2.0)
            pooled_coverage.append(
                {
                    "scope": "pooled",
                    "fold": -1,
                    "held_source_id": -1,
                    "stage": stage,
                    "nominal_coverage": level,
                    "empirical_coverage": float(np.mean(np.abs(y_pool - mu_pool) <= z * sigma)),
                    "mean_interval_width": float(np.mean(2.0 * z * sigma)),
                }
            )
    coverage_df = pd.concat([coverage_df, pd.DataFrame(pooled_coverage)], ignore_index=True)

    conformal_summary = {
        "method": (
            "Fold-local normalized absolute residual q90, finite-sample 'higher' "
            "quantile, fitted on calibration sources only"
        ),
        "kept_separate_from_gaussian_sigma_calibration": True,
        "pooled_picp90": float(
            np.mean(
                np.abs(y_pool - mu_pool)
                <= pred_df["conformal_q90"].to_numpy() * sig_raw_pool
            )
        ),
        "pooled_mean_width90": float(
            np.mean(2.0 * pred_df["conformal_q90"].to_numpy() * sig_raw_pool)
        ),
    }
    ngb_params = frozen["ensemble"].get_params(deep=False)
    metadata = {
        "input": str(DATA_PATH),
        "checkpoint": str(NGB_PATH),
        "target": "log10(Da)",
        "n_records": int(len(df)),
        "n_sources": int(df["source_id"].nunique()),
        "dominant_source_id": dominant_source,
        "dominant_source_n": int(source_counts.iloc[0]),
        "run_mode": "smoke" if args.smoke else "full",
        "n_folds_run": int(len(fold_sources)),
        "architecture": "fold-local clone of frozen preprocess_pipeline and NGBRegressor",
        "frozen_ngb_parameters": {k: str(v) for k, v in ngb_params.items()},
        "sigma_calibration": (
            "Positive multiplicative scale minimizing Gaussian NLL on source-disjoint "
            "calibration groups; no PICP-based escalation"
        ),
        "model_types_used": sorted(folds_df["model_type"].unique().tolist()),
        "fallback_used": bool(fallback_errors),
        "fallback_errors": fallback_errors,
        "exact_limitation": (
            None
            if not fallback_errors
            else "NGB refit failed in listed folds; those folds use boosted means with "
            "homoscedastic Gaussian training-residual scale."
        ),
        "smoke_limitation": (
            "Smoke mode runs three sources and truncates NGB to 40 estimators; its "
            "metrics are diagnostic and must not be reported as full LOSO."
            if args.smoke
            else None
        ),
    }
    dominant_row = folds_df.loc[
        folds_df["held_source_id"] == dominant_source
    ].iloc[0].to_dict() if dominant_source in set(fold_sources) else None
    summary = {
        "method": "source-disjoint LOSO with nested source-disjoint calibration",
        "run_mode": metadata["run_mode"],
        "pooled_before": pooled_before,
        "pooled_after": pooled_after,
        "dominant_source_holdout": dominant_row,
        "conformal_q90": conformal_summary,
        "exposure_duration_support": {
            "pooled_test_fraction_outside_fold_train_range": float(
                np.average(
                    duration_df.loc[duration_df["split"] == "test", "fraction_outside_train_range"],
                    weights=duration_df.loc[duration_df["split"] == "test", "n"],
                )
            ),
            "folds_with_any_test_extrapolation": int(
                np.sum(
                    duration_df.loc[duration_df["split"] == "test", "fraction_outside_train_range"]
                    > 0
                )
            ),
        },
        "leakage_checks": {
            "source_disjoint_train_cal_test": True,
            "all_full_loso_rows_predicted_once": bool(
                not args.smoke
                and args.max_folds is None
                and len(pred_df) == len(df)
                and pred_df["row_index"].nunique() == len(df)
            ),
        },
        "metadata_file": "rev2_grouped_validation_metadata.json",
    }
    production_calibration = {
        "production_lambda": production_lambda,
        "lambda_definition": (
            "Single positive Gaussian sigma multiplier minimizing Gaussian NLL "
            "over all 540 source-disjoint raw LOSO OOF predictions"
        ),
        "calibration_scope": "all source-disjoint LOSO OOF predictions",
        "n_oof": int(len(y_pool)),
        "n_sources": int(df["source_id"].nunique()),
        "target": "log10(Da)",
        "no_picp_escalation": True,
        "metrics_before": pooled_before,
        "metrics_after_production_lambda": pooled_production,
        "fold_local_scales_reported_separately": True,
        "fold_local_metrics": pooled_after,
        "source_predictions": "rev2_grouped_validation_predictions.csv",
        "fold_metrics": "rev2_grouped_validation_folds.csv",
        "coverage_curves": "rev2_coverage_curves.csv",
    }
    summary["production_global_oof_calibration"] = production_calibration

    production_artifact: dict[str, Any] | None = None
    if not args.smoke and args.max_folds is None:
        validation_provenance = {
            "method": summary["method"],
            "n_oof": int(len(y_pool)),
            "n_sources": int(df["source_id"].nunique()),
            "production_lambda_source": "rev2_production_calibration.json",
            "predictions_file": "rev2_grouped_validation_predictions.csv",
            "folds_file": "rev2_grouped_validation_folds.csv",
            "summary_file": "rev2_grouped_validation_summary.json",
            "all_rows_predicted_once": summary["leakage_checks"][
                "all_full_loso_rows_predicted_once"
            ],
            "source_disjoint_train_cal_test": summary["leakage_checks"][
                "source_disjoint_train_cal_test"
            ],
        }
        production_artifact = fit_and_save_production_model(
            frozen,
            x,
            y,
            all_feature_names,
            production_lambda,
            validation_provenance,
        )
        production_calibration["production_artifact"] = production_artifact
        summary["production_artifact"] = production_artifact
    else:
        production_calibration["production_artifact"] = None
        production_calibration["artifact_note"] = (
            "Production artifact is written only by an unrestricted full run."
        )

    pred_df.to_csv(EXPORTS / "rev2_grouped_validation_predictions.csv", index=False)
    folds_df.to_csv(EXPORTS / "rev2_grouped_validation_folds.csv", index=False)
    split_df.to_csv(EXPORTS / "rev2_group_split_audit.csv", index=False)
    duration_df.to_csv(EXPORTS / "rev2_exposure_duration_support_audit.csv", index=False)
    coverage_df.to_csv(EXPORTS / "rev2_coverage_curves.csv", index=False)
    pd.DataFrame(
        [
            {"scope": "pooled", **conformal_summary},
            *[
                {
                    "scope": "fold",
                    "fold": row["fold"],
                    "held_source_id": row["held_source_id"],
                    "q90": row["conformal_q90"],
                    "picp90": row["conformal_q90_picp_test"],
                    "mean_width90": row["conformal_q90_mean_width_test"],
                }
                for row in fold_rows
            ],
        ]
    ).to_csv(EXPORTS / "rev2_conformal_q90.csv", index=False)
    (EXPORTS / "rev2_grouped_validation_metadata.json").write_text(
        json.dumps(jsonable(metadata), indent=2, allow_nan=False), encoding="utf-8"
    )
    (EXPORTS / "rev2_grouped_validation_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    (EXPORTS / "rev2_production_calibration.json").write_text(
        json.dumps(jsonable(production_calibration), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
