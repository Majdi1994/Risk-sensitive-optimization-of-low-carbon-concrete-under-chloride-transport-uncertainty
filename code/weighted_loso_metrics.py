#!/usr/bin/env python3
"""Source- and composition-weighted LOSO metrics on deposited OOF predictions.

No NGB refit. Uses rev2_grouped_validation_predictions.csv joined to the ASTM
corpus for composition fingerprints. Writes rev2_weighted_loso_metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_specification import EXPORTS  # noqa: E402
from grouped_loso_validation import DATA_PATH, MIN_SIGMA, gaussian_crps, gaussian_nll  # noqa: E402

COMP = ["Cement", "SF", "GGBFS", "FA", "MK", "Water", "NFA", "NCA", "SP"]
Z90 = float(stats.norm.ppf(0.95))


def fingerprint(frame: pd.DataFrame) -> pd.Series:
    return frame[COMP].round(3).astype(str).agg("|".join, axis=1)


def picp90(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, MIN_SIGMA)
    return float(np.mean(np.abs(y - mu) <= Z90 * sigma))


def weighted_r2(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    ybar = float(np.sum(w * y))
    ss_res = float(np.sum(w * (y - mu) ** 2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def weighted_rmse(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    return float(np.sqrt(np.sum(w * (y - mu) ** 2)))


def weighted_mae(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    return float(np.sum(w * np.abs(y - mu)))


def weighted_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> float:
    sigma = np.maximum(sigma, MIN_SIGMA)
    w = w / w.sum()
    z = (y - mu) / sigma
    return float(np.sum(w * (0.5 * np.log(2.0 * np.pi) + np.log(sigma) + 0.5 * z**2)))


def weighted_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> float:
    sigma = np.maximum(sigma, MIN_SIGMA)
    w = w / w.sum()
    z = (y - mu) / sigma
    values = sigma * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.sum(w * values))


def weighted_picp90(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> float:
    sigma = np.maximum(sigma, MIN_SIGMA)
    w = w / w.sum()
    hit = (np.abs(y - mu) <= Z90 * sigma).astype(float)
    return float(np.sum(w * hit))


def score_block(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, w: np.ndarray | None) -> dict:
    if w is None:
        return {
            "n": int(len(y)),
            "r2": float(r2_score(y, mu)),
            "rmse": float(np.sqrt(mean_squared_error(y, mu))),
            "mae": float(mean_absolute_error(y, mu)),
            "crps": gaussian_crps(y, mu, sigma),
            "nll": gaussian_nll(y, mu, sigma),
            "picp90": picp90(y, mu, sigma),
            "mean_sigma": float(np.mean(sigma)),
        }
    return {
        "n": int(len(y)),
        "r2": weighted_r2(y, mu, w),
        "rmse": weighted_rmse(y, mu, w),
        "mae": weighted_mae(y, mu, w),
        "crps": weighted_crps(y, mu, sigma, w),
        "nll": weighted_nll(y, mu, sigma, w),
        "picp90": weighted_picp90(y, mu, sigma, w),
        "mean_sigma": float(np.sum((w / w.sum()) * sigma)),
    }


def main() -> int:
    pred = pd.read_csv(EXPORTS / "rev2_grouped_validation_predictions.csv")
    raw = pd.read_csv(DATA_PATH)
    astm = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    if len(astm) != len(pred):
        raise RuntimeError(f"row count mismatch astm={len(astm)} pred={len(pred)}")
    fp = fingerprint(astm)
    counts = fp.map(fp.value_counts())
    source_n = pred.groupby("source_id")["source_id"].transform("count")

    y = pred["y_log10_Da"].to_numpy(float)
    mu = pred["mu"].to_numpy(float)
    sigma_raw = pred["sigma_raw"].to_numpy(float)
    sigma_prod = pred["sigma_calibrated"].to_numpy(float)
    # Prefer deposited production scale; fall back to archived lambda.
    cal = json.loads((EXPORTS / "rev2_production_calibration.json").read_text(encoding="utf-8"))
    lam_prod = float(cal["production_lambda"])
    sigma_prod = lam_prod * sigma_raw

    w_row = np.ones(len(pred), dtype=float)
    w_source = (1.0 / source_n).to_numpy(float)  # equal source mass after normalize
    w_comp = (1.0 / counts).to_numpy(float)

    schemes = {
        "row_equal": w_row,
        "source_equal": w_source,
        "composition_inverse_frequency": w_comp,
    }
    payload = {
        "purpose": (
            "LOSO point and probabilistic metrics under row-, source-, and "
            "composition-inverse-frequency weighting on deposited OOF predictions"
        ),
        "n_records": int(len(pred)),
        "n_sources": int(pred["source_id"].nunique()),
        "n_unique_compositions": int(fp.nunique()),
        "production_lambda": lam_prod,
        "weighting_notes": {
            "row_equal": "each ASTM C1556 record weight 1 (production pooled estimand)",
            "source_equal": "each source receives equal total weight (1/n_source within source)",
            "composition_inverse_frequency": "weight 1/count(composition fingerprint)",
        },
        "raw_scale": {},
        "production_lambda_scale": {},
    }
    for name, w in schemes.items():
        payload["raw_scale"][name] = score_block(y, mu, sigma_raw, None if name == "row_equal" else w)
        payload["production_lambda_scale"][name] = score_block(
            y, mu, sigma_prod, None if name == "row_equal" else w
        )

    # Also report source-mean of per-source R2 (unweighted within source, then mean).
    per_source = []
    for sid, g in pred.groupby("source_id"):
        ys, ms = g["y_log10_Da"].to_numpy(float), g["mu"].to_numpy(float)
        if len(g) < 2 or np.ptp(ys) == 0:
            r2 = float("nan")
        else:
            r2 = float(r2_score(ys, ms))
        per_source.append({
            "source_id": int(sid),
            "n": int(len(g)),
            "r2": r2,
            "rmse": float(np.sqrt(mean_squared_error(ys, ms))),
            "mae": float(mean_absolute_error(ys, ms)),
        })
    r2s = [r["r2"] for r in per_source if np.isfinite(r["r2"])]
    payload["per_source_point_metrics"] = per_source
    payload["mean_of_finite_source_r2"] = float(np.mean(r2s)) if r2s else float("nan")
    payload["median_of_finite_source_r2"] = float(np.median(r2s)) if r2s else float("nan")

    out = EXPORTS / "rev2_weighted_loso_metrics.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {out}")
    for name in schemes:
        row = payload["production_lambda_scale"][name]
        print(
            f"  {name}: R2={row['r2']:.3f} RMSE={row['rmse']:.3f} "
            f"PICP90={row['picp90']:.3f} NLL={row['nll']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
