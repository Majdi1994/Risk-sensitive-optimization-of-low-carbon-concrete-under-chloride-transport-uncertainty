#!/usr/bin/env python3
"""Compressive-strength model for generated candidate mixtures.

Production strength is StandardScaler + Ridge (alpha from SPEC) fitted on
measured ASTM C1556 mixture proportions and validated with leave-one-source-out.
Predictions are clipped to the measured strength range.

`surrogate_cs` is a legacy closed-form comparator retained only for scoring and
sensitivity audits. It is not used by the production search (`matched_search` /
`run_paper.py`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import grouped_loso_validation as gv
from analysis_specification import SPEC

REV = Path(__file__).resolve().parents[1]
EXPORTS = REV / "outputs"
MODEL_PATH = EXPORTS / "rev2_strength_model.joblib"
REPORT_PATH = EXPORTS / "rev2_strength_model.json"
DEPLOYED_ALPHA = float(SPEC.strength_ridge_alpha)

# Mixture composition only. Exposure duration and temperature describe the
# diffusion test, not the strength test, so they are excluded.
FEATURES = [
    "Cement", "SF", "GGBFS", "FA", "MK", "Water", "w_b_ratio", "NFA", "NCA", "SP",
    "Total_Binder", "SCM_Replacement_Ratio", "SF_Ratio", "GGBFS_Ratio", "FA_Ratio",
    "MK_Ratio",
]
SEED = 20260731


def surrogate_cs(frame: pd.DataFrame) -> np.ndarray:
    """Legacy closed-form comparator for scoring against measured strengths."""
    w = frame["w_b_ratio"].to_numpy(dtype=float)
    scm = frame["SCM_Replacement_Ratio"].to_numpy(dtype=float)
    binder = frame["Total_Binder"].to_numpy(dtype=float)
    scm_mass = np.maximum(binder * np.maximum(scm, 0.0), 1e-12)
    sf_frac = np.clip(frame["SF"].to_numpy(dtype=float) / scm_mass, 0.0, 1.0)
    return np.clip(50 - 30 * w + 10 * (1 - scm) + 12 * scm * sf_frac, 20, 80)


def _block(name: str, y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "evaluation": name,
        "n": int(len(y)),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
    }


def make_estimator(alpha: float = DEPLOYED_ALPHA) -> Pipeline:
    """Production strength estimator: nested GroupKFold selected alpha=100."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def load_corpus() -> pd.DataFrame:
    frame = pd.read_csv(gv.DATA_PATH)
    frame = frame[frame["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    missing = [c for c in FEATURES + ["CS", "source_id"] if c not in frame.columns]
    if missing:
        raise KeyError(f"strength corpus is missing columns: {missing}")
    return frame


def fit_and_report() -> dict[str, Any]:
    frame = load_corpus()
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame["CS"].to_numpy(dtype=float)
    groups = frame["source_id"].to_numpy(dtype=int)

    loso = cross_val_predict(
        make_estimator(), x, y, cv=LeaveOneGroupOut(), groups=groups,
    )
    random = cross_val_predict(
        make_estimator(), x, y, cv=KFold(5, shuffle=True, random_state=SEED),
    )

    final = make_estimator(DEPLOYED_ALPHA).fit(x, y)
    in_sample = final.predict(x)

    lo, hi = float(y.min()), float(y.max())
    payload = {
        "purpose": (
            "Supply compressive strength for generated candidate mixtures, which "
            "are never cast and therefore have no measured strength."
        ),
        "estimator": "StandardScaler + Ridge on mixture proportions",
        "selected_alpha": float(DEPLOYED_ALPHA),
        "alpha_source": (
            "final_production_tune_on_all_sources.best_params.model__alpha "
            "from rev2_strength_nested_tuning.json (nested GroupKFold)"
        ),
        "features": FEATURES,
        "excluded_features_and_why": (
            "Exposure_Duration and Exposure_Temperature describe the diffusion "
            "test rather than the strength test, so they are not strength inputs."
        ),
        "n_records": int(len(y)),
        "n_sources": int(len(set(groups))),
        "measured_cs_mpa": {
            "min": lo, "max": hi, "mean": float(y.mean()),
            "sd": float(y.std(ddof=1)),
        },
        "prediction_clip_mpa": [lo, hi],
        "validation": [
            _block("leave-one-source-out (headline)", y, loso),
            _block("random 5-fold (interpolation reference)", y, random),
            _block("in-sample", y, in_sample),
        ],
        "legacy_surrogate": _block(
            "legacy closed-form comparator scored on measured records", y, surrogate_cs(frame),
        ),
        "interpretation": (
            "The grouped validation error estimates strength prediction error on "
            "held-out sources. Production search uses the fitted Ridge model."
        ),
    }
    payload["legacy_surrogate"]["bias_mpa"] = float(
        (surrogate_cs(frame) - y).mean()
    )

    joblib.dump(
        {
            "model": final,
            "features": FEATURES,
            "clip": (lo, hi),
            "validation": payload["validation"],
            "seed": SEED,
            "alpha": float(DEPLOYED_ALPHA),
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class StrengthPredictor:
    """Thin wrapper so the optimizer can score decoded candidate records."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Run strength_model.py first; missing artifact: {path}"
            )
        art = joblib.load(path)
        self.model = art["model"]
        self.features = list(art["features"])
        self.lo, self.hi = art["clip"]

    def predict(self, records: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in records.columns]
        if missing:
            raise KeyError(f"candidate records lack strength features: {missing}")
        x = records[self.features].to_numpy(dtype=float)
        return np.clip(self.model.predict(x), self.lo, self.hi)


def main() -> int:
    payload = fit_and_report()
    print(f"deployed ridge alpha = {payload['selected_alpha']:.4g}")
    print(f"measured CS: {payload['measured_cs_mpa']['min']:.2f}"
          f"-{payload['measured_cs_mpa']['max']:.2f} MPa, "
          f"SD {payload['measured_cs_mpa']['sd']:.2f}")
    print()
    rows = pd.DataFrame(payload["validation"] + [payload["legacy_surrogate"]])
    print(rows[["evaluation", "n", "r2", "rmse", "mae"]].round(4).to_string(index=False))
    print()
    print(f"wrote {MODEL_PATH.name} and {REPORT_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
