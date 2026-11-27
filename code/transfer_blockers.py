# -*- coding: utf-8 -*-
"""Early-age calibration, strength errors-in-variables, source-level coverage, and the B=1000 cluster bootstrap.

These are post-hoc checks on the deposited fronts. They do not replace the
production search or its headline regret.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_specification import EXPORTS, SPEC  # noqa: E402
from grouped_loso_validation import (  # noqa: E402
    DATA_PATH,
    MIN_SIGMA,
    fit_nll_scale,
    gaussian_crps,
    gaussian_nll,
)
from matched_search import (  # noqa: E402
    GWP,
    ReconstructionProblem,
    align_X,
    derive_support,
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
)
from dependence_sensitivity import genotype_from_pool, build_problems  # noqa: E402
from chloride_pipeline import DATA_PATH as RAW_DATA  # noqa: E402

Z90 = float(stats.norm.ppf(0.95))


def write(name: str, obj: Any) -> None:
    path = EXPORTS / name
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("wrote", path.name)


def picp90(y, mu, sigma) -> float:
    sigma = np.maximum(sigma, MIN_SIGMA)
    return float(np.mean(np.abs(y - mu) <= Z90 * sigma))


def fix_source_macro() -> dict:
    pred = pd.read_csv(EXPORTS / "rev2_grouped_validation_predictions.csv")
    wmeta = json.loads((EXPORTS / "rev2_weighted_loso_metrics.json").read_text(encoding="utf-8"))
    lam = float(wmeta["production_lambda"])
    y = pred["y_log10_Da"].to_numpy(float)
    mu = pred["mu"].to_numpy(float)
    sig_prod = np.maximum(pred["sigma_raw"].to_numpy(float) * lam, MIN_SIGMA)

    # source-equal weighted
    ns = pred.groupby("source_id")["source_id"].transform("count").to_numpy(float)
    w = (1.0 / ns)
    w = w / w.sum()
    hit = (np.abs(y - mu) <= Z90 * sig_prod).astype(float)
    weighted_picp = float(np.sum(w * hit))

    per = []
    for sid, g in pred.groupby("source_id"):
        yy, mm = g["y_log10_Da"].to_numpy(float), g["mu"].to_numpy(float)
        ss = np.maximum(g["sigma_raw"].to_numpy(float) * lam, MIN_SIGMA)
        per.append({
            "source_id": int(sid),
            "n": int(len(g)),
            "picp90_production_lambda": picp90(yy, mm, ss),
            "crps_production_lambda": float(gaussian_crps(yy, mm, ss)),
            "nll_production_lambda": float(gaussian_nll(yy, mm, ss)),
            "mean_width90": float(np.mean(2 * Z90 * ss)),
        })

    mean_picp = float(np.mean([r["picp90_production_lambda"] for r in per]))
    assert abs(mean_picp - weighted_picp) < 1e-9, (mean_picp, weighted_picp)

    # fold-local column for contrast (mislabelled before)
    fold_note = None
    mean_fold = None
    if "sigma_calibrated" in pred.columns:
        sig_f = np.maximum(pred["sigma_calibrated"].to_numpy(float), MIN_SIGMA)
        fold_ps = []
        for sid, g in pred.groupby("source_id"):
            yy, mm = g["y_log10_Da"].to_numpy(float), g["mu"].to_numpy(float)
            ss = np.maximum(g["sigma_calibrated"].to_numpy(float), MIN_SIGMA)
            fold_ps.append(picp90(yy, mm, ss))
        mean_fold = float(np.mean(fold_ps))
        fold_note = (
            "CSV column sigma_calibrated is NOT production-λ scaling; "
            f"mean of 25 source PICP90 under that column = {mean_fold:.3f}. "
            "Previous MS incorrectly paired this with production-λ source-equal 0.946."
        )

    out = {
        "purpose": "Corrected source-macro calibration under a single production-λ interval law",
        "production_lambda": lam,
        "interval_law": "sigma_star = production_lambda * sigma_raw",
        "source_equal_weighted_picp90": weighted_picp,
        "mean_of_25_source_picp90": mean_picp,
        "identity_check": "source_equal_weighted_picp90 == mean_of_25_source_picp90 under this law",
        "source_equal_weighted_crps": float(np.sum(w * (
            sig_prod * (
                ((y - mu) / sig_prod) * (2 * stats.norm.cdf((y - mu) / sig_prod) - 1)
                + 2 * stats.norm.pdf((y - mu) / sig_prod)
                - 1 / np.sqrt(np.pi)
            )
        ))),
        "source_equal_weighted_nll": float(np.sum(w * (
            0.5 * np.log(2 * np.pi) + np.log(sig_prod) + 0.5 * ((y - mu) / sig_prod) ** 2
        ))),
        "mean_width90": float(np.mean(2 * Z90 * sig_prod)),
        "fold_local_sigma_calibrated_column": {
            "mean_of_25_source_picp90": mean_fold,
            "note": fold_note,
        },
        "per_source_production_lambda": per,
    }
    write("rev2_source_macro_calibration.json", out)
    return out


def early_age_calibration() -> dict:
    pred = pd.read_csv(EXPORTS / "rev2_grouped_validation_predictions.csv")
    lam_prod, _, _ = load_lambda()
    m = (pred["Exposure_Duration"] >= 28) & (pred["Exposure_Duration"] <= 56)
    sub = pred.loc[m].copy()
    y = sub["y_log10_Da"].to_numpy(float)
    mu = sub["mu"].to_numpy(float)
    sig = np.maximum(sub["sigma_raw"].to_numpy(float), MIN_SIGMA)

    # Source-aware λ: fit NLL scale on all window rows (closed form), plus
    # leave-one-source-out λ for stability band
    lam_window = fit_nll_scale(y, mu, sig)
    loso_lams = []
    for sid in sorted(sub["source_id"].unique()):
        cal = sub[sub["source_id"] != sid]
        if len(cal) < 10:
            continue
        loso_lams.append(fit_nll_scale(
            cal["y_log10_Da"].to_numpy(float),
            cal["mu"].to_numpy(float),
            np.maximum(cal["sigma_raw"].to_numpy(float), MIN_SIGMA),
        ))
    lam_loso_mean = float(np.mean(loso_lams)) if loso_lams else lam_window
    lam_loso_sd = float(np.std(loso_lams, ddof=1)) if len(loso_lams) > 1 else 0.0

    def block(lam: float) -> dict:
        ss = np.maximum(sig * lam, MIN_SIGMA)
        return {
            "lambda": float(lam),
            "picp90": picp90(y, mu, ss),
            "crps": float(gaussian_crps(y, mu, ss)),
            "nll": float(gaussian_nll(y, mu, ss)),
            "mean_width90": float(np.mean(2 * Z90 * ss)),
            "r2": float(r2_score(y, mu)),
            "rmse": float(np.sqrt(mean_squared_error(y, mu))),
        }

    under_prod = block(lam_prod)
    under_age = block(lam_window)
    under_loso = block(lam_loso_mean)

    # Propagate age-local λ through Cat B on deposited fronts (shared residual)
    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    production = joblib.load(EXPORTS / "rev2_production_ngb.joblib")
    ckpt = {
        "ensemble": production["model"],
        "preprocess_pipeline": production["preprocessing"],
        "all_feature_names": production["all_feature_names"],
    }
    all_names = list(ckpt["all_feature_names"])
    raw = pd.read_csv(RAW_DATA)
    training = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    support = derive_support(training)
    m_mk = mk_mean_from_workbook()
    X_train = align_X(training, all_names)
    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    k = max(1, min(SPEC.ood_knn_neighbors, len(training) - 1))
    self_dist = NearestNeighbors(n_neighbors=k + 1).fit(Z_train).kneighbors(Z_train)[0][:, k]
    ood_limit = float(np.quantile(self_dist, SPEC.ood_training_quantile))
    knn = NearestNeighbors(n_neighbors=k).fit(Z_train)
    saa_n = int(pd.read_csv(EXPORTS / "rev2_saa_convergence.csv")["selected_production_n"].iloc[0])
    mc_n = 2000
    seeds = sorted(pools["optimization_seed"].unique())

    def score_catB(lam: float) -> dict:
        problems = build_problems(
            pools, ckpt, all_names, support, scaler, knn, ood_limit, m_mk, lam, saa_n,
        )
        rows = []
        for seed in seeds:
            audit_seed = 92_000 + int(seed)
            scored = {}
            for mode in ("mean", "q10"):
                subset = pools[
                    (pools["pool"] == mode) & (pools["optimization_seed"] == seed)
                ].sort_values("local_id").reset_index(drop=True)
                problem = problems[(mode, int(seed))]
                X = genotype_from_pool(subset, problem.support)
                base = problem.evaluate_details(X)
                rng = np.random.default_rng(audit_seed)
                z = rng.standard_normal(mc_n)
                z = np.broadcast_to(z, (len(subset), mc_n)).copy()
                cs, cc = sample_cs(mc_n, rng), sample_ccrit(mc_n, rng)
                logd = base["mu"][:, None] + lam * base["sigma"][:, None] * z
                d = 10 ** logd * 1e-12
                te = t_eq(t_init_years(d, base["m"][:, None], cs, cc))
                gwp = np.asarray([impact_from_mix(m, GWP) for m in base["mixes"]], dtype=float)
                scored[mode] = gwp[:, None] / np.maximum(te, 1e-12)
            base_b = union_draw_baseline(scored["mean"], scored["q10"])
            for mode in ("mean", "q10"):
                fail, regret, _, mean_b = s3_draw_metrics(scored[mode], 10, baseline=base_b)
                rank = np.argsort(mean_b)
                ktop = min(10, len(mean_b))
                rows.append({
                    "optimization_seed": int(seed),
                    "pool": mode,
                    "s3_expected_category_b_regret": float(np.mean((regret / mc_n)[rank[:ktop]])),
                })
        frame = pd.DataFrame(rows)
        inf = paired_metric_inference(frame, "s3_expected_category_b_regret")
        return inf

    print("Scoring Cat B under production λ…")
    catb_prod = score_catB(float(lam_prod))
    print("Scoring Cat B under age-local λ…")
    catb_age = score_catB(float(lam_window))

    survive = bool(catb_age.get("materially_reduces")) and float(catb_age.get("relative_mean_change", 1)) < 0
    out = {
        "purpose": "Early-age (28–56 d) nested NLL multiplier and Cat B propagation",
        "window": {"n_records": int(len(sub)), "n_sources": int(sub["source_id"].nunique())},
        "production_lambda": float(lam_prod),
        "lambda_28_56_pooled_nll": float(lam_window),
        "lambda_28_56_loso_mean": lam_loso_mean,
        "lambda_28_56_loso_sd": lam_loso_sd,
        "n_loso_lambda_folds": len(loso_lams),
        "window_metrics_production_lambda": under_prod,
        "window_metrics_age_local_lambda": under_age,
        "window_metrics_loso_mean_lambda": under_loso,
        "category_b_regret_production_lambda": {
            "mean_pool_mean": catb_prod.get("mean_pool_mean"),
            "q10_pool_mean": catb_prod.get("q10_pool_mean"),
            "relative_mean_change": catb_prod.get("relative_mean_change"),
            "materially_reduces": catb_prod.get("materially_reduces"),
        },
        "category_b_regret_age_local_lambda": {
            "mean_pool_mean": catb_age.get("mean_pool_mean"),
            "q10_pool_mean": catb_age.get("q10_pool_mean"),
            "relative_mean_change": catb_age.get("relative_mean_change"),
            "materially_reduces": catb_age.get("materially_reduces"),
            "paired_bootstrap_mean_delta_95ci": catb_age.get("paired_bootstrap_mean_delta_95ci"),
        },
        "direction_survives_age_local_lambda": survive,
        "reference_age_decision": (
            "Retain 35 d as the declared reference age for the primary analysis. "
            "The 28–56 day coverage estimate uses held-out records in that age window. "
            f"Age-local λ≈{lam_window:.2f} restores window PICP90 to {under_age['picp90']:.3f}; "
            + (
                "Category B q10 advantage remains material under age-local residual scaling."
                if survive
                else "Category B direction is weakened/overturned under age-local scaling — escalate reference-age sensitivity."
            )
        ),
        "mc_draws": mc_n,
        "n_seeds": len(seeds),
    }
    write("rev2_early_age_calibration.json", out)
    return out


def strength_eiv_catB() -> dict:
    """Sample ridge-strength residuals before D_app evaluation (post-hoc on fronts)."""
    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    cs_meta = json.loads((EXPORTS / "rev2_strength_model.json").read_text(encoding="utf-8"))
    rmse = float(cs_meta["validation"][0]["rmse"])
    mae = float(cs_meta["validation"][0]["mae"])
    # Use LOSO RMSE as residual SD for Gaussian EIV
    sd = rmse

    production = joblib.load(EXPORTS / "rev2_production_ngb.joblib")
    ckpt = {
        "ensemble": production["model"],
        "preprocess_pipeline": production["preprocessing"],
        "all_feature_names": production["all_feature_names"],
    }
    all_names = list(ckpt["all_feature_names"])
    raw = pd.read_csv(RAW_DATA)
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
    saa_n = int(pd.read_csv(EXPORTS / "rev2_saa_convergence.csv")["selected_production_n"].iloc[0])
    problems = build_problems(
        pools, ckpt, all_names, support, scaler, knn, ood_limit, m_mk, lam, saa_n,
    )

    from matched_search import predict_mu_sigma, feature_record, m_of_mix

    mc_n = 800  # lighter MC; 30 seeds still heavy
    seeds = sorted(pools["optimization_seed"].unique())
    # Evaluate all 30 matched seeds using 800 draws per seed.
    rows = []
    for seed in seeds:
        audit_seed = 93_000 + int(seed)
        scored = {}
        for mode in ("mean", "q10"):
            subset = pools[
                (pools["pool"] == mode) & (pools["optimization_seed"] == seed)
            ].sort_values("local_id").reset_index(drop=True)
            problem = problems[(mode, int(seed))]
            X = genotype_from_pool(subset, problem.support)
            mixes, records = problem.decode(X)
            # point strength already in records
            cs_hat = records["CS"].to_numpy(float)
            rng = np.random.default_rng(audit_seed)
            # EIV: independent strength noise per candidate per draw
            eps = rng.normal(0.0, sd, size=(len(subset), mc_n))
            cs_s = np.clip(cs_hat[:, None] + eps, 15.0, 95.0)
            z = rng.standard_normal(mc_n)
            z = np.broadcast_to(z, (len(subset), mc_n)).copy()  # shared diffusivity residual
            cs_exp, cc = sample_cs(mc_n, rng), sample_ccrit(mc_n, rng)

            # Finite-difference sensitivity of (mu, sigma) to CS (one-shot)
            rec_lo = records.copy()
            rec_hi = records.copy()
            delta = 2.0  # MPa
            rec_lo["CS"] = cs_hat - delta
            rec_hi["CS"] = cs_hat + delta
            mu0, sig0 = predict_mu_sigma(ckpt, align_X(records, all_names))
            mu_hi, sig_hi = predict_mu_sigma(ckpt, align_X(rec_hi, all_names))
            mu_lo, sig_lo = predict_mu_sigma(ckpt, align_X(rec_lo, all_names))
            dmu = (mu_hi - mu_lo) / (2 * delta)
            dsig = (sig_hi - sig_lo) / (2 * delta)
            # mu(cs) ≈ mu0 + dmu*(cs - cs_hat)
            mu_s = mu0[:, None] + dmu[:, None] * (cs_s - cs_hat[:, None])
            sig_s = np.maximum(sig0[:, None] + dsig[:, None] * (cs_s - cs_hat[:, None]), MIN_SIGMA)
            logd = mu_s + lam * sig_s * z
            d = 10 ** logd * 1e-12
            mvec = np.array([
                m_of_mix(r["OPC"], r["FA"], r["GGBFS"], r["SF"], r["MK"], m_mk) for r in mixes
            ])
            te = t_eq(t_init_years(d, mvec[:, None], cs_exp, cc))
            gwp = np.asarray([impact_from_mix(m, GWP) for m in mixes], dtype=float)
            scored[mode] = gwp[:, None] / np.maximum(te, 1e-12)

        base_b = union_draw_baseline(scored["mean"], scored["q10"])
        for mode in ("mean", "q10"):
            fail, regret, _, mean_b = s3_draw_metrics(scored[mode], 10, baseline=base_b)
            rank = np.argsort(mean_b)
            ktop = min(10, len(mean_b))
            rows.append({
                "optimization_seed": int(seed),
                "pool": mode,
                "s3_expected_category_b_regret": float(np.mean((regret / mc_n)[rank[:ktop]])),
            })
        print(f"EIV seed {seed} done")

    frame = pd.DataFrame(rows)
    inf = paired_metric_inference(frame, "s3_expected_category_b_regret")
    out = {
        "purpose": "Strength errors-in-variables: Gaussian noise on ridge CS (SD=LOSO RMSE) before D_app",
        "strength_residual_sd_mpa": sd,
        "strength_loso_mae_mpa": mae,
        "approximation": "first-order dmu/dCS, dsig/dCS from ±2 MPa finite difference; shared diffusivity z; independent CS noise",
        "mc_draws": mc_n,
        "n_seeds": len(seeds),
        "category_b_regret": {
            "mean_pool_mean": inf.get("mean_pool_mean"),
            "q10_pool_mean": inf.get("q10_pool_mean"),
            "relative_mean_change": inf.get("relative_mean_change"),
            "materially_reduces": inf.get("materially_reduces"),
            "paired_bootstrap_mean_delta_95ci": inf.get("paired_bootstrap_mean_delta_95ci"),
            "paired_wilcoxon_two_sided_p": inf.get("paired_wilcoxon_two_sided_p"),
        },
        "direction_survives": bool(inf.get("materially_reduces")),
    }
    write("rev2_strength_eiv_catB.json", out)
    return out


def bootstrap_b1000() -> dict:
    pred = pd.read_csv(EXPORTS / "rev2_grouped_validation_predictions.csv")
    rng = np.random.default_rng(20260807)
    sources = pred["source_id"].unique()
    B = 1000
    rows = []
    for b in range(B):
        draw = rng.choice(sources, size=len(sources), replace=True)
        sub = pd.concat([pred[pred["source_id"] == s] for s in draw], ignore_index=True)
        y = sub["y_log10_Da"].to_numpy(float)
        mu = sub["mu"].to_numpy(float)
        rows.append({
            "b": b,
            "r2": float(r2_score(y, mu)),
            "rmse": float(np.sqrt(mean_squared_error(y, mu))),
            "mae": float(mean_absolute_error(y, mu)),
        })
    boot = pd.DataFrame(rows)
    out = {
        "purpose": "Source-cluster bootstrap of deposited OOF transfer scores",
        "B": B,
        "warning": (
            "Resamples deposited OOF predictions by source; does not refit NGB or reoptimize. "
            "Interval = uncertainty in the deposited OOF transfer score only."
        ),
        "loso_r2_bootstrap": {
            "mean": float(boot["r2"].mean()),
            "sd": float(boot["r2"].std(ddof=1)),
            "q05": float(boot["r2"].quantile(0.05)),
            "q50": float(boot["r2"].quantile(0.50)),
            "q95": float(boot["r2"].quantile(0.95)),
        },
        "loso_rmse_bootstrap": {
            "mean": float(boot["rmse"].mean()),
            "q05": float(boot["rmse"].quantile(0.05)),
            "q95": float(boot["rmse"].quantile(0.95)),
        },
    }
    write("rev2_source_cluster_bootstrap_catB.json", out)
    return out


def main() -> int:
    print("=== Fix source-macro PICP ===")
    fix_source_macro()
    print("=== Early-age calibration + Cat B ===")
    early_age_calibration()
    print("=== Strength EIV Cat B ===")
    strength_eiv_catB()
    print("=== Bootstrap B=1000 ===")
    bootstrap_b1000()
    print("DONE transfer-blocker package (in-search conformal is separate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
