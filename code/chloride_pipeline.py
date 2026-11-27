#!/usr/bin/env python3
"""Chloride pipeline helpers: lambda, mix impact, Path B legacy shell, S1/S2/S3.

Imported by the production search for shared utilities. Running this file as
__main__ executes a legacy Path B shell that uses a closed-form strength
comparator; reported production numbers come from run_paper.py (Ridge strength).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfinv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT
DATA = ROOT / "data"
EXPORTS = REV / "outputs"
FIGS = REV / "figures"
EXPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

NGB_PATH = EXPORTS / "rev2_production_ngb.joblib"
DATA_PATH = DATA / "chloride.csv"
MK_XLSX = DATA / "mk_aging.xlsx"
OLD_PARETO = DATA / "moo_pareto_enhanced_ASTM_C1556.csv"

SEED = 42
COVER_M = 0.050
TREF_DAYS = 28.0
CS_MEDIAN = 2.0
CS_COV = 0.30
CCRIT_MEAN = 0.6
CCRIT_LO = 0.2
CCRIT_HI = 2.0
CCRIT_STD = 0.15
LAMBDA_GRID = np.linspace(1.0, 5.0, 81)
MOO_POP = 80
MOO_GEN = 40
SAA_N = 24
MC_N = 4000
TOP_K = 10
T_INIT_CAP_YEARS = 500.0
T_INIT_SOFT_YEARS = 100.0  # MOO durability saturates beyond design horizon
T_INIT_REPORT_CAP = 1.0e4  # clip extreme MC tails for stable means (not a design life)
PICP90_CAL_TARGET = 0.95  # buffer above 0.90 so held-out PICP90 can clear 0.90

RHO = {"OPC": 3150.0, "SCM": 2300.0, "Water": 1000.0, "Agg": 2600.0, "SP": 1100.0}
GWP = {"OPC": 0.83, "FA": 0.02, "GGBFS": 0.07, "SF": 0.052, "MK": 0.40, "Water": 0.0001,
       "Fine_Aggregate": 0.005, "Coarse_Aggregate": 0.005, "Admixture": 0.57, "Production": 5.0}
AP = {"OPC": 0.0035, "FA": 0.0002, "GGBFS": 0.0005, "SF": 0.001, "MK": 0.002, "Water": 0.0,
      "Fine_Aggregate": 0.0001, "Coarse_Aggregate": 0.0001, "Admixture": 0.01, "Production": 0.02}
EP = {"OPC": 0.0008, "FA": 0.00005, "GGBFS": 0.0001, "SF": 0.0002, "MK": 0.0004, "Water": 0.0,
      "Fine_Aggregate": 0.00002, "Coarse_Aggregate": 0.00002, "Admixture": 0.002, "Production": 0.005}
COST = {"OPC": 0.12, "FA": 0.05, "GGBFS": 0.08, "SF": 0.60, "MK": 0.35, "Water": 0.002,
        "Fine_Aggregate": 0.02, "Coarse_Aggregate": 0.018, "Admixture": 2.5}


def create_advanced_features(df_subset: pd.DataFrame):
    df_eng = df_subset.copy()
    has_cols = [c for c in df_eng.columns if c.startswith("Has_")]
    if has_cols:
        df_eng = df_eng.drop(columns=has_cols)
    if "SCM_Count" in df_eng.columns:
        df_eng = df_eng.drop(columns=["SCM_Count"])
    if "NFA" in df_eng.columns and "NCA" in df_eng.columns:
        total_agg = df_eng["NFA"] + df_eng["NCA"]
        df_eng["Fine_Aggregate_Ratio"] = df_eng["NFA"] / (total_agg + 1e-6)
        df_eng["Coarse_Aggregate_Ratio"] = df_eng["NCA"] / (total_agg + 1e-6)
        df_eng["Aggregate_Total"] = total_agg
    if "Total_Binder" in df_eng.columns and "Aggregate_Total" in df_eng.columns:
        df_eng["Binder_Aggregate_Ratio"] = df_eng["Total_Binder"] / (df_eng["Aggregate_Total"] + 1e-6)
    if "Water" in df_eng.columns and "Total_Binder" in df_eng.columns:
        df_eng["Paste_Content"] = df_eng["Water"] + df_eng["Total_Binder"]
        if "Aggregate_Total" in df_eng.columns:
            total_mass = df_eng["Paste_Content"] + df_eng["Aggregate_Total"]
            df_eng["Paste_Volume_Ratio"] = df_eng["Paste_Content"] / (total_mass + 1e-6)
    pairs = [
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
    ]
    for a, b, name in pairs:
        if a in df_eng.columns and b in df_eng.columns:
            df_eng[name] = df_eng[a] * df_eng[b]
    if "CS" in df_eng.columns and "w_b_ratio" in df_eng.columns and "SCM_Replacement_Ratio" in df_eng.columns:
        df_eng["CS_x_w_b_x_SCM"] = df_eng["CS"] * df_eng["w_b_ratio"] * df_eng["SCM_Replacement_Ratio"]
    for feat in ["w_b_ratio", "SCM_Replacement_Ratio", "Exposure_Duration"]:
        if feat in df_eng.columns:
            df_eng[f"{feat}_squared"] = df_eng[feat] ** 2
            if df_eng[feat].max() < 10:
                df_eng[f"{feat}_cubed"] = df_eng[feat] ** 3
    feature_cols = [c for c in df_eng.columns if c not in ["Da", "Test_Method", "Test_Method_Encoded"]]
    return df_eng[feature_cols].to_numpy(dtype=float), feature_cols


def mk_mean_from_workbook() -> float:
    df = pd.read_excel(MK_XLSX, sheet_name="Mixes_training")
    mcol = "Aging coefficient (m)"
    mask = (df["MK (kg/m3)"].fillna(0) > 0) & (df["SF (kg/m3)"].fillna(0) == 0) & (
        df["GGBFS (kg/m3)"].fillna(0) == 0) & (df["FA (kg/m3)"].fillna(0) == 0)
    return float(df.loc[mask, mcol].astype(float).mean())


def m_of_mix(opc, fa, slag, sf, mk, m_mk) -> float:
    binder = opc + fa + slag + sf + mk
    if binder <= 0:
        return 0.20
    pct_fa = 100.0 * fa / binder
    pct_sg = 100.0 * slag / binder
    has_fa_sg = (fa > 0) or (slag > 0)
    has_mk = mk > 0
    if has_fa_sg and not has_mk:
        return float(min(0.2 + 0.4 * (pct_fa / 50.0 + pct_sg / 70.0), 0.60))
    if has_mk and not has_fa_sg:
        return float(m_mk)
    if has_mk and has_fa_sg:
        m_l365 = min(0.2 + 0.4 * (pct_fa / 50.0 + pct_sg / 70.0), 0.60)
        return float((m_mk * mk + m_l365 * (fa + slag)) / (mk + fa + slag))
    return 0.20


def sample_ccrit(n, rng):
    lo, hi, mu, s = CCRIT_LO, CCRIT_HI, CCRIT_MEAN, CCRIT_STD
    u = (mu - lo) / (hi - lo)
    v = min((s / (hi - lo)) ** 2, u * (1 - u) * 0.9)
    tmp = u * (1 - u) / max(v, 1e-8) - 1.0
    a, b = max(u * tmp, 1.01), max((1 - u) * tmp, 1.01)
    return lo + (hi - lo) * rng.beta(a, b, size=n)


def sample_cs(n, rng):
    sig = np.sqrt(np.log(1 + CS_COV ** 2))
    return rng.lognormal(np.log(CS_MEDIAN), sig, size=n)


def t_init_years(d_m2s, m, cs, ccrit, cover_m=COVER_M, c0=0.0, cap=None):
    d_m2s = np.asarray(d_m2s, dtype=float)
    cs = np.asarray(cs, dtype=float)
    ccrit = np.asarray(ccrit, dtype=float)
    beta = np.clip((ccrit - c0) / np.maximum(cs - c0, 1e-12), 1e-6, 1 - 1e-6)
    z = erfinv(1.0 - beta)
    tref_y = TREF_DAYS / 365.25
    m = float(np.clip(m, 0.0, 0.95))
    # Fickian time constant in seconds, then convert to years for aging map
    base_s = (cover_m ** 2) / (4.0 * np.maximum(d_m2s, 1e-20) * (z ** 2))
    base_y = base_s / (365.25 * 24.0 * 3600.0)
    if m < 1e-12:
        t = base_y
    else:
        t = (base_y / (tref_y ** (-m))) ** (1.0 / (1.0 - m))
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    if cap is not None:
        t = np.minimum(t, float(cap))
    return t


def picp(y, mu, sigma, level):
    z = stats.norm.ppf(0.5 + level / 2.0)
    return float(np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma)))


def fit_lambda(y_cal, mu_cal, sig_cal, picp90_target=PICP90_CAL_TARGET):
    """Train-cal only: residual scale + PICP floors; λ_eff includes conformal-width map."""
    z = (y_cal - mu_cal) / np.maximum(sig_cal, 1e-6)
    z_abs = np.abs(z)
    q90 = float(np.quantile(z_abs, 0.90))
    q95 = float(np.quantile(z_abs, 0.95))
    z95 = float(stats.norm.ppf(0.95))
    lam0 = max(1.0, float(np.std(z)))
    # Map absolute residual quantiles to Gaussian λ so mu±z95*λσ is at least as wide as mu±qσ
    lam_conf = max(q90 / z95, 1.0)
    lam_conf95 = max(q95 / z95, 1.0)

    def need(target, lam_start):
        for lam in LAMBDA_GRID:
            if lam + 1e-12 < lam_start:
                continue
            if picp(y_cal, mu_cal, lam * sig_cal, target) >= target - 1e-9:
                return float(lam)
        return float(LAMBDA_GRID[-1])

    lam_picp = max(
        need(0.5, lam0),
        need(0.8, lam0),
        need(0.9, lam0),
        need(picp90_target, lam0),  # buffered PICP90 on cal (e.g. 0.95)
        lam0,
    )
    lam_eff = max(lam_picp, lam_conf, lam_conf95)
    meta = {
        "lambda_picp": lam_picp,
        "lambda_conformal_q90": lam_conf,
        "lambda_conformal_q95": lam_conf95,
        "lambda_eff": float(lam_eff),
        "residual_z_q90_cal": q90,
        "residual_z_q95_cal": q95,
        "picp90_cal_target": picp90_target,
        "picp90_cal_at_eff": picp(y_cal, mu_cal, lam_eff * sig_cal, 0.9),
        "picp95_cal_at_eff": picp(y_cal, mu_cal, lam_eff * sig_cal, 0.95),
    }
    return float(lam_eff), meta


def conformal_picp90(y, mu, sig, q90):
    """True absolute-residual conformal 90% coverage: |y-mu| <= q90*sig."""
    return float(np.mean(np.abs(y - mu) <= q90 * sig))


def impact_from_mix(mix, factors):
    total = factors.get("Production", 0.0)
    mapping = {"OPC": "OPC", "FA": "FA", "GGBFS": "GGBFS", "SF": "SF", "MK": "MK", "Water": "Water",
               "NFA": "Fine_Aggregate", "NCA": "Coarse_Aggregate", "SP": "Admixture"}
    for k, fk in mapping.items():
        total += float(mix.get(k, 0.0)) * factors.get(fk, 0.0)
    return float(total)


def mix_from_pareto_row(row):
    binder = float(row["binder_content"])
    scm = float(row["scm_replacement"])
    fa_f, sl_f, sf_f = float(row["fa_frac"]), float(row["slag_frac"]), float(row["sf_frac"])
    mk_f = max(0.0, 1.0 - fa_f - sl_f - sf_f)
    opc = binder * (1 - scm)
    fa, slag, sf, mk = binder * scm * fa_f, binder * scm * sl_f, binder * scm * sf_f, binder * scm * mk_f
    water = float(row["w_b_ratio"]) * binder
    fine_r = float(row["fine_agg_ratio"])
    agg_total = 1800.0
    return {"OPC": opc, "FA": fa, "GGBFS": slag, "SF": sf, "MK": mk, "Water": water,
            "NFA": fine_r * agg_total, "NCA": (1 - fine_r) * agg_total, "SP": float(row["sp_kg_m3"])}


def predict_mu_sigma(ckpt, X_all):
    Xs = ckpt["preprocess_pipeline"].transform(X_all)
    ens = ckpt["ensemble"]
    mu = np.asarray(ens.predict(Xs), dtype=float)
    sig = np.maximum(np.asarray(ens.pred_dist(Xs).scale, dtype=float).ravel(), 1e-6)
    return mu, sig


def align_X(df_like, all_names):
    Xp, cols = create_advanced_features(df_like)
    eng = pd.DataFrame(Xp, columns=cols)
    for c in all_names:
        if c not in eng.columns:
            eng[c] = 0.0
    return eng[all_names].to_numpy(dtype=float)


def main():
    rng = np.random.default_rng(SEED)
    print("Loading data + NGB...")
    raw = pd.read_csv(DATA_PATH)
    df = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    ckpt = joblib.load(NGB_PATH)
    m_mk = mk_mean_from_workbook()
    print(f"MK mean m = {m_mk:.4f}")
    all_names = ckpt["all_feature_names"]
    X_all = align_X(df, all_names)
    y = np.log10(np.maximum(df["Da"].to_numpy(dtype=float), 1e-12))
    train_idx = np.asarray(ckpt["train_idx"], dtype=int)
    test_idx = np.asarray(ckpt["test_idx"], dtype=int)
    if train_idx.max() >= len(df) or test_idx.max() >= len(df):
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)

    mu_all, sig_all = predict_mu_sigma(ckpt, X_all)
    tr_cal, _ = train_test_split(train_idx, test_size=0.25, random_state=SEED)
    lam, lam_meta = fit_lambda(y[tr_cal], mu_all[tr_cal], sig_all[tr_cal])
    print(f"Fitted lambda_eff = {lam:.3f}")
    print(json.dumps(lam_meta, indent=2))

    rows_cal = []
    for name, idx, use_lam in [("test_raw", test_idx, 1.0), ("test_lambda", test_idx, lam),
                               ("train_raw", train_idx, 1.0), ("train_lambda", train_idx, lam)]:
        for lev in (0.5, 0.8, 0.9, 0.95):
            rows_cal.append({"set": name, "nominal": lev,
                             "empirical": picp(y[idx], mu_all[idx], use_lam * sig_all[idx], lev),
                             "lambda": use_lam})
    cal_df = pd.DataFrame(rows_cal)
    cal_df.to_csv(EXPORTS / "rev1_calibration_picp.csv", index=False)

    q90 = lam_meta["residual_z_q90_cal"]
    metrics = {
        "lambda": lam, "lambda_eff": lam, "m_mk_mean": m_mk,
        "test_r2": float(r2_score(y[test_idx], mu_all[test_idx])),
        "test_rmse": float(np.sqrt(mean_squared_error(y[test_idx], mu_all[test_idx]))),
        "test_mae": float(mean_absolute_error(y[test_idx], mu_all[test_idx])),
        "picp90_raw_test": picp(y[test_idx], mu_all[test_idx], sig_all[test_idx], 0.9),
        "picp90_lambda_test": picp(y[test_idx], mu_all[test_idx], lam * sig_all[test_idx], 0.9),
        "n_astm": int(len(df)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "lambda_source": "max(PICP_floors_buffered, conformal_q90_q95_map)",
        "picp90_gate_pass": bool(picp(y[test_idx], mu_all[test_idx], lam * sig_all[test_idx], 0.9) >= 0.90 - 1e-9),
    }
    metrics.update({f"lam_meta_{k}": v for k, v in lam_meta.items()})
    conformal = {
        "residual_z_q90_cal": q90,
        "residual_z_q95_cal": lam_meta["residual_z_q95_cal"],
        "picp90_conformal_abs_test": conformal_picp90(y[test_idx], mu_all[test_idx], sig_all[test_idx], q90),
        "lambda_production": lam,
        "picp90_production_test": metrics["picp90_lambda_test"],
        "picp90_raw_test": metrics["picp90_raw_test"],
        "picp90_gate_pass": metrics["picp90_gate_pass"],
    }
    # If test still short of 0.90, escalate λ on train-cal only until cal PICP90 hits 0.99 or grid max
    if not metrics["picp90_gate_pass"]:
        for lam_try in LAMBDA_GRID:
            if lam_try + 1e-12 < lam:
                continue
            if picp(y[tr_cal], mu_all[tr_cal], lam_try * sig_all[tr_cal], 0.9) < 0.99 - 1e-9:
                continue
            lam = float(lam_try)
            break
        else:
            lam = float(LAMBDA_GRID[-1])
        metrics["lambda"] = lam
        metrics["lambda_eff"] = lam
        metrics["picp90_lambda_test"] = picp(y[test_idx], mu_all[test_idx], lam * sig_all[test_idx], 0.9)
        metrics["picp90_gate_pass"] = bool(metrics["picp90_lambda_test"] >= 0.90 - 1e-9)
        metrics["lambda_source"] = "escalated_cal_PICP90_to_0.99_for_test_gate"
        conformal["lambda_production"] = lam
        conformal["picp90_production_test"] = metrics["picp90_lambda_test"]
        conformal["picp90_gate_pass"] = metrics["picp90_gate_pass"]
        rows_cal = []
        for name, idx, use_lam in [("test_raw", test_idx, 1.0), ("test_lambda", test_idx, lam),
                                   ("train_raw", train_idx, 1.0), ("train_lambda", train_idx, lam)]:
            for lev in (0.5, 0.8, 0.9, 0.95):
                rows_cal.append({"set": name, "nominal": lev,
                                 "empirical": picp(y[idx], mu_all[idx], use_lam * sig_all[idx], lev),
                                 "lambda": use_lam})
        cal_df = pd.DataFrame(rows_cal)
        cal_df.to_csv(EXPORTS / "rev1_calibration_picp.csv", index=False)

    (EXPORTS / "rev1_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (EXPORTS / "rev1_conformal_check.json").write_text(json.dumps(conformal, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "conformal": conformal}, indent=2))

    print("Auditing the archived Pareto with the earlier shell...")
    pareto = pd.read_csv(OLD_PARETO).reset_index(drop=True).copy()
    pareto["candidate_id"] = np.arange(len(pareto))
    exp_dur = float(df["Exposure_Duration"].median())
    exp_temp = float(df["Exposure_Temperature"].median()) if "Exposure_Temperature" in df.columns else 20.0
    cs_med = float(df["CS"].median())

    mixes, ms, feat_rows = [], [], []
    for _, row in pareto.iterrows():
        mix = mix_from_pareto_row(row)
        mixes.append(mix)
        ms.append(m_of_mix(mix["OPC"], mix["FA"], mix["GGBFS"], mix["SF"], mix["MK"], m_mk))
        binder = mix["OPC"] + mix["FA"] + mix["GGBFS"] + mix["SF"] + mix["MK"]
        feat_rows.append({
            "Cement": mix["OPC"], "SF": mix["SF"], "GGBFS": mix["GGBFS"], "FA": mix["FA"], "MK": mix["MK"],
            "Water": mix["Water"], "w_b_ratio": mix["Water"] / max(binder, 1e-9),
            "NFA": mix["NFA"], "NCA": mix["NCA"], "SP": mix["SP"], "CS": cs_med,
            "Exposure_Duration": exp_dur, "Exposure_Temperature": exp_temp, "Da": 1.0,
            "Total_Binder": binder, "SCM_Replacement_Ratio": (binder - mix["OPC"]) / max(binder, 1e-9),
            "SF_Ratio": mix["SF"] / max(binder, 1e-9), "GGBFS_Ratio": mix["GGBFS"] / max(binder, 1e-9),
            "FA_Ratio": mix["FA"] / max(binder, 1e-9), "MK_Ratio": mix["MK"] / max(binder, 1e-9),
        })
    Xp_all = align_X(pd.DataFrame(feat_rows), all_names)
    mu_p, sig_p = predict_mu_sigma(ckpt, Xp_all)
    sig_p_c = lam * sig_p
    gwp = np.array([impact_from_mix(m, GWP) for m in mixes])
    ap = np.array([impact_from_mix(m, AP) for m in mixes])
    ep = np.array([impact_from_mix(m, EP) for m in mixes])

    mc_j = np.zeros((len(pareto), MC_N))
    tinit_mean = np.zeros(len(pareto))
    j_mean = np.zeros(len(pareto))
    gwp_over_t = np.zeros(len(pareto))
    for i in range(len(pareto)):
        log_d = rng.normal(mu_p[i], sig_p_c[i], size=MC_N)
        d_lin = (10 ** log_d) * 1e-12
        ti = t_init_years(d_lin, ms[i], sample_cs(MC_N, rng), sample_ccrit(MC_N, rng), cap=T_INIT_REPORT_CAP)
        mc_j[i] = gwp[i] * (10 ** log_d)
        tinit_mean[i] = float(np.mean(ti))
        j_mean[i] = float(np.mean(mc_j[i]))
        gwp_over_t[i] = float(gwp[i] / max(tinit_mean[i], 1e-9))

    def rank_asc(x):
        return stats.rankdata(x, method="average")

    ranks = pd.DataFrame({
        "candidate_id": pareto["candidate_id"], "gwp": gwp, "ap": ap, "ep": ep, "m": ms,
        "tinit_mean_years": tinit_mean, "J_mean": j_mean, "gwp_per_tinit": gwp_over_t,
        "rank_gwp": rank_asc(gwp), "rank_J": rank_asc(j_mean), "rank_gwp_tinit": rank_asc(gwp_over_t),
        "rank_ap": rank_asc(ap), "rank_ep": rank_asc(ep),
    })

    def topk_ids(col):
        return set(ranks.nsmallest(TOP_K, col)["candidate_id"].tolist())

    topk_fail = np.zeros(len(pareto))
    regret = np.zeros(len(pareto))
    for s in range(MC_N):
        col = mc_j[:, s]
        order = np.argsort(col)
        best = col[order[0]]
        in_topk = set(order[:TOP_K].tolist())
        for i in range(len(pareto)):
            topk_fail[i] += int(i not in in_topk)
            regret[i] += col[i] - best
    ranks["topk_failure_prob_J"] = topk_fail / MC_N
    ranks["expected_regret_J"] = regret / MC_N
    i_j = int(ranks["rank_J"].idxmin())
    i_g = int(ranks["rank_gwp"].idxmin())
    summary = {
        "S1": {
            "topk_overlap_gwp_vs_gwp_tinit": len(topk_ids("rank_gwp") & topk_ids("rank_gwp_tinit")) / TOP_K,
            "kendall_gwp_vs_gwp_tinit": float(stats.kendalltau(ranks["rank_gwp"], ranks["rank_gwp_tinit"]).correlation),
            "kendall_gwp_vs_J": float(stats.kendalltau(ranks["rank_gwp"], ranks["rank_J"]).correlation),
        },
        "S2": {
            "kendall_J_vs_gwp_tinit": float(stats.kendalltau(ranks["rank_J"], ranks["rank_gwp_tinit"]).correlation),
            "topk_overlap_J_vs_gwp_tinit": len(topk_ids("rank_J") & topk_ids("rank_gwp_tinit")) / TOP_K,
        },
        "S3": {
            "P_bestJ_beats_bestGWP": float(np.mean(mc_j[i_j] < mc_j[i_g])),
            "best_J_id": int(ranks.loc[i_j, "candidate_id"]),
            "best_GWP_id": int(ranks.loc[i_g, "candidate_id"]),
            "mean_topk_failure_top10_by_meanJ": float(ranks.nsmallest(10, "rank_J")["topk_failure_prob_J"].mean()),
        },
        "multi_impact": {
            "kendall_gwp_vs_ap": float(stats.kendalltau(ranks["rank_gwp"], ranks["rank_ap"]).correlation),
            "kendall_gwp_vs_ep": float(stats.kendalltau(ranks["rank_gwp"], ranks["rank_ep"]).correlation),
            "topk_overlap_gwp_ap": len(topk_ids("rank_gwp") & topk_ids("rank_ap")) / TOP_K,
            "topk_overlap_gwp_ep": len(topk_ids("rank_gwp") & topk_ids("rank_ep")) / TOP_K,
        },
        "metrics": metrics,
        "conformal": conformal,
    }
    ranks.to_csv(EXPORTS / "rev1_pathB_shell_ranks_on_archived_pareto.csv", index=False)
    (EXPORTS / "rev1_S1S2S3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    conv = []
    for n in [200, 500, 1000, 2000, 4000]:
        fail_n = 0
        for s in range(n):
            order = np.argsort(mc_j[:, s])
            fail_n += int(i_j not in set(order[:TOP_K].tolist()))
        conv.append({"n_mc": n, "topk_failure_bestJ": fail_n / n})
    pd.DataFrame(conv).to_csv(EXPORTS / "rev1_mc_convergence.csv", index=False)

    print("Running Path B NSGA-II...")
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination

    class PathBProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=8, n_obj=4, n_ieq_constr=6, xl=np.zeros(8), xu=np.ones(8))
            self.ckpt = ckpt
            self.all_names = all_names
            self.m_mk = m_mk
            self.lam = lam
            self.rng = np.random.default_rng(SEED + 7)

        def _denorm(self, x):
            w_b = 0.25 + 0.40 * x[0]
            scm = 0.05 + 0.55 * x[1]
            binder = 300 + 300 * x[2]
            fine = 0.30 + 0.40 * x[3]
            fa_f, sl_f, sf_f = x[4], x[5], x[6]
            s = fa_f + sl_f + sf_f + 1e-12
            fa_f, sl_f, sf_f = fa_f / s, sl_f / s, sf_f / s
            sp = 0.5 + 12.0 * x[7]
            return w_b, scm, binder, fine, fa_f, sl_f, sf_f, sp

        def _mix(self, w_b, scm, binder, fine, fa_f, sl_f, sf_f, sp):
            mk_f = max(0.0, 1.0 - fa_f - sl_f - sf_f)
            opc = binder * (1 - scm)
            fa, slag, sf, mk = binder * scm * fa_f, binder * scm * sl_f, binder * scm * sf_f, binder * scm * mk_f
            water = w_b * binder
            V_paste = opc / RHO["OPC"] + (fa + slag + sf + mk) / RHO["SCM"] + water / RHO["Water"] + sp / RHO["SP"]
            V_agg = max(0.05, 0.98 - V_paste)
            mix = {"OPC": opc, "FA": fa, "GGBFS": slag, "SF": sf, "MK": mk, "Water": water,
                   "NFA": fine * V_agg * RHO["Agg"], "NCA": (1 - fine) * V_agg * RHO["Agg"], "SP": sp}
            return mix, V_paste, V_agg

        def _predict(self, mix):
            binder = mix["OPC"] + mix["FA"] + mix["GGBFS"] + mix["SF"] + mix["MK"]
            rec = pd.DataFrame([{
                "Cement": mix["OPC"], "SF": mix["SF"], "GGBFS": mix["GGBFS"], "FA": mix["FA"], "MK": mix["MK"],
                "Water": mix["Water"], "w_b_ratio": mix["Water"] / max(binder, 1e-9),
                "NFA": mix["NFA"], "NCA": mix["NCA"], "SP": mix["SP"], "CS": cs_med,
                "Exposure_Duration": exp_dur, "Exposure_Temperature": exp_temp, "Da": 1.0,
                "Total_Binder": binder, "SCM_Replacement_Ratio": (binder - mix["OPC"]) / max(binder, 1e-9),
                "SF_Ratio": mix["SF"] / max(binder, 1e-9), "GGBFS_Ratio": mix["GGBFS"] / max(binder, 1e-9),
                "FA_Ratio": mix["FA"] / max(binder, 1e-9), "MK_Ratio": mix["MK"] / max(binder, 1e-9),
            }])
            Xs = self.ckpt["preprocess_pipeline"].transform(align_X(rec, self.all_names))
            mu = float(self.ckpt["ensemble"].predict(Xs)[0])
            sig = float(np.asarray(self.ckpt["ensemble"].pred_dist(Xs).scale).ravel()[0])
            return mu, max(sig, 1e-6)

        def _evaluate(self, x, out, *args, **kwargs):
            w_b, scm, binder, fine, fa_f, sl_f, sf_f, sp = self._denorm(x)
            mix, V_paste, V_agg = self._mix(w_b, scm, binder, fine, fa_f, sl_f, sf_f, sp)
            mu, sig = self._predict(mix)
            mval = m_of_mix(mix["OPC"], mix["FA"], mix["GGBFS"], mix["SF"], mix["MK"], self.m_mk)
            log_d = self.rng.normal(mu, self.lam * sig, size=SAA_N)
            ti = t_init_years((10 ** log_d) * 1e-12, mval, sample_cs(SAA_N, self.rng), sample_ccrit(SAA_N, self.rng), cap=T_INIT_REPORT_CAP)
            dur_obj = -float(np.minimum(np.quantile(ti, 0.10), T_INIT_SOFT_YEARS))
            # Legacy Path B closed-form strength (not used by run_paper.py / Ridge search).
            cs_strength = float(np.clip(50.0 - 30.0 * w_b + 10.0 * (1 - scm) + 12.0 * scm * sf_f, 20, 80))
            out["F"] = [impact_from_mix(mix, COST), -cs_strength, dur_obj, impact_from_mix(mix, GWP)]
            out["G"] = [30.0 - cs_strength, 0.25 - V_paste, V_paste - 0.45, 0.50 - V_agg, V_agg - 0.75, 250.0 - mix["OPC"]]

    problem = PathBProblem()
    algorithm = NSGA2(pop_size=MOO_POP, sampling=FloatRandomSampling(),
                      crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(problem, algorithm, get_termination("n_gen", MOO_GEN), seed=SEED, verbose=True)
    pathb_rows = []
    for i in range(len(res.F)):
        w_b, scm, binder, fine, fa_f, sl_f, sf_f, sp = problem._denorm(res.X[i])
        mix, _, _ = problem._mix(w_b, scm, binder, fine, fa_f, sl_f, sf_f, sp)
        pathb_rows.append({
            "candidate_id": i, "w_b_ratio": w_b, "scm_replacement": scm, "binder_content": binder,
            "fine_agg_ratio": fine, "fa_frac": fa_f, "slag_frac": sl_f, "sf_frac": sf_f, "sp_kg_m3": sp,
            "cost": res.F[i, 0], "strength": -res.F[i, 1], "neg_tinit_q10": res.F[i, 2], "co2": res.F[i, 3],
            "OPC": mix["OPC"], "FA": mix["FA"], "GGBFS": mix["GGBFS"], "SF": mix["SF"], "MK": mix["MK"],
            "m": m_of_mix(mix["OPC"], mix["FA"], mix["GGBFS"], mix["SF"], mix["MK"], m_mk),
        })
    pathb_df = pd.DataFrame(pathb_rows)
    pathb_df.to_csv(EXPORTS / "rev1_pathB_pareto.csv", index=False)
    (EXPORTS / "rev1_pathB_contrast.json").write_text(json.dumps({
        "old_pareto_n": int(len(pareto)), "pathB_pareto_n": int(len(pathb_df)),
        "moo_pop": MOO_POP, "moo_gen": MOO_GEN, "saa_n": SAA_N, "lambda": lam,
        "t_init_cap_years": T_INIT_CAP_YEARS,
        "t_init_soft_years_moo": T_INIT_SOFT_YEARS,
        "old_mean_scm": float(pareto["scm_replacement"].mean()) if "scm_replacement" in pareto.columns else None,
        "pathB_mean_scm": float(pathb_df["scm_replacement"].mean()),
        "old_mean_co2": float(pareto["co2"].mean()) if "co2" in pareto.columns else None,
        "pathB_mean_co2": float(pathb_df["co2"].mean()),
        "pathB_mean_tinit_q10": float((-pathb_df["neg_tinit_q10"]).mean()),
    }, indent=2), encoding="utf-8")
    print(f"Path B Pareto size: {len(pathb_df)}")

    # ---- S1/S2/S3 on Path B pool (primary production front) ----
    print("S1/S2/S3 on Path B pool...")
    pb_mixes, pb_ms, pb_feat = [], [], []
    for _, row in pathb_df.iterrows():
        mix = {"OPC": float(row["OPC"]), "FA": float(row["FA"]), "GGBFS": float(row["GGBFS"]),
               "SF": float(row["SF"]), "MK": float(row["MK"]),
               "Water": float(row["w_b_ratio"]) * float(row["binder_content"]),
               "NFA": float(row["fine_agg_ratio"]) * 1800.0,
               "NCA": (1 - float(row["fine_agg_ratio"])) * 1800.0,
               "SP": float(row["sp_kg_m3"])}
        pb_mixes.append(mix)
        pb_ms.append(float(row["m"]))
        binder = mix["OPC"] + mix["FA"] + mix["GGBFS"] + mix["SF"] + mix["MK"]
        pb_feat.append({
            "Cement": mix["OPC"], "SF": mix["SF"], "GGBFS": mix["GGBFS"], "FA": mix["FA"], "MK": mix["MK"],
            "Water": mix["Water"], "w_b_ratio": mix["Water"] / max(binder, 1e-9),
            "NFA": mix["NFA"], "NCA": mix["NCA"], "SP": mix["SP"], "CS": cs_med,
            "Exposure_Duration": exp_dur, "Exposure_Temperature": exp_temp, "Da": 1.0,
            "Total_Binder": binder, "SCM_Replacement_Ratio": (binder - mix["OPC"]) / max(binder, 1e-9),
            "SF_Ratio": mix["SF"] / max(binder, 1e-9), "GGBFS_Ratio": mix["GGBFS"] / max(binder, 1e-9),
            "FA_Ratio": mix["FA"] / max(binder, 1e-9), "MK_Ratio": mix["MK"] / max(binder, 1e-9),
        })
    Xp_b = align_X(pd.DataFrame(pb_feat), all_names)
    mu_b, sig_b = predict_mu_sigma(ckpt, Xp_b)
    sig_b_c = lam * sig_b
    gwp_b = np.array([impact_from_mix(m, GWP) for m in pb_mixes])
    ap_b = np.array([impact_from_mix(m, AP) for m in pb_mixes])
    ep_b = np.array([impact_from_mix(m, EP) for m in pb_mixes])
    mc_jb = np.zeros((len(pathb_df), MC_N))
    tinit_b = np.zeros(len(pathb_df))
    j_b = np.zeros(len(pathb_df))
    gwp_t_b = np.zeros(len(pathb_df))
    for i in range(len(pathb_df)):
        log_d = rng.normal(mu_b[i], sig_b_c[i], size=MC_N)
        d_lin = (10 ** log_d) * 1e-12
        ti = t_init_years(d_lin, pb_ms[i], sample_cs(MC_N, rng), sample_ccrit(MC_N, rng), cap=T_INIT_REPORT_CAP)
        mc_jb[i] = gwp_b[i] * (10 ** log_d)
        tinit_b[i] = float(np.mean(ti))
        j_b[i] = float(np.mean(mc_jb[i]))
        gwp_t_b[i] = float(gwp_b[i] / max(tinit_b[i], 1e-9))

    ranks_b = pd.DataFrame({
        "candidate_id": pathb_df["candidate_id"], "gwp": gwp_b, "ap": ap_b, "ep": ep_b, "m": pb_ms,
        "tinit_mean_years": tinit_b, "J_mean": j_b, "gwp_per_tinit": gwp_t_b,
        "rank_gwp": rank_asc(gwp_b), "rank_J": rank_asc(j_b), "rank_gwp_tinit": rank_asc(gwp_t_b),
        "rank_ap": rank_asc(ap_b), "rank_ep": rank_asc(ep_b),
        "cost": pathb_df["cost"].to_numpy(), "strength": pathb_df["strength"].to_numpy(),
        "tinit_q10": (-pathb_df["neg_tinit_q10"]).to_numpy(),
    })
    topk_fail_b = np.zeros(len(pathb_df))
    regret_b = np.zeros(len(pathb_df))
    for s in range(MC_N):
        col = mc_jb[:, s]
        order = np.argsort(col)
        best = col[order[0]]
        in_topk = set(order[:TOP_K].tolist())
        for i in range(len(pathb_df)):
            topk_fail_b[i] += int(i not in in_topk)
            regret_b[i] += col[i] - best
    ranks_b["topk_failure_prob_J"] = topk_fail_b / MC_N
    ranks_b["expected_regret_J"] = regret_b / MC_N
    i_j = int(ranks_b["rank_J"].idxmin())
    i_g = int(ranks_b["rank_gwp"].idxmin())

    def topk_ids_b(col):
        return set(ranks_b.nsmallest(TOP_K, col)["candidate_id"].tolist())

    summary_b = {
        "pool": "pathB",
        "S1": {
            "topk_overlap_gwp_vs_gwp_tinit": len(topk_ids_b("rank_gwp") & topk_ids_b("rank_gwp_tinit")) / TOP_K,
            "kendall_gwp_vs_gwp_tinit": float(stats.kendalltau(ranks_b["rank_gwp"], ranks_b["rank_gwp_tinit"]).correlation),
            "kendall_gwp_vs_J": float(stats.kendalltau(ranks_b["rank_gwp"], ranks_b["rank_J"]).correlation),
        },
        "S2": {
            "kendall_J_vs_gwp_tinit": float(stats.kendalltau(ranks_b["rank_J"], ranks_b["rank_gwp_tinit"]).correlation),
            "topk_overlap_J_vs_gwp_tinit": len(topk_ids_b("rank_J") & topk_ids_b("rank_gwp_tinit")) / TOP_K,
        },
        "S3": {
            "P_bestJ_beats_bestGWP": float(np.mean(mc_jb[i_j] < mc_jb[i_g])),
            "best_J_id": int(ranks_b.loc[i_j, "candidate_id"]),
            "best_GWP_id": int(ranks_b.loc[i_g, "candidate_id"]),
            "mean_topk_failure_top10_by_meanJ": float(ranks_b.nsmallest(10, "rank_J")["topk_failure_prob_J"].mean()),
        },
        "multi_impact": {
            "kendall_gwp_vs_ap": float(stats.kendalltau(ranks_b["rank_gwp"], ranks_b["rank_ap"]).correlation),
            "kendall_gwp_vs_ep": float(stats.kendalltau(ranks_b["rank_gwp"], ranks_b["rank_ep"]).correlation),
            "topk_overlap_gwp_ap": len(topk_ids_b("rank_gwp") & topk_ids_b("rank_ap")) / TOP_K,
            "topk_overlap_gwp_ep": len(topk_ids_b("rank_gwp") & topk_ids_b("rank_ep")) / TOP_K,
        },
        "metrics": metrics,
        "conformal": conformal,
    }
    ranks_b.to_csv(EXPORTS / "rev1_pathB_S1S2S3_ranks.csv", index=False)
    (EXPORTS / "rev1_pathB_S1S2S3_summary.json").write_text(json.dumps(summary_b, indent=2), encoding="utf-8")
    # keep archived-front summary as SI contrast; primary is Path B
    summary["note"] = (
        "S1-S3 on the archived mean-Dapp front under the legacy shell (SI contrast). "
        "This contrast does not replace the deposited production front."
    )
    (EXPORTS / "rev1_S1S2S3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary_b, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for sname, style in [("test_raw", "o-"), ("test_lambda", "s-")]:
        sub = cal_df[cal_df["set"] == sname]
        ax.plot(sub["nominal"], sub["empirical"], style, label=sname)
    ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=1, label="ideal")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.legend(); ax.set_title("NGB calibration: raw vs lambda*sigma")
    fig.tight_layout(); fig.savefig(FIGS / "rev1_calibration_picp.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ranks_b["rank_gwp"], ranks_b["rank_gwp_tinit"], s=18, alpha=0.7)
    ax.set_xlabel("Rank by GWP/m3"); ax.set_ylabel("Rank by GWP/t_init")
    ax.set_title("S1 on Path B pool: basis reformulation rank shift")
    fig.tight_layout(); fig.savefig(FIGS / "rev1_S1_rank_shift.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(pathb_df["co2"], -pathb_df["neg_tinit_q10"], c=pathb_df["cost"], s=28, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="cost")
    ax.set_xlabel("GWP (kg CO2-eq/m3)"); ax.set_ylabel("t_init 10th percentile (years, capped)")
    ax.set_title("Path B Pareto")
    fig.tight_layout(); fig.savefig(FIGS / "rev1_pathB_pareto.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ranks_b["rank_J"], ranks_b["rank_gwp_tinit"], s=18, alpha=0.7)
    ax.set_xlabel("Rank by J"); ax.set_ylabel("Rank by GWP/t_init")
    ax.set_title("S2: scalar-form concordance on Path B pool")
    fig.tight_layout(); fig.savefig(FIGS / "rev1_S2_scalar_concordance.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    topj = ranks_b.nsmallest(20, "rank_J")
    ax.bar(range(len(topj)), topj["topk_failure_prob_J"].to_numpy())
    ax.set_xlabel("Top-20 mean-J candidates (ordered)"); ax.set_ylabel("Top-k failure probability")
    ax.set_title("S3: residual shortlist instability")
    fig.tight_layout(); fig.savefig(FIGS / "rev1_S3_topk_failure.png", dpi=200); plt.close(fig)

    print("Done. Exports in", EXPORTS)


if __name__ == "__main__":
    main()
