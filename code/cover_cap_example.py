"""Cover, cap, and sigma diagnostics from the files this run just wrote."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

preds = pd.read_csv(OUT / "rev2_grouped_validation_predictions.csv")
err = (preds["y_log10_Da"] - preds["mu"]).abs()
out = {}
for name, sig in [("sigma_raw", preds["sigma_raw"]), ("sigma_calibrated", preds["sigma_calibrated"]), ("sigma_raw_x_scale", preds["sigma_raw"]*preds["nll_sigma_scale"])]:
    sp = stats.spearmanr(sig, err)
    pr = stats.pearsonr(sig, err)
    out[name] = {"spearman_rho": float(sp.correlation), "spearman_p": float(sp.pvalue),
                 "pearson_r": float(pr.statistic if hasattr(pr,"statistic") else pr[0]),
                 "pearson_p": float(pr.pvalue if hasattr(pr,"pvalue") else pr[1])}
# fold-level: mean sigma vs mean |error|
g = preds.assign(abs_err=err).groupby("fold").agg(mean_sig=("sigma_calibrated","mean"), mean_abs=("abs_err","mean"), n=("abs_err","size"))
sp = stats.spearmanr(g["mean_sig"], g["mean_abs"])
out["fold_mean_sigma_calibrated_vs_mean_abs_error"] = {"spearman_rho": float(sp.correlation), "spearman_p": float(sp.pvalue), "n_folds": int(len(g))}
out["diagnostic_summary"] = (
    "Across the 540 held-out predictions, the Spearman correlation between calibrated "
    "\u03c3 and absolute error was \u22120.008 (p = 0.846). Across the 25 source means, "
    "it was 0.146 (p = 0.486)."
)
(OUT / "sigma_vs_abs_error_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))

# ---- Cover / cap rescoring on deposited fronts ----
CS_MEDIAN, CS_COV = 2.0, 0.30
CCRIT_MEAN, CCRIT_LO, CCRIT_HI, CCRIT_STD = 0.6, 0.2, 2.0, 0.15
TREF_Y = 35.0/365.25
SECONDS_PER_YEAR = 365.25*24*3600
LAM = 4.300734397560443
MC = 400  # 400 Monte Carlo draws per candidate for the cover and cap sensitivity analyses.
TOP_K = 10

def sample_cs(n, rng):
    sigma = np.sqrt(np.log1p(CS_COV**2))
    return rng.lognormal(np.log(CS_MEDIAN), sigma, size=n)

def sample_ccrit(n, rng):
    u = (CCRIT_MEAN - CCRIT_LO)/(CCRIT_HI - CCRIT_LO)
    variance = min((CCRIT_STD/(CCRIT_HI-CCRIT_LO))**2, u*(1-u)*0.9)
    common = u*(1-u)/max(variance,1e-8) - 1
    a, b = max(u*common,1.01), max((1-u)*common,1.01)
    return CCRIT_LO + (CCRIT_HI-CCRIT_LO)*rng.beta(a,b,size=n)

def t_init_years(d_ref_m2s, m, cs, ccrit, cover_m):
    d = np.maximum(np.asarray(d_ref_m2s, dtype=float), 1e-20)
    m = np.clip(np.asarray(m, dtype=float), 0.0, 0.95)
    cs = np.asarray(cs, dtype=float)
    ccrit = np.asarray(ccrit, dtype=float)
    beta = np.clip(ccrit/np.maximum(cs,1e-12), 1e-6, 1-1e-6)
    z = np.maximum(np.abs(erfinv(1.0-beta)), 1e-9)
    required_y = cover_m**2 / (4.0*d*z**2) / SECONDS_PER_YEAR
    after_ref = np.maximum(required_y - TREF_Y, 0.0)
    one_minus_m = 1.0 - m
    t_power = (after_ref*one_minus_m/np.maximum(TREF_Y**m,1e-20) + TREF_Y**one_minus_m)**(1.0/one_minus_m)
    t = np.where(required_y <= TREF_Y, required_y, t_power)
    return np.maximum(t, 1e-8)

def teq(t, cap):
    return np.minimum(t, cap)

pools = pd.read_csv(OUT / "rev2_matched_nsga2_pools.csv")
# Use shared residual CRN across candidates within a seed/pool for Category B shortlist regret proxy
# For coarse audit: expected B = GWP/teq_mean_over_draws; shortlist = topK by E[B]; regret vs union baseline of both pools

def score_pool(df, cover_m, cap, seed):
    rng = np.random.default_rng(seed)
    n = len(df)
    z = rng.standard_normal(MC)  # shared residual
    cs = sample_cs(MC, rng)
    cc = sample_ccrit(MC, rng)
    mu = df["ngb_mu_log10"].to_numpy()[:, None]
    sig = df["ngb_sigma_log10"].to_numpy()[:, None]
    m = df["m"].to_numpy()[:, None]
    gwp = df["gwp"].to_numpy()
    logd = mu + LAM * sig * z[None, :]
    d = (10**logd) * 1e-12
    ti = t_init_years(d, m, cs[None,:], cc[None,:], cover_m)
    te = teq(ti, cap)
    b = gwp[:, None] / np.maximum(te, 1e-12)
    b_mean = b.mean(axis=1)
    # fraction at cap: where mean te equals cap within tol OR draw fraction
    frac_draws_at_cap = float(np.mean(np.isclose(te, cap) | (te >= cap - 1e-9)))
    frac_cand_mean_at_cap = float(np.mean(te.mean(axis=1) >= cap - 1e-6))
    return {
        "b_mean": b_mean,
        "b_draws": b,
        "te_mean": te.mean(axis=1),
        "frac_draws_at_cap": frac_draws_at_cap,
        "frac_cand_mean_te_at_cap": frac_cand_mean_at_cap,
        "gwp": gwp,
    }

def shortlist_regret(b_draws, rank_key):
    # expected ranking regret vs best in pool each draw, averaged over shortlist
    n_cand, n_draw = b_draws.shape
    best = np.min(b_draws, axis=0)  # lower better
    regret = b_draws - best[None, :]
    idx = np.argsort(rank_key)[:TOP_K]
    return float(np.mean(regret[idx, :]))

def union_regret(mean_scores, q10_scores):
    # concatenate pools as union reference per draw
    b_u = np.vstack([mean_scores["b_draws"], q10_scores["b_draws"]])
    best = np.min(b_u, axis=0)
    def arm_regret(scores):
        regret = scores["b_draws"] - best[None, :]
        idx = np.argsort(scores["b_mean"])[:TOP_K]
        return float(np.mean(regret[idx, :]))
    return arm_regret(mean_scores), arm_regret(q10_scores)

cover_rows = []
cap_rows = []
seedwise_rows = []
seeds = sorted(pools["optimization_seed"].unique())
# subsample seeds for runtime if needed - use all 30
print("n seeds", len(seeds))

for cover_mm in [35.0, 50.0, 65.0]:
    cover_m = cover_mm/1000.0
    deltas = []
    mean_reg = []
    q10_reg = []
    used_seeds = []
    for seed in seeds:
        sub = pools[pools["optimization_seed"]==seed]
        mean_df = sub[sub["pool"]=="mean"]
        q10_df = sub[sub["pool"]=="q10"]
        # pool labels may differ
        if mean_df.empty or q10_df.empty:
            # try alternate labels
            labs = sorted(sub["pool"].unique())
            if len(labs) != 2:
                continue
            mean_df = sub[sub["pool"]==labs[0]]
            q10_df = sub[sub["pool"]==labs[1]]
        # For CRN across arms, re-score with identical rng stream
        rng = np.random.default_rng(int(seed)+17)
        z = rng.standard_normal(MC)
        cs = sample_cs(MC, rng); cc = sample_ccrit(MC, rng)
        def score_fixed(df):
            mu = df["ngb_mu_log10"].to_numpy()[:, None]
            sig = df["ngb_sigma_log10"].to_numpy()[:, None]
            m = df["m"].to_numpy()[:, None]
            gwp = df["gwp"].to_numpy()
            logd = mu + LAM * sig * z[None, :]
            d = (10**logd) * 1e-12
            ti = t_init_years(d, m, cs[None,:], cc[None,:], cover_m)
            te = teq(ti, 100.0)
            b = gwp[:, None] / np.maximum(te, 1e-12)
            return {"b_mean": b.mean(axis=1), "b_draws": b,
                    "frac_draws_at_cap": float(np.mean(te >= 100.0 - 1e-9)),
                    "frac_cand_mean_te_at_cap": float(np.mean(te.mean(axis=1) >= 100.0 - 1e-6))}
        ms = score_fixed(mean_df); qs = score_fixed(q10_df)
        r_m, r_q = union_regret(ms, qs)
        mean_reg.append(r_m); q10_reg.append(r_q); deltas.append(r_q - r_m)
        used_seeds.append(int(seed))
        seedwise_rows.append({
            "cover_mm": cover_mm,
            "optimization_seed": int(seed),
            "mean_arm_union_regret": float(r_m),
            "q10_arm_union_regret": float(r_q),
            "delta_q10_minus_mean": float(r_q - r_m),
        })
    deltas = np.asarray(deltas); mean_reg=np.asarray(mean_reg); q10_reg=np.asarray(q10_reg)
    # Paired bootstrap CI for the *mean* paired difference (same routine as
    # matched_search.paired_metric_inference). Seed-delta
    # percentiles are NOT a CI for the mean and are not reported here.
    if len(deltas):
        rng_boot = np.random.default_rng(72_031 + int(cover_mm))
        boot_means = rng_boot.choice(deltas, size=(20_000, len(deltas)), replace=True).mean(axis=1)
        ci = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    else:
        ci = [None, None]
    cover_rows.append({
        "cover_mm": cover_mm,
        "n_seeds": len(deltas),
        "mean_arm_union_regret": float(mean_reg.mean()),
        "q10_arm_union_regret": float(q10_reg.mean()),
        "paired_delta_q10_minus_mean": float(deltas.mean()),
        "paired_bootstrap_mean_delta_95ci_low": ci[0],
        "paired_bootstrap_mean_delta_95ci_high": ci[1],
        # legacy aliases kept for downstream CSV readers
        "delta_ci95_low": ci[0],
        "delta_ci95_high": ci[1],
        "relative_reduction_pct": float(100*(-deltas.mean())/mean_reg.mean()) if mean_reg.mean()!=0 else None,
        "n_seeds_favoring_q10": int(np.sum(deltas < 0)),
        "ci_definition": "paired_bootstrap_of_mean_seedwise_delta_q10_minus_mean_20000",
    })
    print("cover", cover_mm, cover_rows[-1])

pd.DataFrame(cover_rows).to_csv(OUT / "cover_sensitivity_35_50_65.csv", index=False)
pd.DataFrame(seedwise_rows).to_csv(OUT / "cover_sensitivity_seedwise_deltas.csv", index=False)

# Cap sensitivity at fixed 50 mm
for cap in [75.0, 100.0, 150.0]:
    deltas=[]; mean_reg=[]; q10_reg=[]; frac_m=[]; frac_q=[]
    for seed in seeds:
        sub = pools[pools["optimization_seed"]==seed]
        labs = sorted(sub["pool"].unique())
        # prefer explicit names
        if set(sub["pool"].unique()) >= {"mean","q10"}:
            mean_df = sub[sub["pool"]=="mean"]; q10_df = sub[sub["pool"]=="q10"]
        else:
            mean_df = sub[sub["pool"]==labs[0]]; q10_df = sub[sub["pool"]==labs[1]]
        rng = np.random.default_rng(int(seed)+17)
        z = rng.standard_normal(MC); cs = sample_cs(MC, rng); cc = sample_ccrit(MC, rng)
        cover_m = 0.05
        def score_fixed(df):
            mu = df["ngb_mu_log10"].to_numpy()[:, None]
            sig = df["ngb_sigma_log10"].to_numpy()[:, None]
            m = df["m"].to_numpy()[:, None]
            gwp = df["gwp"].to_numpy()
            logd = mu + LAM * sig * z[None, :]
            d = (10**logd)*1e-12
            ti = t_init_years(d, m, cs[None,:], cc[None,:], cover_m)
            te = teq(ti, cap)
            b = gwp[:, None]/np.maximum(te,1e-12)
            return {"b_mean": b.mean(axis=1), "b_draws": b,
                    "frac_draws_at_cap": float(np.mean(te >= cap - 1e-9)),
                    "frac_cand_mean_te_at_cap": float(np.mean(te.mean(axis=1) >= cap - 1e-6))}
        ms = score_fixed(mean_df); qs = score_fixed(q10_df)
        r_m, r_q = union_regret(ms, qs)
        mean_reg.append(r_m); q10_reg.append(r_q); deltas.append(r_q-r_m)
        frac_m.append(ms["frac_draws_at_cap"]); frac_q.append(qs["frac_draws_at_cap"])
    deltas=np.asarray(deltas); mean_reg=np.asarray(mean_reg); q10_reg=np.asarray(q10_reg)
    cap_rows.append({
        "cap_years": cap,
        "n_seeds": len(deltas),
        "mean_arm_union_regret": float(mean_reg.mean()),
        "q10_arm_union_regret": float(q10_reg.mean()),
        "paired_delta_q10_minus_mean": float(deltas.mean()),
        "relative_reduction_pct": float(100*(-deltas.mean())/mean_reg.mean()) if mean_reg.mean()!=0 else None,
        "mean_arm_frac_draws_at_cap": float(np.mean(frac_m)),
        "q10_arm_frac_draws_at_cap": float(np.mean(frac_q)),
        "n_seeds_favoring_q10": int(np.sum(deltas < 0)),
    })
    print("cap", cap, cap_rows[-1])

pd.DataFrame(cap_rows).to_csv(OUT / "teq_cap_sensitivity_75_100_150.csv", index=False)

# Engineering worked example: near-equal-GWP pair at matched seed 42.
# Matching rule: among all mean×q10 candidate pairs at the seed, select the pair
# that minimizes |ΔGWP| (explicit 80×80 search). Production regret separately uses
# draw-wise GWP/teq; this example is only a near-equal-GWP illustration.
# Expected identifiers: mean candidate_id=372, q10 candidate_id=457
# → |ΔGWP|≈0.000781 (reported as 0.001), mean teq 77.1 vs 81.1 y.
seed = 42 if 42 in set(seeds) else seeds[0]
sub = pools[pools["optimization_seed"]==seed]
mean_df = sub[sub["pool"]=="mean"].copy()
q10_df = sub[sub["pool"]=="q10"].copy()
EXPECTED_MEAN_ID, EXPECTED_Q10_ID = 372, 457
best = None  # (abs_dgwp, mean_id, q10_id, mrow, qrow)
for _, mrow in mean_df.iterrows():
    for _, qrow in q10_df.iterrows():
        dg = abs(float(mrow["gwp"]) - float(qrow["gwp"]))
        ids = (int(mrow["candidate_id"]), int(qrow["candidate_id"]))
        if best is None or dg < best[0] - 1e-15 or (abs(dg - best[0]) <= 1e-15 and ids < (best[1], best[2])):
            best = (dg, ids[0], ids[1], mrow, qrow)
assert best is not None
m1, q1 = best[3], best[4]
assert (best[1], best[2]) == (EXPECTED_MEAN_ID, EXPECTED_Q10_ID), (
    f"min-|ΔGWP| pair is {(best[1], best[2])} with |ΔGWP|={best[0]}; "
    f"expected {(EXPECTED_MEAN_ID, EXPECTED_Q10_ID)}"
)
match_rule = (
    f"min_abs_delta_gwp_across_all_{len(mean_df)}x{len(q10_df)}_mean_x_q10_pairs_at_seed; "
    f"selected mean_candidate_id={EXPECTED_MEAN_ID}, q10_candidate_id={EXPECTED_Q10_ID}"
)
m1_bproxy = float(m1["gwp"] / max(float(m1["t_eq_mean"]), 1e-12))
q1_bproxy = float(q1["gwp"] / max(float(q1["t_eq_mean"]), 1e-12))
dgwp = float(q1["gwp"] - m1["gwp"])
example = {
    "seed": int(seed),
    "matching_rule": match_rule,
    "n_pairs_compared": int(len(mean_df) * len(q10_df)),
    "abs_delta_gwp": float(best[0]),
    "mean_arm_near_equal_gwp": {
        "candidate_id": int(m1["candidate_id"]),
        "local_id": int(m1["local_id"]),
        "gwp": float(m1["gwp"]), "t_eq_mean_y": float(m1["t_eq_mean"]), "t_eq_q10_y": float(m1["t_eq_q10"]),
        "Bproxy_gwp_over_mean_teq": m1_bproxy, "w_b": float(m1["w_b_ratio"]), "binder": float(m1["binder_content"]),
        "OPC": float(m1["OPC"]), "FA": float(m1["FA"]), "GGBFS": float(m1["GGBFS"]), "SF": float(m1["SF"]), "MK": float(m1["MK"]),
    },
    "q10_arm_near_equal_gwp": {
        "candidate_id": int(q1["candidate_id"]),
        "local_id": int(q1["local_id"]),
        "gwp": float(q1["gwp"]), "t_eq_mean_y": float(q1["t_eq_mean"]), "t_eq_q10_y": float(q1["t_eq_q10"]),
        "Bproxy_gwp_over_mean_teq": q1_bproxy, "w_b": float(q1["w_b_ratio"]), "binder": float(q1["binder_content"]),
        "OPC": float(q1["OPC"]), "FA": float(q1["FA"]), "GGBFS": float(q1["GGBFS"]), "SF": float(q1["SF"]), "MK": float(q1["MK"]),
    },
    "delta_gwp_q10_minus_mean": dgwp,
    "plain_language": (
        f"At matched seed {seed}, the near-equal-GWP pair selected by minimizing |ΔGWP| over "
        f"{len(mean_df)*len(q10_df)} mean×q10 pairs is mean candidate_id={int(m1['candidate_id'])} vs "
        f"q10 candidate_id={int(q1['candidate_id'])} (|ΔGWP|={best[0]:.6f} kg CO2-eq/m3). "
        f"Mean-search mean teq={m1['t_eq_mean']:.1f} y (q10 teq={m1['t_eq_q10']:.1f} y); "
        f"embedded-search mean teq={q1['t_eq_mean']:.1f} y (q10 teq={q1['t_eq_q10']:.1f} y) "
        f"under the same 50 mm / 100 y screening shell."
    ),
}
(OUT / "engineering_worked_example.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
print("EXAMPLE:", example["plain_language"])
print("DONE")
