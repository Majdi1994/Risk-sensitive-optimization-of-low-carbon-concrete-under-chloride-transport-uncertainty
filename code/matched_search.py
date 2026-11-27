#!/usr/bin/env python3
"""Matched mean-vs-q10 multi-objective search under calibrated diffusivity uncertainty.

Reuses the frozen NGB checkpoint and shared helpers; deposited artifacts use
``rev2_*`` names. Primary service variable: t_eq = min(t_init, 100 y).
Use ``--smoke`` for a fast integration test, or no flag for production.
``--reuse-pools`` rescores outputs/rev2_matched_nsga2_pools.csv (the deposited
candidate list behind the reported numbers).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfinv
from pymoo.core.problem import Problem
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chloride_pipeline import (  # noqa: E402
    AP,
    COST,
    DATA_PATH,
    EP,
    GWP,
    NGB_PATH,
    RHO,
    align_X,
    impact_from_mix,
    m_of_mix,
    mk_mean_from_workbook,
    predict_mu_sigma,
)
from analysis_specification import EXPORTS, REV, SPEC, write_specification  # noqa: E402
from strength_model import StrengthPredictor  # noqa: E402

FIGS = REV / "figures"
TREF_Y = SPEC.reference_age_days / 365.25
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
COVER_M = 0.050
CS_MEDIAN = 2.0
CS_COV = 0.30
CCRIT_MEAN, CCRIT_LO, CCRIT_HI, CCRIT_STD = 0.6, 0.2, 2.0, 0.15
TOP_K = SPEC.top_k
SCM_KEYS = ("FA", "GGBFS", "SF", "MK")


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas values to strict JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")


def sample_cs(n: int, rng: np.random.Generator) -> np.ndarray:
    sigma = np.sqrt(np.log1p(CS_COV**2))
    return rng.lognormal(np.log(CS_MEDIAN), sigma, size=n)


def sample_ccrit(n: int, rng: np.random.Generator) -> np.ndarray:
    u = (CCRIT_MEAN - CCRIT_LO) / (CCRIT_HI - CCRIT_LO)
    variance = min((CCRIT_STD / (CCRIT_HI - CCRIT_LO)) ** 2, u * (1 - u) * 0.9)
    common = u * (1 - u) / max(variance, 1e-8) - 1
    a, b = max(u * common, 1.01), max((1 - u) * common, 1.01)
    return CCRIT_LO + (CCRIT_HI - CCRIT_LO) * rng.beta(a, b, size=n)


def t_init_years(
    d_ref_m2s: np.ndarray,
    m: np.ndarray,
    cs: np.ndarray,
    ccrit: np.ndarray,
    freeze_years: float | None = None,
) -> np.ndarray:
    """Initiation time under a reference-age piecewise aging law.

    D is constant through SPEC.reference_age_days and follows
    D_ref*(t_ref/t)^m afterwards. The optional sensitivity freezes D at its
    25-year value. The inversion is analytic in cumulative diffusivity time
    and supports candidate-by-draw arrays.
    """
    d = np.maximum(np.asarray(d_ref_m2s, dtype=float), 1e-20)
    m = np.clip(np.asarray(m, dtype=float), 0.0, 0.95)
    cs = np.asarray(cs, dtype=float)
    ccrit = np.asarray(ccrit, dtype=float)
    beta = np.clip(ccrit / np.maximum(cs, 1e-12), 1e-6, 1 - 1e-6)
    z = np.maximum(np.abs(erfinv(1.0 - beta)), 1e-9)
    required_y = COVER_M**2 / (4.0 * d * z**2) / SECONDS_PER_YEAR
    after_ref = np.maximum(required_y - TREF_Y, 0.0)
    one_minus_m = 1.0 - m

    def aging_clock(t: np.ndarray) -> np.ndarray:
        return (TREF_Y**m) * (np.maximum(t, TREF_Y) ** one_minus_m - TREF_Y**one_minus_m) / one_minus_m

    t_power = (
        after_ref * one_minus_m / np.maximum(TREF_Y**m, 1e-20) + TREF_Y**one_minus_m
    ) ** (1.0 / one_minus_m)
    t = np.where(required_y <= TREF_Y, required_y, t_power)
    if freeze_years is not None:
        tf = float(freeze_years)
        clock_f = aging_clock(np.full_like(t, tf))
        d_rel_f = (TREF_Y / tf) ** m
        t_frozen = tf + np.maximum(after_ref - clock_f, 0.0) / np.maximum(d_rel_f, 1e-20)
        t = np.where((required_y > TREF_Y) & (t_power > tf), t_frozen, t)
    return np.maximum(t, 1e-8)


def t_eq(t_init: np.ndarray) -> np.ndarray:
    """The sole production service variable."""
    return np.minimum(np.asarray(t_init, dtype=float), SPEC.decision_horizon_years)


def load_lambda() -> tuple[float, str, dict[str, Any]]:
    """Read grouped-validation calibration when available, otherwise document fallback."""
    stable = EXPORTS / "rev2_production_calibration.json"
    candidates = ([stable] if stable.exists() else []) + sorted(
        EXPORTS.glob("grouped_loso_validation*")
    )
    keys = ("lambda_production", "production_lambda", "lambda_eff", "lambda")
    for path in candidates:
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                stack = [payload]
                while stack:
                    item = stack.pop()
                    if isinstance(item, dict):
                        for key in keys:
                            if key in item and np.isfinite(float(item[key])):
                                return float(item[key]), f"{path.name}:{key}", payload
                        stack.extend(item.values())
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                for key in keys:
                    if key in frame.columns:
                        vals = pd.to_numeric(frame[key], errors="coerce").dropna()
                        if len(vals):
                            return float(vals.iloc[-1]), f"{path.name}:{key}", {"rows": len(frame)}
        except (ValueError, TypeError, OSError, json.JSONDecodeError):
            continue
    return 2.20, "documented fallback (earlier production calibration, 2.20); rerun after grouped validation. This fallback is not the production lambda.", {}


def feature_record(mix: dict[str, float], cs_strength: float = 40.0) -> dict[str, float]:
    binder = sum(mix[k] for k in ("OPC",) + SCM_KEYS)
    return {
        "Cement": mix["OPC"],
        "SF": mix["SF"],
        "GGBFS": mix["GGBFS"],
        "FA": mix["FA"],
        "MK": mix["MK"],
        "Water": mix["Water"],
        "w_b_ratio": mix["Water"] / max(binder, 1e-12),
        "NFA": mix["NFA"],
        "NCA": mix["NCA"],
        "SP": mix["SP"],
        "CS": cs_strength,
        "Exposure_Duration": SPEC.reference_age_days,
        "Exposure_Temperature": 20.0,
        "Da": 1.0,
        "Total_Binder": binder,
        "SCM_Replacement_Ratio": sum(mix[k] for k in SCM_KEYS) / max(binder, 1e-12),
        "SF_Ratio": mix["SF"] / max(binder, 1e-12),
        "GGBFS_Ratio": mix["GGBFS"] / max(binder, 1e-12),
        "FA_Ratio": mix["FA"] / max(binder, 1e-12),
        "MK_Ratio": mix["MK"] / max(binder, 1e-12),
    }


def legacy_surrogate_strength(w_b: float, scm: float, sf_frac: float) -> float:
    """Legacy closed-form strength comparator (not used in production search)."""
    return float(np.clip(50 - 30 * w_b + 10 * (1 - scm) + 12 * scm * sf_frac, 20, 80))


# Backward-compatible alias for older call sites.
superseded_surrogate_strength = legacy_surrogate_strength


_STRENGTH_PREDICTOR: StrengthPredictor | None = None


def strength_predictor() -> StrengthPredictor:
    """Lazily load the source-validated candidate-strength model."""
    global _STRENGTH_PREDICTOR
    if _STRENGTH_PREDICTOR is None:
        _STRENGTH_PREDICTOR = StrengthPredictor()
    return _STRENGTH_PREDICTOR


def derive_support(df: pd.DataFrame) -> dict[str, Any]:
    """Conservative active bounds from the model's ASTM C1556 training support."""
    qlo, qhi = 0.01, 0.99
    result: dict[str, Any] = {
        "derivation": "ASTM C1556 model-training rows; 1st/99th percentiles, component caps also bounded by frozen specification",
        "n": int(len(df)),
        "w_b_min": max(0.25, float(df["w_b_ratio"].quantile(qlo))),
        "w_b_max": min(0.65, float(df["w_b_ratio"].quantile(qhi))),
        "binder_min": max(300.0, float(df["Total_Binder"].quantile(qlo))),
        "binder_max": min(600.0, float(df["Total_Binder"].quantile(qhi))),
        "sp_min": max(0.0, float(df["SP"].quantile(qlo))),
        "sp_max": min(12.5, float(df["SP"].quantile(qhi))),
    }
    static = {
        "SCM_Replacement_Ratio": SPEC.max_scm_total_ratio,
        "FA_Ratio": SPEC.max_fa_ratio,
        "GGBFS_Ratio": SPEC.max_ggbfs_ratio,
        "SF_Ratio": SPEC.max_sf_ratio,
        "MK_Ratio": SPEC.max_mk_ratio,
    }
    caps = {}
    counts = {}
    for col, hard_cap in static.items():
        positive = pd.to_numeric(df[col], errors="coerce")
        positive = positive[positive > 1e-12]
        counts[col] = int(len(positive))
        empirical = float(positive.quantile(qhi)) if len(positive) else 0.0
        caps[col] = min(float(hard_cap), empirical)
    result["ratio_caps"] = caps
    result["positive_training_counts"] = counts
    return result


class ReconstructionProblem(Problem):
    """Vectorized pymoo problem with active empirical and kNN applicability gates."""

    def __init__(
        self,
        mode: str,
        saa_n: int,
        opt_seed: int,
        ckpt: dict[str, Any],
        all_names: list[str],
        support: dict[str, Any],
        scaler: StandardScaler,
        knn: NearestNeighbors,
        ood_limit: float,
        m_mk: float,
        lam: float,
    ) -> None:
        super().__init__(n_var=9, n_obj=4, n_ieq_constr=12, xl=np.zeros(9), xu=np.ones(9))
        self.mode, self.saa_n, self.opt_seed = mode, saa_n, opt_seed
        self.ckpt, self.all_names, self.support = ckpt, all_names, support
        self.scaler, self.knn, self.ood_limit = scaler, knn, ood_limit
        # Screen candidates on the same neighbour rank the threshold was built from.
        self.ood_k = int(support.get("ood", {}).get("neighbor_rank", 1))
        self.m_mk, self.lam = m_mk, lam
        self.strength = strength_predictor()
        rng = np.random.default_rng(1_000_003 + opt_seed)
        self.z = rng.standard_normal(saa_n)
        self.cs = sample_cs(saa_n, rng)
        self.ccrit = sample_ccrit(saa_n, rng)

    def decode(self, X: np.ndarray) -> tuple[list[dict[str, float]], pd.DataFrame]:
        X = np.atleast_2d(X)
        s = self.support
        mk_zero = float(SPEC.mk_practical_zero_kg_m3)
        rows, records = [], []
        for x in X:
            w_b = s["w_b_min"] + (s["w_b_max"] - s["w_b_min"]) * x[0]
            scm = SPEC.max_scm_total_ratio * x[1]
            binder = s["binder_min"] + (s["binder_max"] - s["binder_min"]) * x[2]
            fine = 0.30 + 0.40 * x[3]
            raw = np.maximum(x[4:8], 1e-12)
            frac = raw / raw.sum()
            fa, slag, sf, mk = binder * scm * frac
            opc = binder * (1 - scm)
            # Practical zero: doses below training MK-positive minimum are absent;
            # redistributed mass preserves total binder before ratio recomputation.
            if mk < mk_zero:
                opc = opc + mk
                mk = 0.0
            scm_mass = fa + slag + sf + mk
            scm_replacement = scm_mass / max(binder, 1e-12)
            sp = s["sp_min"] + (s["sp_max"] - s["sp_min"]) * x[8]
            water = w_b * binder
            v_paste = opc / RHO["OPC"] + (fa + slag + sf + mk) / RHO["SCM"] + water / RHO["Water"] + sp / RHO["SP"]
            v_agg = max(0.01, 0.98 - v_paste)
            mix = {
                "OPC": opc, "FA": fa, "GGBFS": slag, "SF": sf, "MK": mk,
                "Water": water, "NFA": fine * v_agg * RHO["Agg"],
                "NCA": (1 - fine) * v_agg * RHO["Agg"], "SP": sp,
            }
            records.append(feature_record(mix))
            rows.append({
                **mix, "w_b_ratio": w_b, "scm_replacement": scm_replacement,
                "binder_content": binder,
                "fine_agg_ratio": fine, "fa_frac": frac[0], "slag_frac": frac[1],
                "sf_frac": frac[2], "mk_frac": (mk / max(scm_mass, 1e-12) if scm_mass > 1e-12 else 0.0),
                "V_paste": v_paste, "V_agg": v_agg,
            })
        # Candidate mixtures are never cast, so strength is predicted by the
        # source-validated model rather than assumed. It enters as objective 2, as
        # the feasibility constraint, and as the CS predictor of the transport model.
        frame = pd.DataFrame(records)
        cs = self.strength.predict(frame)
        frame["CS"] = cs
        for row, value in zip(rows, cs):
            row["strength"] = float(value)
        return rows, frame

    def adjust_mu(self, mu: np.ndarray, design: np.ndarray) -> np.ndarray:
        """Hook for degrading the conditional mean. Identity in production.

        Present so that a sensitivity analysis can weaken the predictive law
        inside the search rather than only in a post-hoc audit.
        """
        return mu

    def evaluate_details(self, X: np.ndarray, lam: float | None = None) -> dict[str, Any]:
        mixes, records = self.decode(X)
        aligned = align_X(records, self.all_names)
        mu, sigma = predict_mu_sigma(self.ckpt, aligned)
        mu = self.adjust_mu(mu, aligned)
        use_lam = self.lam if lam is None else float(lam)
        log_d = mu[:, None] + use_lam * sigma[:, None] * self.z[None, :]
        d = 10**log_d * 1e-12
        m = np.array([
            m_of_mix(r["OPC"], r["FA"], r["GGBFS"], r["SF"], r["MK"], self.m_mk) for r in mixes
        ])
        ti = t_init_years(d, m[:, None], self.cs[None, :], self.ccrit[None, :])
        teq = t_eq(ti)
        dist = self.knn.kneighbors(
            self.scaler.transform(aligned), n_neighbors=self.ood_k, return_distance=True
        )[0][:, self.ood_k - 1]
        caps = self.support["ratio_caps"]
        F, G = [], []
        for i, mix in enumerate(mixes):
            binder = mix["binder_content"]
            durability = float(np.mean(teq[i])) if self.mode == "mean" else float(np.quantile(teq[i], 0.10))
            F.append([impact_from_mix(mix, COST), -mix["strength"], -durability, impact_from_mix(mix, GWP)])
            # Active strength floor: predict - nested MAE >= engineering target.
            G.append([
                SPEC.active_strength_floor_mpa - mix["strength"],
                0.25 - mix["V_paste"], mix["V_paste"] - 0.45,
                0.50 - mix["V_agg"], mix["V_agg"] - 0.75,
                SPEC.min_opc_kg_m3 - mix["OPC"],
                mix["scm_replacement"] - caps["SCM_Replacement_Ratio"],
                mix["FA"] / binder - caps["FA_Ratio"],
                mix["GGBFS"] / binder - caps["GGBFS_Ratio"],
                mix["SF"] / binder - caps["SF_Ratio"],
                mix["MK"] / binder - caps["MK_Ratio"],
                dist[i] - self.ood_limit,
            ])
        return {
            "F": np.asarray(F), "G": np.asarray(G), "mixes": mixes, "records": records,
            "mu": mu, "sigma": sigma, "teq": teq, "tinit": ti, "ood_distance": dist, "m": m,
        }

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        details = self.evaluate_details(X)
        out["F"], out["G"] = details["F"], details["G"]


def nested_crn(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8_000_003 + seed)
    return rng.standard_normal(n), sample_cs(n, rng), sample_ccrit(n, rng)


def saa_convergence(
    probe_problem: ReconstructionProblem,
    X: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, int]:
    """Nested-CRN q10 convergence against N=250; choose smallest stable N."""
    base = probe_problem.evaluate_details(X)
    rows = []
    max_n = max(SPEC.saa_grid)
    for seed in seeds:
        z, cs, cc = nested_crn(seed, max_n)
        log_d = base["mu"][:, None] + probe_problem.lam * base["sigma"][:, None] * z
        ti = t_eq(t_init_years(10**log_d * 1e-12, base["m"][:, None], cs, cc))
        ref = np.quantile(ti, 0.10, axis=1)
        ref_rank = stats.rankdata(-ref)
        ref_top = set(np.argsort(-ref)[:TOP_K])
        scale = max(float(np.ptp(ref)), 1e-9)
        for n in SPEC.saa_grid:
            q = np.quantile(ti[:, :n], 0.10, axis=1)
            top_overlap = len(ref_top & set(np.argsort(-q)[:TOP_K])) / TOP_K
            rows.append({
                "seed": seed, "saa_n": n,
                "kendall_vs_250": stats.kendalltau(ref_rank, stats.rankdata(-q)).correlation,
                "nmae_vs_250": float(np.mean(np.abs(q - ref)) / scale),
                "top10_overlap_vs_250": top_overlap,
            })
    frame = pd.DataFrame(rows)
    summary = frame.groupby("saa_n", as_index=False).agg(
        median_kendall=("kendall_vs_250", "median"),
        p10_kendall=("kendall_vs_250", lambda x: x.quantile(0.10)),
        median_nmae=("nmae_vs_250", "median"),
        median_top10_overlap=("top10_overlap_vs_250", "median"),
        p10_top10_overlap=("top10_overlap_vs_250", lambda x: x.quantile(0.10)),
    )
    stable = summary[
        (summary["median_kendall"] >= 0.90)
        & (summary["p10_kendall"] >= 0.80)
        & (summary["median_nmae"] <= 0.05)
        & (summary["median_top10_overlap"] >= 0.80)
        & (summary["p10_top10_overlap"] >= 0.60)
    ]
    auto_selected = int(stable["saa_n"].min()) if len(stable) else max_n
    # Production SAA size is locked in SPEC.
    selected = int(SPEC.saa_production_n)
    frame = frame.merge(summary, on="saa_n", how="left")
    frame["auto_selected_n"] = auto_selected
    frame["selected_production_n"] = selected
    return frame, selected


def extract_feasible_fronts(res: Any) -> list[np.ndarray]:
    """Collect feasible nondominated fronts from a pymoo history object.

    Generations whose optimum set contains no feasible member are dropped, so
    the returned list is shorter than the requested budget whenever the earliest
    generations are wholly infeasible.  Elitism keeps feasible solutions once
    they appear, so the dropped entries form a prefix and the last element is
    always the final generation.  Downstream indices therefore count
    feasible-front generations, not raw pymoo generations.
    """
    fronts: list[np.ndarray] = []
    for hist in getattr(res, "history", []) or []:
        F = hist.opt.get("F")
        cv = hist.opt.get("CV")
        if F is None or not len(F):
            continue
        F = np.asarray(F, dtype=float)
        if cv is not None:
            F = F[np.asarray(cv).ravel() <= 1e-12]
        if len(F):
            fronts.append(F)
    return fronts


def optimize(
    problem: ReconstructionProblem,
    pop_size: int,
    generations: int,
) -> tuple[Any, list[np.ndarray]]:
    """Run NSGA-II and return the result plus feasible front history.

    Hypervolume and IGD are computed later so that both per-run and shared
    reference frames can be applied to the same archived fronts.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    algorithm = NSGA2(
        pop_size=pop_size, sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True,
    )
    res = minimize(
        problem, algorithm, ("n_gen", generations),
        seed=problem.opt_seed, save_history=True, verbose=False,
    )
    return res, extract_feasible_fronts(res)


def _normalize_fronts(
    fronts: list[np.ndarray], lo: np.ndarray, hi: np.ndarray,
) -> list[np.ndarray]:
    span = np.maximum(hi - lo, 1e-12)
    return [(F - lo) / span for F in fronts]


def compute_run_indicators(
    fronts: list[np.ndarray],
    pool: str,
    seed: int,
    shared_lo: np.ndarray | None = None,
    shared_hi: np.ndarray | None = None,
    pooled_reference: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-run HV plus optional within-arm shared-reference HV and IGD.

    Per-run normalization uses the union of fronts within that run. Within-arm
    shared normalization uses one lo/hi across the seeds of that arm only, never
    across mean and q10 arms, because those arms optimize different durability
    statistics in objective 3. The attainable hypervolume under a reference
    point of 1.1 in each of four normalized objectives is 1.1**4 = 1.4641.
    """
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD

    if not fronts:
        return pd.DataFrame()
    all_f = np.vstack(fronts)
    lo = np.nanmin(all_f, axis=0)
    hi = np.nanmax(all_f, axis=0)
    per_run = _normalize_fronts(fronts, lo, hi)
    ref = np.full(4, SPEC.hv_reference_point)
    hv = HV(ref_point=ref)
    shared_hv = None
    igd = None
    if shared_lo is not None and shared_hi is not None and pooled_reference is not None:
        shared = _normalize_fronts(fronts, shared_lo, shared_hi)
        shared_hv = HV(ref_point=ref)
        igd = IGD(pooled_reference)
    rows = []
    for generation, Fn in enumerate(per_run, start=1):
        row = {
            "pool": pool,
            "optimization_seed": seed,
            "generation": generation,
            "hypervolume": float(hv(Fn)),
            "hypervolume_per_run": float(hv(Fn)),
            "n_nondominated": int(len(Fn)),
            "hv_reference_point": SPEC.hv_reference_point,
            "hv_theoretical_max": float(SPEC.hv_reference_point ** 4),
        }
        if shared_hv is not None and igd is not None:
            Fs = shared[generation - 1]
            row["hypervolume_shared"] = float(shared_hv(Fs))
            row["igd_to_pooled_reference"] = float(igd(Fs))
        rows.append(row)
    return pd.DataFrame(rows)


def _arm_reference(
    items: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-arm shared lo/hi and nondominated union of final fronts."""
    stacked = np.vstack([F for item in items for F in item["fronts"]])
    shared_lo = np.nanmin(stacked, axis=0)
    shared_hi = np.nanmax(stacked, axis=0)
    finals = [
        _normalize_fronts([item["fronts"][-1]], shared_lo, shared_hi)[0]
        for item in items
    ]
    pooled = np.vstack(finals)
    keep = np.ones(len(pooled), dtype=bool)
    for i in range(len(pooled)):
        dominated_by_other = np.all(pooled <= pooled[i] + 1e-12, axis=1) & np.any(
            pooled < pooled[i] - 1e-12, axis=1
        )
        if np.any(dominated_by_other):
            keep[i] = False
    return shared_lo, shared_hi, pooled[keep] if np.any(keep) else pooled


def annotate_shared_indicators(
    front_store: list[dict[str, Any]],
) -> pd.DataFrame:
    """Recompute indicators under within-arm shared objective normalization.

    Mean-search and q10-search optimize different fourth-dimensional durability
    statistics, so their fronts are never pooled into one reference. Each arm
    gets its own shared lo/hi and its own nondominated union of final fronts.
    """
    if not front_store:
        return pd.DataFrame()
    frames = []
    for pool in sorted({item["pool"] for item in front_store}):
        items = [item for item in front_store if item["pool"] == pool]
        shared_lo, shared_hi, pooled_reference = _arm_reference(items)
        for item in items:
            frames.append(
                compute_run_indicators(
                    item["fronts"], item["pool"], item["optimization_seed"],
                    shared_lo=shared_lo, shared_hi=shared_hi,
                    pooled_reference=pooled_reference,
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plateau_relative_increment(
    indicators: pd.DataFrame,
    pool: str,
    metric: str = "hypervolume_per_run",
    trailing: int = SPEC.plateau_trailing_generations,
) -> dict[str, Any]:
    """Median relative HV gain over the trailing generations for one arm.

    Per-run indicator series do not all terminate at the same generation, so
    both endpoints are taken inside each run's own trailing window before the
    medians are formed.  Forming a median at each generation instead would
    compare endpoints resting on different seed panels.
    """
    sub = indicators[indicators["pool"] == pool]
    if sub.empty:
        return {"pool": pool, "metric": metric, "available": False}
    starts: list[float] = []
    ends: list[float] = []
    terminal: list[int] = []
    for _, grp in sub.groupby("optimization_seed"):
        series = grp.set_index("generation")[metric].sort_index()
        if len(series) < trailing + 1:
            continue
        end_gen = int(series.index[-1])
        start_gen = end_gen - trailing
        starts.append(
            float(series.loc[start_gen]) if start_gen in series.index
            else float(series.iloc[-(trailing + 1)])
        )
        ends.append(float(series.iloc[-1]))
        terminal.append(end_gen)
    if not ends:
        return {
            "pool": pool, "metric": metric, "available": False,
            "n_generations": int(sub["generation"].nunique()),
        }
    start = float(np.median(starts))
    end = float(np.median(ends))
    rel = (end - start) / max(abs(end), 1e-12)
    return {
        "pool": pool,
        "metric": metric,
        "available": True,
        "endpoint_rule": "per-run trailing window; both endpoints on a common seed panel",
        "index_definition": (
            "generation index counts feasible-front generations; runs whose "
            "earliest generations were wholly infeasible carry a lower terminal "
            "index although every run completed the full budget"
        ),
        "n_seeds": len(ends),
        "n_generations": int(sub["generation"].nunique()),
        "terminal_index_min": int(min(terminal)),
        "terminal_index_max": int(max(terminal)),
        "trailing_generations": trailing,
        "median_hv_start": start,
        "median_hv_end": end,
        "relative_increment": float(rel),
        "plateaued": bool(rel < SPEC.plateau_relative_increment),
        "plateau_threshold": SPEC.plateau_relative_increment,
    }


def plateau_payload_for(indicators: pd.DataFrame, generations: int) -> dict[str, Any]:
    """Full plateau diagnostic block for both arms."""
    return {
        "pre_specified_rule": (
            f"median q10 {SPEC.plateau_trailing_generations}-generation relative "
            f"HV increment < {SPEC.plateau_relative_increment}"
        ),
        "generations_requested": generations,
        "mean": plateau_relative_increment(indicators, "mean"),
        "q10": plateau_relative_increment(indicators, "q10"),
        "hv_definition": {
            "per_run": "min-max normalize each run over its own generation fronts; ref=1.1^4",
            "within_arm_shared": (
                "one lo/hi across seeds of the same arm only; mean and q10 arms "
                "are never pooled because objective 3 differs"
            ),
            "igd": "IGD to nondominated union of that arm's final fronts in within-arm space",
            "theoretical_max": float(SPEC.hv_reference_point ** 4),
        },
    }


def pool_rows(problem: ReconstructionProblem, res: Any, mode: str, seed: int) -> pd.DataFrame:
    if res.X is None or res.F is None:
        return pd.DataFrame()
    details = problem.evaluate_details(res.X)
    rows = []
    for i, mix in enumerate(details["mixes"]):
        te = details["teq"][i]
        rows.append({
            "pool": mode, "optimization_seed": seed, "local_id": i,
            **mix, "m": details["m"][i], "ngb_mu_log10": details["mu"][i],
            "ngb_sigma_log10": details["sigma"][i], "ood_knn_distance": details["ood_distance"][i],
            "t_eq_mean": np.mean(te), "t_eq_q10": np.quantile(te, 0.10),
            "gwp": details["F"][i, 3], "cost": details["F"][i, 0],
            "constraint_violation_max": np.maximum(details["G"][i], 0).max(),
        })
    return pd.DataFrame(rows)


def permutation_null(
    a: np.ndarray, b: np.ndarray, top_k: int, draws: int, seed: int,
) -> dict[str, float]:
    """Analytical random-permutation nulls for tau and top-k overlap.

    scipy's tie-corrected Kendall test and the exact hypergeometric overlap null
    avoid Monte Carlo error; ``draws`` is retained as the registered equivalent
    simulation budget for provenance.
    """
    del seed
    kt = stats.kendalltau(a, b)
    observed_tau = float(kt.correlation)
    ids_a = set(np.argsort(a)[:top_k])
    observed_overlap = len(ids_a & set(np.argsort(b)[:top_k])) / top_k
    n = len(a)
    overlap_count = int(round(observed_overlap * top_k))
    tau_null_sd = math.sqrt(2.0 * (2 * n + 5) / (9.0 * n * max(n - 1, 1)))
    return {
        "observed_tau": observed_tau,
        "tau_null_mean": 0.0,
        "tau_null_sd_no_ties_reference": tau_null_sd,
        "tau_two_sided_p": float(kt.pvalue),
        "observed_topk_overlap": observed_overlap,
        "overlap_null_mean": top_k / n,
        "overlap_upper_p": float(stats.hypergeom.sf(overlap_count - 1, n, top_k, top_k)),
        "null_method": "tie-corrected asymptotic Kendall permutation null; exact hypergeometric top-k null",
        "registered_mc_equivalent_draws": draws,
    }


def s3_draw_metrics(
    scores: np.ndarray,
    top_k: int,
    baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-candidate top-k failure counts and regret sums over Monte Carlo draws.

    ``scores`` has shape (n_candidates, n_draws) and is minimized.
    ``baseline``, when supplied, is a length-``n_draws`` vector of the
    draw-specific optimum against which regret is measured. When omitted, the
    arm-local minimum is used; that value is also always returned as
    ``local_regret`` so within-shortlist dispersion remains auditable.
    """
    n_cand, n_draws = scores.shape
    k = min(top_k, n_cand)
    top_fail = np.zeros(n_cand)
    regret = np.zeros(n_cand)
    local_regret = np.zeros(n_cand)
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float).reshape(-1)
        if baseline.shape[0] != n_draws:
            raise ValueError(
                f"baseline length {baseline.shape[0]} != n_draws {n_draws}"
            )
    for draw in range(n_draws):
        order = np.argsort(scores[:, draw])
        local_best = scores[order[0], draw]
        best = local_best if baseline is None else float(baseline[draw])
        top = set(order[:k].tolist())
        top_fail += np.array([i not in top for i in range(n_cand)], dtype=float)
        regret += scores[:, draw] - best
        local_regret += scores[:, draw] - local_best
    mean_score = scores.mean(axis=1)
    return top_fail, regret, local_regret, mean_score


def s3_prefix_stability(
    scores: np.ndarray,
    top_k: int,
    prefixes: tuple[int, ...] = SPEC.mc_convergence_draws,
    baseline: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Recompute front-level S3 summaries on nested CRN prefixes."""
    rows = []
    for n in prefixes:
        if n > scores.shape[1]:
            continue
        base_n = None if baseline is None else baseline[:n]
        fail, regret, local_regret, mean_score = s3_draw_metrics(
            scores[:, :n], top_k, baseline=base_n,
        )
        rank = np.argsort(mean_score)
        k = min(top_k, len(mean_score))
        idx = rank[:k]
        rows.append({
            "mc_draws": int(n),
            "mean_topk_failure": float(np.mean((fail / n)[idx])),
            "mean_expected_regret": float(np.mean((regret / n)[idx])),
            "median_expected_regret": float(np.median((regret / n)[idx])),
            "mean_within_shortlist_dispersion": float(
                np.mean((local_regret / n)[idx])
            ),
        })
    return rows


def union_draw_baseline(*score_blocks: np.ndarray) -> np.ndarray:
    """Per-draw minimum across one or more candidate blocks under a shared CRN."""
    if not score_blocks:
        raise ValueError("at least one score block is required")
    n_draws = score_blocks[0].shape[1]
    for block in score_blocks:
        if block.shape[1] != n_draws:
            raise ValueError("all score blocks must share the same draw count")
    stacked = np.vstack(score_blocks)
    return stacked.min(axis=0)


def audit_pool(
    pool: pd.DataFrame,
    problem: ReconstructionProblem,
    lam_values: tuple[float, ...],
    mc_n: int,
    perm_n: int,
    seed: int,
    shared_baselines: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Identical post-hoc S1/S2/S3, freeze, lambda, and correlation audits.

    When ``shared_baselines`` supplies per-draw minima for keys ``\"b\"`` and
    ``\"j\"`` (taken over the matched mean+q10 union under the same CRN),
    headline expected regret is measured against that common yardstick.
    Arm-local regret is always retained as within-shortlist dispersion.
    """
    # Reconstruct normalized genotype exactly enough for evaluation from exported decisions.
    s = problem.support
    X = np.column_stack([
        (pool["w_b_ratio"] - s["w_b_min"]) / (s["w_b_max"] - s["w_b_min"]),
        pool["scm_replacement"] / SPEC.max_scm_total_ratio,
        (pool["binder_content"] - s["binder_min"]) / (s["binder_max"] - s["binder_min"]),
        (pool["fine_agg_ratio"] - 0.30) / 0.40,
        pool["fa_frac"], pool["slag_frac"], pool["sf_frac"], pool["mk_frac"],
        (pool["SP"] - s["sp_min"]) / (s["sp_max"] - s["sp_min"]),
    ])
    base = problem.evaluate_details(X)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(mc_n)
    cs, cc = sample_cs(mc_n, rng), sample_ccrit(mc_n, rng)
    audit_rows, lambda_rows = [], []
    gwp = np.asarray([impact_from_mix(m, GWP) for m in base["mixes"]])
    ap = np.asarray([impact_from_mix(m, AP) for m in base["mixes"]])
    ep = np.asarray([impact_from_mix(m, EP) for m in base["mixes"]])
    scenarios: dict[float, dict[str, np.ndarray]] = {}
    for lam in lam_values:
        logd = base["mu"][:, None] + lam * base["sigma"][:, None] * z
        d = 10**logd * 1e-12
        ti = t_init_years(d, base["m"][:, None], cs, cc)
        te = t_eq(ti)
        te_freeze = t_eq(t_init_years(d, base["m"][:, None], cs, cc, SPEC.freeze_sensitivity_years))
        j_draw = gwp[:, None] * (d / 1e-12)
        # Per-draw Category B uses the same initiation solves already required for te.
        # E[GWP/t_eq] differs from GWP/E[t_eq]; S3 ranks on the per-draw quantity.
        b_draw = gwp[:, None] / np.maximum(te, 1e-12)
        scenarios[lam] = {"teq": te, "j": j_draw, "b": b_draw, "freeze": te_freeze}
        for i in range(len(pool)):
            lambda_rows.append({
                "pool": pool["pool"].iloc[0],
                "optimization_seed": int(pool["optimization_seed"].iloc[0]),
                "candidate_id": int(pool["local_id"].iloc[i]), "lambda": lam,
                "t_eq_mean": te[i].mean(), "t_eq_q10": np.quantile(te[i], 0.10),
                "category_b": gwp[i] / max(te[i].mean(), 1e-12),
                "category_b_mean_per_draw": float(b_draw[i].mean()),
                "J_mean": j_draw[i].mean(),
            })
    prod_lam = min(lam_values, key=lambda value: abs(value - problem.lam))
    sc = scenarios[prod_lam]
    mean_te, q10_te = sc["teq"].mean(axis=1), np.quantile(sc["teq"], 0.10, axis=1)
    freeze_te = sc["freeze"].mean(axis=1)
    jmean = sc["j"].mean(axis=1)
    bmean = sc["b"].mean(axis=1)
    cat_b = gwp / np.maximum(mean_te, 1e-12)
    inv_te = 1.0 / np.maximum(mean_te, 1e-12)

    base_b = None if shared_baselines is None else shared_baselines.get("b")
    base_j = None if shared_baselines is None else shared_baselines.get("j")
    top_fail_j, regret_j, local_regret_j, _ = s3_draw_metrics(
        sc["j"], TOP_K, baseline=base_j,
    )
    top_fail_b, regret_b, local_regret_b, _ = s3_draw_metrics(
        sc["b"], TOP_K, baseline=base_b,
    )
    rank_j = np.argsort(jmean)
    rank_b = np.argsort(bmean)
    k = min(TOP_K, len(pool))
    regret_baseline = (
        "matched_seed_union" if shared_baselines is not None else "arm_local"
    )

    for i in range(len(pool)):
        audit_rows.append({
            "pool": pool["pool"].iloc[0],
            "optimization_seed": int(pool["optimization_seed"].iloc[0]),
            "candidate_id": int(pool["local_id"].iloc[i]), "gwp": gwp[i], "ap": ap[i], "ep": ep[i],
            "t_eq_mean": mean_te[i], "t_eq_q10": q10_te[i],
            "category_b_gwp_per_t_eq": cat_b[i],
            "category_b_mean_per_draw": bmean[i],
            "inverse_t_eq": inv_te[i],
            "J_mean_dimensionless_D_multiplier": jmean[i],
            "t_eq_mean_freeze25": freeze_te[i],
            "topk_failure_probability_category_b": top_fail_b[i] / mc_n,
            "expected_category_b_regret": regret_b[i] / mc_n,
            "within_shortlist_category_b_dispersion": local_regret_b[i] / mc_n,
            "topk_failure_probability": top_fail_j[i] / mc_n,
            "expected_J_regret": regret_j[i] / mc_n,
            "within_shortlist_J_dispersion": local_regret_j[i] / mc_n,
            "regret_baseline": regret_baseline,
        })

    def safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            return None
        try:
            out = fn(a, b)
            return float(getattr(out, "correlation", getattr(out, "statistic", out)))
        except Exception:
            return None

    def overlap(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 1:
            return float("nan")
        return len(set(np.argsort(a)[:k]) & set(np.argsort(b)[:k])) / k

    summary = {
        "pool": pool["pool"].iloc[0], "optimization_seed": int(pool["optimization_seed"].iloc[0]),
        "n": len(pool), "production_lambda_used": prod_lam,
        "S1": {
            "kendall_gwp_vs_category_b": safe_corr(stats.kendalltau, gwp, cat_b),
            "topk_overlap_gwp_vs_category_b": overlap(gwp, cat_b),
        },
        "S2": {
            "kendall_J_vs_category_b": safe_corr(stats.kendalltau, jmean, cat_b),
            "topk_overlap_J_vs_category_b": overlap(jmean, cat_b),
            "J_definition": "GWP * [Dapp/(1e-12 m2/s)]",
        },
        "S3": {
            "primary_quantity": "Category B = GWP / t_eq per Monte Carlo draw",
            "secondary_quantity": "J = GWP * [Dapp/(1e-12 m2/s)] per Monte Carlo draw",
            "regret_baseline": regret_baseline,
            "note": (
                "S3 Category B uses E[GWP/t_eq] over draws; S1/S2 point Category B uses "
                "GWP/E[t_eq]. These are not interchangeable. Headline expected regret is "
                "measured against the matched-seed union baseline when supplied; "
                "arm-local regret is retained as within-shortlist dispersion."
            ),
            "mean_topk_failure_among_meanB_topk": float(np.mean((top_fail_b / mc_n)[rank_b[:k]])),
            "mean_expected_category_b_regret_among_meanB_topk": float(
                np.mean((regret_b / mc_n)[rank_b[:k]])
            ),
            "median_expected_category_b_regret_among_meanB_topk": float(
                np.median((regret_b / mc_n)[rank_b[:k]])
            ),
            "mean_within_shortlist_category_b_dispersion_among_meanB_topk": float(
                np.mean((local_regret_b / mc_n)[rank_b[:k]])
            ),
            "mean_topk_failure_among_meanJ_topk": float(np.mean((top_fail_j / mc_n)[rank_j[:k]])),
            "mean_expected_J_regret_among_meanJ_topk": float(np.mean((regret_j / mc_n)[rank_j[:k]])),
            "median_expected_J_regret_among_meanJ_topk": float(np.median((regret_j / mc_n)[rank_j[:k]])),
            "mean_within_shortlist_J_dispersion_among_meanJ_topk": float(
                np.mean((local_regret_j / mc_n)[rank_j[:k]])
            ),
            "mc_draws": mc_n, "common_random_numbers": True,
            "mc_convergence_category_b": s3_prefix_stability(
                sc["b"], TOP_K, baseline=base_b,
            ),
            "mc_convergence_J": s3_prefix_stability(
                sc["j"], TOP_K, baseline=base_j,
            ),
        },
        "mean_vs_q10": {
            "kendall": safe_corr(stats.kendalltau, -mean_te, -q10_te),
            "topk_overlap": overlap(-mean_te, -q10_te),
        },
        "freeze25_sensitivity": {
            "kendall_primary_vs_freeze": safe_corr(stats.kendalltau, -mean_te, -freeze_te),
            "topk_overlap": overlap(-mean_te, -freeze_te),
        },
        "category_b_vs_inverse_t_eq": {
            "pearson": safe_corr(stats.pearsonr, cat_b, inv_te),
            "spearman": safe_corr(stats.spearmanr, cat_b, inv_te),
            "kendall": safe_corr(stats.kendalltau, cat_b, inv_te),
        },
        "permutation_null_S1": (
            permutation_null(gwp, cat_b, k, perm_n, seed + 11) if len(pool) >= 2
            else {"observed_tau": None, "tau_two_sided_p": None, "observed_topk_overlap": None,
                  "overlap_upper_p": None}
        ),
        "permutation_null_S2": (
            permutation_null(jmean, cat_b, k, perm_n, seed + 29) if len(pool) >= 2
            else {"observed_tau": None, "tau_two_sided_p": None, "observed_topk_overlap": None,
                  "overlap_upper_p": None}
        ),
        # Temporary arrays for matched-seed union baselines; stripped before export.
        "_draw_scores": {"b": sc["b"], "j": sc["j"]},
    }
    return pd.DataFrame(audit_rows), pd.DataFrame(lambda_rows), summary


def patch_shared_regret(
    audit: pd.DataFrame,
    summary: dict[str, Any],
    shared_baselines: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace arm-local headline regret with matched-seed union regret."""
    scores = summary.pop("_draw_scores")
    mc_n = int(summary["S3"]["mc_draws"])
    k = min(TOP_K, int(summary["n"]))
    bmean = audit["category_b_mean_per_draw"].to_numpy(dtype=float)
    jmean = audit["J_mean_dimensionless_D_multiplier"].to_numpy(dtype=float)
    rank_b = np.argsort(bmean)
    rank_j = np.argsort(jmean)

    top_fail_b, regret_b, local_b, _ = s3_draw_metrics(
        scores["b"], TOP_K, baseline=shared_baselines["b"],
    )
    top_fail_j, regret_j, local_j, _ = s3_draw_metrics(
        scores["j"], TOP_K, baseline=shared_baselines["j"],
    )
    audit = audit.copy()
    audit["topk_failure_probability_category_b"] = top_fail_b / mc_n
    audit["expected_category_b_regret"] = regret_b / mc_n
    audit["within_shortlist_category_b_dispersion"] = local_b / mc_n
    audit["topk_failure_probability"] = top_fail_j / mc_n
    audit["expected_J_regret"] = regret_j / mc_n
    audit["within_shortlist_J_dispersion"] = local_j / mc_n
    audit["regret_baseline"] = "matched_seed_union"

    s3 = summary["S3"]
    s3["regret_baseline"] = "matched_seed_union"
    s3["mean_topk_failure_among_meanB_topk"] = float(
        np.mean((top_fail_b / mc_n)[rank_b[:k]])
    )
    s3["mean_expected_category_b_regret_among_meanB_topk"] = float(
        np.mean((regret_b / mc_n)[rank_b[:k]])
    )
    s3["median_expected_category_b_regret_among_meanB_topk"] = float(
        np.median((regret_b / mc_n)[rank_b[:k]])
    )
    s3["mean_within_shortlist_category_b_dispersion_among_meanB_topk"] = float(
        np.mean((local_b / mc_n)[rank_b[:k]])
    )
    s3["mean_topk_failure_among_meanJ_topk"] = float(
        np.mean((top_fail_j / mc_n)[rank_j[:k]])
    )
    s3["mean_expected_J_regret_among_meanJ_topk"] = float(
        np.mean((regret_j / mc_n)[rank_j[:k]])
    )
    s3["median_expected_J_regret_among_meanJ_topk"] = float(
        np.median((regret_j / mc_n)[rank_j[:k]])
    )
    s3["mean_within_shortlist_J_dispersion_among_meanJ_topk"] = float(
        np.mean((local_j / mc_n)[rank_j[:k]])
    )
    s3["mc_convergence_category_b"] = s3_prefix_stability(
        scores["b"], TOP_K, baseline=shared_baselines["b"],
    )
    s3["mc_convergence_J"] = s3_prefix_stability(
        scores["j"], TOP_K, baseline=shared_baselines["j"],
    )
    s3["note"] = (
        "S3 Category B uses E[GWP/t_eq] over draws; S1/S2 point Category B uses "
        "GWP/E[t_eq]. These are not interchangeable. Headline expected regret is "
        "measured against the matched-seed union of both arms under a shared CRN; "
        "arm-local regret is retained as within-shortlist dispersion."
    )
    return audit, summary


def audit_matched_seed(
    pools_by_mode: dict[str, pd.DataFrame],
    problems_by_mode: dict[str, ReconstructionProblem],
    lam_values: tuple[float, ...],
    mc_n: int,
    perm_n: int,
    audit_seed: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict[str, Any]]]:
    """Audit both arms of one matched seed against a shared regret baseline.

    Both arms are scored under the identical CRN (``audit_seed``). Per-draw
    Category B and J minima are then taken over the candidate union, and each
    arm's headline expected regret is measured against that common yardstick.
    Arm-local regret is retained as within-shortlist dispersion.
    """
    provisional: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for mode, pool in pools_by_mode.items():
        provisional[mode] = audit_pool(
            pool, problems_by_mode[mode], lam_values, mc_n, perm_n, audit_seed,
        )
    shared = {
        "b": union_draw_baseline(
            *[provisional[m][2]["_draw_scores"]["b"] for m in pools_by_mode]
        ),
        "j": union_draw_baseline(
            *[provisional[m][2]["_draw_scores"]["j"] for m in pools_by_mode]
        ),
    }
    audits, lambdas, summaries = [], [], []
    for mode in pools_by_mode:
        audit, lambda_sens, summary = provisional[mode]
        audit, summary = patch_shared_regret(audit, summary, shared)
        audits.append(audit)
        lambdas.append(lambda_sens)
        summaries.append(summary)
    return audits, lambdas, summaries


def flatten_seed_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """One inferential row per optimization seed and pool."""
    return {
        "optimization_seed": summary["optimization_seed"],
        "pool": summary["pool"],
        "n_candidates": summary["n"],
        "production_lambda": summary["production_lambda_used"],
        "s1_kendall_gwp_vs_category_b": summary["S1"]["kendall_gwp_vs_category_b"],
        "s1_top10_overlap": summary["S1"]["topk_overlap_gwp_vs_category_b"],
        "s2_kendall_J_vs_category_b": summary["S2"]["kendall_J_vs_category_b"],
        "s2_top10_overlap": summary["S2"]["topk_overlap_J_vs_category_b"],
        "s3_top10_failure_category_b": summary["S3"]["mean_topk_failure_among_meanB_topk"],
        "s3_expected_category_b_regret": summary["S3"][
            "mean_expected_category_b_regret_among_meanB_topk"
        ],
        "s3_within_shortlist_category_b_dispersion": summary["S3"].get(
            "mean_within_shortlist_category_b_dispersion_among_meanB_topk"
        ),
        "s3_top10_failure": summary["S3"]["mean_topk_failure_among_meanJ_topk"],
        "s3_expected_J_regret": summary["S3"]["mean_expected_J_regret_among_meanJ_topk"],
        "s3_within_shortlist_J_dispersion": summary["S3"].get(
            "mean_within_shortlist_J_dispersion_among_meanJ_topk"
        ),
        "s3_regret_baseline": summary["S3"].get("regret_baseline", "arm_local"),
        "mean_vs_q10_kendall": summary["mean_vs_q10"]["kendall"],
        "mean_vs_q10_top10_overlap": summary["mean_vs_q10"]["topk_overlap"],
        "freeze25_kendall": summary["freeze25_sensitivity"]["kendall_primary_vs_freeze"],
        "freeze25_top10_overlap": summary["freeze25_sensitivity"]["topk_overlap"],
        "category_b_vs_inverse_t_eq_kendall": summary["category_b_vs_inverse_t_eq"]["kendall"],
        "s1_tau_null_p": summary["permutation_null_S1"]["tau_two_sided_p"],
        "s1_overlap_null_p": summary["permutation_null_S1"]["overlap_upper_p"],
        "s2_tau_null_p": summary["permutation_null_S2"]["tau_two_sided_p"],
        "s2_overlap_null_p": summary["permutation_null_S2"]["overlap_upper_p"],
        "null_n": summary["n"],
        "mc_convergence_category_b": json.dumps(summary["S3"]["mc_convergence_category_b"]),
        "mc_convergence_J": json.dumps(summary["S3"]["mc_convergence_J"]),
    }


def paired_metric_inference(
    seedwise: pd.DataFrame, metric: str, bootstrap_draws: int = 20_000,
) -> dict[str, Any]:
    """Paired q10-minus-mean inference across optimization seeds."""
    wide = seedwise.pivot(index="optimization_seed", columns="pool", values=metric).dropna()
    delta = (wide["q10"] - wide["mean"]).to_numpy(dtype=float)
    rng = np.random.default_rng(72_031 + sum(map(ord, metric)))
    boot = rng.choice(delta, size=(bootstrap_draws, len(delta)), replace=True).mean(axis=1)
    if np.allclose(delta, 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(delta, alternative="two-sided", zero_method="pratt").pvalue)
    mean_base = float(wide["mean"].mean())
    # Signed relative change: positive means q10 is larger than mean.
    relative_change = float(delta.mean() / max(abs(mean_base), 1e-12))
    relative_reduction = float(-relative_change)
    ci = np.quantile(boot, [0.025, 0.975])
    material = bool(delta.mean() < 0 and ci[1] < 0 and relative_reduction >= 0.05)
    return {
        "metric": metric,
        "direction": "negative q10-minus-mean favors q10",
        "n_paired_seeds": int(len(delta)),
        "mean_pool_mean": mean_base,
        "q10_pool_mean": float(wide["q10"].mean()),
        "q10_minus_mean_by_seed": {
            str(int(seed)): float(wide.loc[seed, "q10"] - wide.loc[seed, "mean"]) for seed in wide.index
        },
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_sd": float(np.std(delta, ddof=1)),
        "delta_iqr": [float(np.quantile(delta, 0.25)), float(np.quantile(delta, 0.75))],
        "paired_bootstrap_mean_delta_95ci": [float(ci[0]), float(ci[1])],
        "paired_wilcoxon_two_sided_p": wilcoxon_p,
        "wilcoxon_floor_note": (
            f"With n={len(delta)} and all differences sharing a sign, the two-sided "
            f"Wilcoxon signed-rank p equals 2/2^{len(delta)} = {2 / 2 ** len(delta):.6g}, "
            "its minimum attainable value under the exact null."
        ),
        "relative_mean_change": relative_change,
        "relative_mean_reduction": relative_reduction,
        "material_reduction_rule": "mean delta<0, paired-bootstrap upper 95% CI<0, and >=5% relative reduction",
        "materially_reduces": material,
    }


def holm_adjust(p_by_name: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment across a family of tests."""
    items = sorted(p_by_name.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items, start=1):
        value = min(1.0, (m - rank + 1) * float(p))
        running = max(running, value)
        adjusted[name] = float(running)
    return adjusted


def aggregate_seedwise(seedwise: pd.DataFrame) -> dict[str, Any]:
    skip = {
        "optimization_seed", "pool", "n_candidates", "production_lambda", "null_n",
        "mc_convergence_category_b", "mc_convergence_J", "s3_regret_baseline", "cs_arm",
    }
    metrics = [c for c in seedwise.columns if c not in skip]
    descriptive: dict[str, Any] = {}
    for pool, sub in seedwise.groupby("pool"):
        descriptive[pool] = {}
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            descriptive[pool][metric] = {
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)),
                "iqr": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
            }
    failure_b = paired_metric_inference(seedwise, "s3_top10_failure_category_b")
    regret_b = paired_metric_inference(seedwise, "s3_expected_category_b_regret")
    failure_j = paired_metric_inference(seedwise, "s3_top10_failure")
    regret_j = paired_metric_inference(seedwise, "s3_expected_J_regret")
    holm = holm_adjust({
        "s3_top10_failure_category_b": failure_b["paired_wilcoxon_two_sided_p"],
        "s3_expected_category_b_regret": regret_b["paired_wilcoxon_two_sided_p"],
        "s3_top10_failure_J": failure_j["paired_wilcoxon_two_sided_p"],
        "s3_expected_J_regret": regret_j["paired_wilcoxon_two_sided_p"],
    })
    for block, key in (
        (failure_b, "s3_top10_failure_category_b"),
        (regret_b, "s3_expected_category_b_regret"),
        (failure_j, "s3_top10_failure_J"),
        (regret_j, "s3_expected_J_regret"),
    ):
        block["holm_adjusted_p"] = holm[key]
        block["family"] = list(holm.keys())

    def wording_for(failure: dict[str, Any], regret: dict[str, Any], label: str) -> str:
        if failure["materially_reduces"] and regret["materially_reduces"]:
            return (
                f"The q10 pools materially reduce both paired S3 top-k failure and "
                f"expected {label} regret."
            )
        if failure["materially_reduces"]:
            return (
                f"The q10 pools materially reduce paired S3 top-k failure, but not "
                f"expected {label} regret."
            )
        if regret["materially_reduces"]:
            return (
                f"The q10 pools do not materially reduce paired S3 top-k failure, but "
                f"do reduce expected {label} regret."
            )
        return (
            f"The paired seed analysis does not establish a material q10 reduction "
            f"in S3 failure or {label} regret."
        )

    # Aggregate nested MC-convergence rows across seeds for SI export.
    mc_rows = []
    for _, row in seedwise.iterrows():
        for quantity, key in (("category_b", "mc_convergence_category_b"), ("J", "mc_convergence_J")):
            for item in json.loads(row[key]):
                mc_rows.append({
                    "pool": row["pool"],
                    "optimization_seed": int(row["optimization_seed"]),
                    "quantity": quantity,
                    **item,
                })
    return {
        "inferential_unit": "matched optimization seed (one n=80 front per pool per seed)",
        "n_paired_seeds": int(seedwise["optimization_seed"].nunique()),
        "within_seed_front_n": sorted(seedwise["n_candidates"].unique().tolist()),
        "seedwise_descriptive": descriptive,
        "primary_s3_quantity": "Category B",
        "secondary_s3_quantity": "J",
        "regret_baseline": (
            str(seedwise["s3_regret_baseline"].iloc[0])
            if "s3_regret_baseline" in seedwise.columns and len(seedwise)
            else "arm_local"
        ),
        "holm_family_adjustment": holm,
        "paired_headline": {
            "s3_top10_failure_category_b": failure_b,
            "s3_expected_category_b_regret": regret_b,
            "s3_top10_failure_J": failure_j,
            "s3_expected_J_regret": regret_j,
        },
        "conclusion": {
            "q10_materially_reduces_failure_category_b": failure_b["materially_reduces"],
            "q10_materially_reduces_regret_category_b": regret_b["materially_reduces"],
            "q10_materially_reduces_failure_J": failure_j["materially_reduces"],
            "q10_materially_reduces_regret_J": regret_j["materially_reduces"],
            "wording_category_b": wording_for(failure_b, regret_b, "Category B"),
            "wording_J": wording_for(failure_j, regret_j, "J"),
            "wording": wording_for(failure_b, regret_b, "Category B"),
        },
        "mc_convergence_rows": mc_rows,
    }


def _savefig(fig: Any, stem: str, dpi: int = 600) -> None:
    """Write PNG at journal resolution and a vector PDF companion for line art."""
    import matplotlib.pyplot as plt

    fig.savefig(FIGS / f"{stem}.png", dpi=dpi)
    fig.savefig(FIGS / f"{stem}.pdf")
    plt.close(fig)


def make_figures(
    saa: pd.DataFrame,
    pools: pd.DataFrame,
    audits: pd.DataFrame,
    lambdas: pd.DataFrame,
    indicators: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import apply_plot_style, style_axes

    apply_plot_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(EXPORTS / "rev2_coverage_curves.csv")
    pooled_coverage = coverage[coverage["scope"] == "pooled"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = {
        "before_nll_calibration": "Raw NGB",
        "after_global_oof_nll_calibration": "Global OOF NLL scale",
        "after_nll_calibration": "Fold-local NLL scales",
    }
    for stage, sub in pooled_coverage.groupby("stage"):
        ax.plot(
            sub["nominal_coverage"], sub["empirical_coverage"], "o-",
            markersize=3, label=labels.get(stage, stage),
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    ax.set(
        xlabel="Nominal Gaussian interval coverage",
        ylabel="Empirical source-disjoint coverage",
        xlim=(0, 1), ylim=(0, 1),
    )
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    _savefig(fig, "rev2_grouped_calibration")

    agg = saa.groupby("saa_n").agg(kendall=("kendall_vs_250", "median"), nmae=("nmae_vs_250", "median"))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(agg.index, agg["kendall"], "o-", label="median Kendall vs N=250")
    ax1.set(xlabel="SAA N", ylabel="Kendall tau", ylim=(0, 1.02))
    ax2 = ax1.twinx()
    ax2.plot(agg.index, agg["nmae"], "s--", color="#4C9BD6", label="normalized MAE")
    ax2.set_ylabel("normalized MAE")
    ax1.legend(frameon=False, loc="lower right")
    style_axes(ax1)
    style_axes(ax2)
    fig.tight_layout(); _savefig(fig, "rev2_saa_convergence")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, sub in pools.groupby("pool"):
        ax.scatter(sub["gwp"], sub["t_eq_q10"], s=12, alpha=0.45, label=name)
    ax.set(xlabel="GWP (kg CO2-eq/m3)", ylabel="q10 t_eq (years)")
    ax.legend(frameon=False); style_axes(ax); fig.tight_layout(); _savefig(fig, "rev2_matched_pools")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, sub in audits.groupby("pool"):
        ax.scatter(sub["category_b_gwp_per_t_eq"], sub["J_mean_dimensionless_D_multiplier"], s=14, alpha=0.6, label=name)
    ax.set(xlabel="Category B: GWP/t_eq", ylabel="J = GWP [D/(1e-12)]")
    ax.legend(frameon=False); style_axes(ax); fig.tight_layout(); _savefig(fig, "rev2_categoryB_vs_J")

    fig, ax = plt.subplots(figsize=(6, 4))
    sens = lambdas.groupby(["pool", "lambda"])["t_eq_q10"].median().reset_index()
    for name, sub in sens.groupby("pool"):
        ax.plot(sub["lambda"], sub["t_eq_q10"], "o-", label=name)
    ax.set(xlabel="lambda", ylabel="median candidate q10 t_eq (years)")
    ax.legend(frameon=False); style_axes(ax); fig.tight_layout(); _savefig(fig, "rev2_lambda_sensitivity")

    if len(indicators):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        hv_col = "hypervolume_per_run" if "hypervolume_per_run" in indicators else "hypervolume"
        igd_col = (
            "igd_to_pooled_reference" if "igd_to_pooled_reference" in indicators
            else ("igd_to_final" if "igd_to_final" in indicators else None)
        )
        cols = [c for c in [hv_col, igd_col] if c is not None]
        med = indicators.groupby(["pool", "generation"])[cols].median().reset_index()
        for name, sub in med.groupby("pool"):
            axes[0].plot(sub["generation"], sub[hv_col], label=name)
            if igd_col is not None:
                axes[1].plot(sub["generation"], sub[igd_col], label=name)
        axes[0].set(xlabel="Generation", ylabel="Hypervolume (per-run normalized)")
        axes[1].set(
            xlabel="Generation",
            ylabel="IGD to pooled reference" if igd_col and "pooled" in igd_col else "IGD",
        )
        axes[0].legend(frameon=False); axes[1].legend(frameon=False)
        style_axes(axes[0]); style_axes(axes[1]); fig.tight_layout()
        _savefig(fig, "rev2_generation_indicators")

        if "hypervolume_shared" in indicators.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            med_s = indicators.groupby(["pool", "generation"])["hypervolume_shared"].median().reset_index()
            for name, sub in med_s.groupby("pool"):
                ax.plot(sub["generation"], sub["hypervolume_shared"], label=name)
            ax.set(xlabel="Generation", ylabel="Hypervolume (shared normalization)")
            ax.legend(frameon=False); style_axes(ax); fig.tight_layout()
            _savefig(fig, "rev2_generation_indicators_shared")

    seedwise = pd.read_csv(EXPORTS / "rev2_seedwise_decision_audits.csv")
    has_b = "s3_top10_failure_category_b" in seedwise.columns
    if has_b:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        panel_specs = (
            (axes[0, 0], "s3_top10_failure_category_b", "Top-10 failure (Category B)"),
            (axes[0, 1], "s3_expected_category_b_regret", "Expected Category B regret"),
            (axes[1, 0], "s3_top10_failure", "Top-10 failure (J)"),
            (axes[1, 1], "s3_expected_J_regret", "Expected J regret"),
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        panel_specs = (
            (axes[0], "s3_top10_failure", "Top-10 failure probability"),
            (axes[1], "s3_expected_J_regret", "Expected J regret"),
        )
    for ax, metric, ylabel in panel_specs:
        wide = seedwise.pivot(index="optimization_seed", columns="pool", values=metric)
        for _, row in wide.iterrows():
            ax.plot([0, 1], [row["mean"], row["q10"]], color="#7EB6D9", linewidth=1)
        means = wide[["mean", "q10"]].mean()
        ax.plot([0, 1], means, "o-", color="#0B3D91", linewidth=2.5, markersize=7)
        ax.set(xticks=[0, 1], xticklabels=["Mean search", "q10 search"], ylabel=ylabel)
        style_axes(ax)
    fig.tight_layout()
    _savefig(fig, "rev2_paired_s3")


def recompute_plateau_only(generations: int | None) -> int:
    """Rewrite the plateau diagnostic from the saved indicator table.

    The diagnostic is a function of the deposited generation indicators alone,
    so it can be corrected without re-optimizing.  The plateau block inside the
    run summary is patched in place for the same reason.
    """
    indicators_path = EXPORTS / "rev2_generation_indicators.csv"
    if not indicators_path.exists():
        raise FileNotFoundError(f"{indicators_path} is required for --plateau-only")
    indicators = pd.read_csv(indicators_path)
    requested = generations or int(indicators["generation"].max())
    payload = plateau_payload_for(indicators, requested)
    write_json(EXPORTS / "rev2_plateau_diagnostics.json", payload)
    summary_path = EXPORTS / "rev2_statistical_reconstruction_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["plateau_diagnostics"] = payload
        write_json(summary_path, summary)
    print(json.dumps({
        "recomputed": "plateau_diagnostics",
        "generations_requested": requested,
        "mean": {
            k: payload["mean"][k]
            for k in ("n_seeds", "relative_increment", "plateaued")
        },
        "q10": {
            k: payload["q10"][k]
            for k in ("n_seeds", "relative_increment", "plateaued")
        },
        "summary_patched": summary_path.exists(),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fast integration run; production remains the default")
    parser.add_argument("--pop", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--mc", type=int, default=None)
    parser.add_argument(
        "--max-seeds", type=int, default=None,
        help=(
            "limit the number of matched seeds (probe uses 1; production uses all "
            f"{len(SPEC.optimization_seeds)})"
        ),
    )
    parser.add_argument(
        "--reuse-pools", action="store_true",
        help="reuse existing matched NSGA-II pools and rerun convergence/audits only",
    )
    parser.add_argument(
        "--skip-saa", action="store_true",
        help="reuse the previously selected SAA N from rev2_saa_convergence.csv",
    )
    parser.add_argument(
        "--plateau-only", action="store_true",
        help=(
            "recompute the plateau diagnostic from the saved generation indicators "
            "and exit; no model loading, optimization, or decision audit"
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    if args.plateau_only:
        return recompute_plateau_only(args.generations)
    EXPORTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    write_specification()
    raw = pd.read_csv(DATA_PATH)
    astm = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    production_path = EXPORTS / "rev2_production_ngb.joblib"
    if not production_path.exists():
        raise FileNotFoundError(
            "Run grouped_loso_validation.py first; the full-data production "
            f"artifact is missing: {production_path}"
        )
    production = joblib.load(production_path)
    # Compatibility wrapper for the stable inference helpers from the earlier shell.
    # Unlike the archived checkpoint, this model and preprocessing were refit on
    # all 540 rows only after source-disjoint validation was complete. Do not
    # swap this back to the archived checkpoint.
    ckpt = {
        "ensemble": production["model"],
        "preprocess_pipeline": production["preprocessing"],
        "all_feature_names": production["all_feature_names"],
    }
    all_names = list(ckpt["all_feature_names"])
    training = astm.copy()
    support = derive_support(training)
    m_mk = mk_mean_from_workbook()
    lam, lambda_source, lambda_payload = load_lambda()

    X_train = align_X(training, all_names)
    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    # Both sides of the applicability test use distance to the k-th nearest
    # training record: the threshold is the quantile of that distance among
    # training records, and candidates are screened on the same statistic.
    k = max(1, min(SPEC.ood_knn_neighbors, len(training) - 1))
    support_nn = NearestNeighbors(n_neighbors=k + 1).fit(Z_train)
    # Column 0 is the record itself at distance 0, so column k is its k-th neighbour.
    self_dist = support_nn.kneighbors(Z_train, return_distance=True)[0][:, k]
    ood_limit = float(np.quantile(self_dist, SPEC.ood_training_quantile))
    knn = NearestNeighbors(n_neighbors=k).fit(Z_train)
    support["ood"] = {
        "standardized_feature_space": True, "neighbor_rank": k,
        "threshold_and_screen_use_same_rank": True,
        "training_self_distance_median": float(np.median(self_dist)),
        "threshold_quantile": SPEC.ood_training_quantile, "distance_limit": ood_limit,
        "active_constraint": True,
    }
    write_json(EXPORTS / "rev2_feasibility_support.json", support)
    write_json(EXPORTS / "rev2_support_rules_v6.json", {
        "version": 6,
        "reference_age_days": SPEC.reference_age_days,
        "mk_practical_zero_kg_m3": SPEC.mk_practical_zero_kg_m3,
        "mk_snap_rule": (
            f"if decoded MK < {SPEC.mk_practical_zero_kg_m3} kg/m3 then MK=0 "
            "and mass is redistributed to OPC before feature construction"
        ),
        "ood_knn_neighbors": k,
        "ood_training_quantile": SPEC.ood_training_quantile,
        "ood_distance_limit": ood_limit,
        "ood_training_self_distance_median": float(np.median(self_dist)),
        "strength_engineering_target_mpa": SPEC.min_strength_mpa,
        "strength_mae_mpa": SPEC.strength_mae_mpa,
        "strength_conservative_k": SPEC.strength_conservative_k,
        "active_strength_floor_mpa": SPEC.active_strength_floor_mpa,
        "strength_ridge_alpha": SPEC.strength_ridge_alpha,
        "saa_production_n": SPEC.saa_production_n,
    })
    write_json(EXPORTS / "rev2_lambda_provenance.json", {
        "production_lambda": lam, "source": lambda_source, "source_payload": lambda_payload,
        "production_model": str(production_path),
        "production_model_validation": production.get("validation_provenance", {}),
        "rerun_behavior": "The script re-reads grouped_loso_validation* on every invocation.",
    })

    base_problem = ReconstructionProblem(
        "q10", max(SPEC.saa_grid), SPEC.optimization_seeds[0], ckpt, all_names,
        support, scaler, knn, ood_limit, m_mk, lam,
    )
    saa_path = EXPORTS / "rev2_saa_convergence.csv"
    if args.skip_saa and saa_path.exists():
        saa = pd.read_csv(saa_path)
        selected_n = int(saa["selected_production_n"].iloc[0])
    else:
        rng = np.random.default_rng(20260730)
        probe_X = rng.random((48 if args.smoke else 256, 9))
        saa_seeds = SPEC.optimization_seeds[:2] if args.smoke else SPEC.optimization_seeds
        saa, selected_n = saa_convergence(base_problem, probe_X, saa_seeds)
        saa.to_csv(saa_path, index=False)

    pop = args.pop or (16 if args.smoke else SPEC.nsga2_population)
    generations = args.generations or (3 if args.smoke else SPEC.nsga2_generations)
    n_seeds = args.max_seeds or (2 if args.smoke else len(SPEC.optimization_seeds))
    seeds = SPEC.optimization_seeds[:n_seeds]
    pools_path = EXPORTS / "rev2_matched_nsga2_pools.csv"
    indicators_path = EXPORTS / "rev2_generation_indicators.csv"
    plateau_path = EXPORTS / "rev2_plateau_diagnostics.json"
    if args.reuse_pools:
        pools = pd.read_csv(pools_path)
        indicators = pd.read_csv(indicators_path) if indicators_path.exists() else pd.DataFrame()
        expected = {(int(seed), mode) for seed in seeds for mode in ("mean", "q10")}
        observed = set(zip(pools["optimization_seed"].astype(int), pools["pool"]))
        counts = pools.groupby(["optimization_seed", "pool"]).size()
        if observed != expected or not (counts == pop).all():
            raise ValueError(
                f"Existing pools are not the registered matched {pop}-candidate design: "
                f"groups={len(observed)}, counts={sorted(counts.unique().tolist())}"
            )
    else:
        pool_frames, front_store = [], []
        for seed in seeds:
            for mode in ("mean", "q10"):
                problem = ReconstructionProblem(
                    mode, selected_n, seed, ckpt, all_names, support, scaler, knn,
                    ood_limit, m_mk, lam,
                )
                res, fronts = optimize(problem, pop, generations)
                frame = pool_rows(problem, res, mode, seed)
                pool_frames.append(frame)
                if fronts:
                    front_store.append({
                        "pool": mode, "optimization_seed": seed, "fronts": fronts,
                    })
        pools = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        pools.insert(0, "candidate_id", np.arange(len(pools)))
        pools.to_csv(pools_path, index=False)
        indicators = annotate_shared_indicators(front_store)
        indicators.to_csv(indicators_path, index=False)

    # Refresh plateau diagnostics whenever indicators are available.
    if len(indicators):
        write_json(plateau_path, plateau_payload_for(indicators, generations))

    audit_frames, lambda_frames, seed_summaries = [], [], []
    mc_n = args.mc or (200 if args.smoke else SPEC.mc_draws)
    perm_n = 500 if args.smoke else SPEC.permutation_draws
    sensitivity_lambdas = tuple(sorted(set((1.0, 1.53, 2.20, lam))))
    for seed in seeds:
        # The same audit seed for both modes is a paired common-random-number design.
        audit_seed = 91_000 + int(seed)
        pools_by_mode: dict[str, pd.DataFrame] = {}
        problems_by_mode: dict[str, ReconstructionProblem] = {}
        for mode in ("mean", "q10"):
            subset = pools[
                (pools["pool"] == mode) & (pools["optimization_seed"] == seed)
            ].sort_values("local_id").reset_index(drop=True)
            if len(subset) < 2:
                print(f"WARNING: skipping audit for {mode} seed={seed} n={len(subset)}")
                continue
            pools_by_mode[mode] = subset
            problems_by_mode[mode] = ReconstructionProblem(
                mode, selected_n, seed, ckpt, all_names, support,
                scaler, knn, ood_limit, m_mk, lam,
            )
        if len(pools_by_mode) < 2:
            print(f"WARNING: skipping matched audit for seed={seed}; need both arms")
            continue
        audits_seed, lambdas_seed, summaries_seed = audit_matched_seed(
            pools_by_mode, problems_by_mode, sensitivity_lambdas, mc_n, perm_n,
            audit_seed,
        )
        audit_frames.extend(audits_seed)
        lambda_frames.extend(lambdas_seed)
        seed_summaries.extend(summaries_seed)
    audits = (
        pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    )
    lambda_sens = (
        pd.concat(lambda_frames, ignore_index=True) if lambda_frames else pd.DataFrame()
    )
    if not len(audits):
        raise RuntimeError(
            "No auditable fronts were produced. Increase population/generations "
            "or relax feasibility constraints."
        )
    audits.to_csv(EXPORTS / "rev2_identical_S3_audits.csv", index=False)
    lambda_sens.to_csv(EXPORTS / "rev2_lambda_sensitivity.csv", index=False)
    seedwise = pd.DataFrame([flatten_seed_summary(item) for item in seed_summaries])
    seedwise.to_csv(EXPORTS / "rev2_seedwise_decision_audits.csv", index=False)
    head_to_head = aggregate_seedwise(seedwise)
    head_to_head["lambda_sensitivity_values"] = list(sensitivity_lambdas)
    head_to_head["common_random_numbers_within_pair"] = True
    if head_to_head.get("mc_convergence_rows"):
        pd.DataFrame(head_to_head["mc_convergence_rows"]).to_csv(
            EXPORTS / "rev2_s3_mc_convergence.csv", index=False,
        )
    write_json(EXPORTS / "rev2_matched_pool_head_to_head.json", head_to_head)
    plateau_payload = {}
    if plateau_path.exists():
        plateau_payload = json.loads(plateau_path.read_text(encoding="utf-8"))
    elif len(indicators):
        plateau_payload = {
            "mean": plateau_relative_increment(indicators, "mean"),
            "q10": plateau_relative_increment(indicators, "q10"),
        }
    write_json(EXPORTS / "rev2_statistical_reconstruction_summary.json", {
        "run_mode": "smoke" if args.smoke else "production",
        "primary_service_function": "t_eq=min(t_init,100 years), used in every durability and Category B calculation",
        "candidate_inference_exposure_days": SPEC.reference_age_days,
        "production_lambda": lam, "lambda_source": lambda_source,
        "selected_saa_n": selected_n,
        "saa_selection_rule": (
            f"Production N={SPEC.saa_production_n} from SPEC; auto-selected "
            "smallest N meeting median tau>=0.90, p10 tau>=0.80, median normalized "
            "MAE<=0.05, median top-10 overlap>=0.80 and p10 overlap>=0.60 vs "
            "nested-CRN N=250 is recorded as auto_selected_n"
        ),
        "active_strength_floor_mpa": SPEC.active_strength_floor_mpa,
        "mk_practical_zero_kg_m3": SPEC.mk_practical_zero_kg_m3,
        "ood_knn_neighbors": SPEC.ood_knn_neighbors,
        "strength_ridge_alpha": SPEC.strength_ridge_alpha,
        "optimization_seeds": list(seeds), "registered_production_seeds": list(SPEC.optimization_seeds),
        "population": pop, "generations": generations,
        "objectives": [
            {"order": 1, "direction": "minimize", "name": "cost", "units": "USD/m3"},
            {"order": 2, "direction": "maximize", "name": "compressive_strength", "units": "MPa"},
            {
                "order": 3, "direction": "maximize", "name": "durability_statistic",
                "units": "years",
                "mean_arm": "mean t_eq", "q10_arm": "q10 t_eq",
            },
            {"order": 4, "direction": "minimize", "name": "GWP", "units": "kg CO2-eq/m3"},
        ],
        "n_decision_variables": 9,
        "n_inequality_constraints": 12,
        "audited_non_objectives": ["AP", "EP"],
        "matched_design": "mean and q10 pools use identical genotype, constraints, NSGA-II seed, and CRN stream per seed",
        "decision_audit_inferential_unit": "optimization seed; n=80 within each seed/pool",
        "pooled_n800_role": "candidate-level storage only; not used for nulls or headline inference",
        "s3_primary": "Category B per-draw GWP/t_eq",
        "s3_secondary": "J = GWP*[Dapp/(1e-12)]",
        "plateau_diagnostics": plateau_payload,
        "support": support, "matched_pool_head_to_head": head_to_head,
        "reused_existing_pools": bool(args.reuse_pools),
        "runtime_seconds": time.perf_counter() - started,
    })
    make_figures(saa, pools, audits, lambda_sens, indicators)
    print(json.dumps({
        "mode": "smoke" if args.smoke else "production", "selected_saa_n": selected_n,
        "lambda": lam, "lambda_source": lambda_source, "pool_rows": len(pools),
        "generations": generations, "n_seeds": len(seeds),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "conclusion": head_to_head.get("conclusion", {}).get("wording"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
