#!/usr/bin/env python3
"""OPC carbon benchmark for the fronts on the same inventory as impact_from_mix.

Two anchors on the identical cradle-to-gate inventory (water, aggregates,
admixture, and batching included):

  Empirical (headline)
      ASTM C1556 OPC-only records (SF=GGBFS=FA=MK=0), scored through
      impact_from_mix, restricted to measured compressive strength within one
      Ridge LOSO MAE of the front median strength.

  Model-optimal (conservative)
      Grid search over OPC-only cement x w/b x SP inside the training support,
      aggregates fixed at the front convention, retaining the minimum-GWP mixture
      whose production-Ridge predicted strength meets the front median. Also
      reports 1-NN and 5-NN applicability distances against the OOD gate.
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
from chloride_pipeline import GWP, impact_from_mix  # noqa: E402
from grouped_loso_validation import DATA_PATH, aligned_design  # noqa: E402
from analysis_specification import EXPORTS, SPEC  # noqa: E402
from strength_model import FEATURES, StrengthPredictor  # noqa: E402

AGG_TOTAL = 1800.0
RIDGE_MAE = 11.809637115005057


def row_to_mix(row: pd.Series) -> dict[str, float]:
    return {
        "OPC": float(row["Cement"]),
        "FA": float(row["FA"]),
        "GGBFS": float(row["GGBFS"]),
        "SF": float(row["SF"]),
        "MK": float(row["MK"]),
        "Water": float(row["Water"]),
        "NFA": float(row["NFA"]),
        "NCA": float(row["NCA"]),
        "SP": float(row["SP"]),
    }


def make_strength_row(
    cement: float, water: float, nfa: float, nca: float, sp: float,
) -> dict[str, float]:
    binder = cement
    return {
        "Cement": cement,
        "SF": 0.0,
        "GGBFS": 0.0,
        "FA": 0.0,
        "MK": 0.0,
        "Water": water,
        "w_b_ratio": water / binder,
        "NFA": nfa,
        "NCA": nca,
        "SP": sp,
        "Total_Binder": binder,
        "SCM_Replacement_Ratio": 0.0,
        "SF_Ratio": 0.0,
        "GGBFS_Ratio": 0.0,
        "FA_Ratio": 0.0,
        "MK_Ratio": 0.0,
    }


def make_ood_row(
    cement: float, water: float, nfa: float, nca: float, sp: float,
    exposure_duration: float = 28.0,
    exposure_temperature: float = 23.0,
) -> dict[str, float]:
    """Build a row that create_advanced_features / aligned_design can consume."""
    return {
        "Cement": cement,
        "SF": 0.0,
        "GGBFS": 0.0,
        "FA": 0.0,
        "MK": 0.0,
        "Water": water,
        "w_b_ratio": water / cement,
        "NFA": nfa,
        "NCA": nca,
        "SP": sp,
        "CS": 43.0,  # placeholder; OOD features do not depend on CS
        "Exposure_Duration": exposure_duration,
        "Exposure_Temperature": exposure_temperature,
        "Da": 1.0,
        "Total_Binder": cement,
        "SCM_Replacement_Ratio": 0.0,
        "SF_Ratio": 0.0,
        "GGBFS_Ratio": 0.0,
        "FA_Ratio": 0.0,
        "MK_Ratio": 0.0,
        "Has_SF": 0,
        "Has_GGBFS": 0,
        "Has_FA": 0,
        "Has_MK": 0,
        "Has_Any_SCM": 0,
        "SCM_Count": 0,
        "Test_Method": "ASTM C1556",
        "Test_Method_Encoded": 0,
    }


def main() -> int:
    pools = pd.read_csv(EXPORTS / "rev2_matched_nsga2_pools.csv")
    front_strength_median = float(pools["strength"].median())
    front_gwp_min = float(pools["gwp"].min())
    front_gwp_max = float(pools["gwp"].max())
    front_gwp_median = float(pools["gwp"].median())
    front_fine_ratio = float(pools["fine_agg_ratio"].median())

    raw = pd.read_csv(DATA_PATH)
    training = raw[raw["Test_Method"] == "ASTM C1556"].reset_index(drop=True)
    opc = training[
        (training["SF"] == 0)
        & (training["GGBFS"] == 0)
        & (training["FA"] == 0)
        & (training["MK"] == 0)
    ].copy()

    # --- Empirical anchor -------------------------------------------------
    gwps = [impact_from_mix(row_to_mix(r), GWP) for _, r in opc.iterrows()]
    opc["gwp_full_boundary"] = gwps
    band_lo = front_strength_median - RIDGE_MAE
    band_hi = front_strength_median + RIDGE_MAE
    matched = opc[
        (opc["CS"] >= band_lo) & (opc["CS"] <= band_hi)
    ].copy()
    emp_gwp = matched["gwp_full_boundary"].to_numpy(dtype=float)
    empirical = {
        "n_opc_only_records": int(len(opc)),
        "n_strength_matched": int(len(matched)),
        "strength_band_mpa": [float(band_lo), float(band_hi)],
        "front_strength_median_mpa": front_strength_median,
        "ridge_loso_mae_mpa": RIDGE_MAE,
        "gwp_median": float(np.median(emp_gwp)) if len(emp_gwp) else float("nan"),
        "gwp_q25": float(np.quantile(emp_gwp, 0.25)) if len(emp_gwp) else float("nan"),
        "gwp_q75": float(np.quantile(emp_gwp, 0.75)) if len(emp_gwp) else float("nan"),
        "gwp_mean": float(np.mean(emp_gwp)) if len(emp_gwp) else float("nan"),
        "gwp_min": float(np.min(emp_gwp)) if len(emp_gwp) else float("nan"),
        "gwp_max": float(np.max(emp_gwp)) if len(emp_gwp) else float("nan"),
        "measured_cs_median": float(matched["CS"].median()) if len(matched) else float("nan"),
        "measured_cs_min": float(matched["CS"].min()) if len(matched) else float("nan"),
        "measured_cs_max": float(matched["CS"].max()) if len(matched) else float("nan"),
        "note": (
            "OPC-only ASTM C1556 records scored through impact_from_mix "
            "(identical boundary to the fronts), restricted to measured CS "
            "within one ridge LOSO MAE of the front median strength."
        ),
    }

    # --- Model-optimal anchor ---------------------------------------------
    strength = StrengthPredictor()
    cement_grid = np.linspace(280.0, 500.0, 45)
    wb_grid = np.linspace(0.30, 0.55, 26)
    sp_grid = np.array([0.0, 1.0, 2.0, 3.1, 4.0, 5.0, 7.0, 10.0])

    # Fit OOD machinery once, then score every strength-feasible grid point.
    production = joblib.load(EXPORTS / "rev2_production_ngb.joblib")
    all_names = list(production["all_feature_names"])
    X_train = aligned_design(training, all_names)
    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    knn1 = NearestNeighbors(n_neighbors=1).fit(Z_train)
    knn5 = NearestNeighbors(n_neighbors=min(5, len(training))).fit(Z_train)
    k1 = max(1, min(SPEC.ood_knn_neighbors, len(training) - 1))
    self_1 = NearestNeighbors(n_neighbors=k1 + 1).fit(Z_train).kneighbors(
        Z_train, return_distance=True
    )[0][:, k1]
    ood_limit = float(np.quantile(self_1, SPEC.ood_training_quantile))

    best: dict | None = None
    best_ood: dict | None = None
    rows_out: list[dict] = []
    for cement in cement_grid:
        for wb in wb_grid:
            water = cement * wb
            for sp in sp_grid:
                nfa = front_fine_ratio * AGG_TOTAL
                nca = (1.0 - front_fine_ratio) * AGG_TOTAL
                rec = make_strength_row(cement, water, nfa, nca, sp)
                pred_cs = float(strength.predict(pd.DataFrame([rec]))[0])
                if pred_cs + 1e-9 < front_strength_median:
                    continue
                mix = {
                    "OPC": cement, "FA": 0.0, "GGBFS": 0.0, "SF": 0.0, "MK": 0.0,
                    "Water": water, "NFA": nfa, "NCA": nca, "SP": sp,
                }
                gwp = float(impact_from_mix(mix, GWP))
                ood_row = make_ood_row(cement, water, nfa, nca, sp)
                Z_ref = scaler.transform(aligned_design(pd.DataFrame([ood_row]), all_names))
                d1 = float(knn1.kneighbors(Z_ref, return_distance=True)[0][0, 0])
                # Production screens on the k-th neighbour distance (k=5), not the mean.
                d5 = float(knn5.kneighbors(Z_ref, return_distance=True)[0][0, -1])
                cand = {
                    "cement": float(cement),
                    "w_b_ratio": float(wb),
                    "water": float(water),
                    "nfa": float(nfa),
                    "nca": float(nca),
                    "sp": float(sp),
                    "predicted_cs": pred_cs,
                    "gwp": gwp,
                    "ood_1nn_distance": d1,
                    "ood_5nn_distance": d5,
                    "passes_production_ood_gate": bool(d5 <= ood_limit),
                }
                rows_out.append(cand)
                if best is None or gwp < best["gwp"]:
                    best = cand
                if cand["passes_production_ood_gate"] and (
                    best_ood is None or gwp < best_ood["gwp"]
                ):
                    best_ood = cand

    if best is None:
        raise RuntimeError(
            "No OPC-only mixture in the grid met the front median strength "
            f"target of {front_strength_median:.2f} MPa under the ridge model."
        )
    # Prefer the OOD-passing minimum; fall back to unconstrained with a flag.
    chosen = best_ood if best_ood is not None else best

    model_optimal = {
        **chosen,
        "agg_total_kg_m3": AGG_TOTAL,
        "fine_agg_ratio": front_fine_ratio,
        "target_strength_mpa": front_strength_median,
        "ood_knn_limit_production": ood_limit,
        "ood_knn_neighbors": int(k1),
        "n_grid_feasible": int(len(rows_out)),
        "n_grid_feasible_and_ood": int(
            sum(1 for r in rows_out if r["passes_production_ood_gate"])
        ),
        "unconstrained_min_gwp": float(best["gwp"]),
        "unconstrained_passes_ood": bool(best["passes_production_ood_gate"]),
        "selection_rule": (
            "minimum GWP among strength-feasible grid points that also pass "
            "the production 5-NN OOD gate; unconstrained min retained for audit"
            if best_ood is not None
            else "no OOD-passing grid point; unconstrained strength-feasible min used"
        ),
        "note": (
            "Minimum-GWP OPC-only mixture inside the training cement/w_b/SP "
            "support whose production-ridge predicted strength meets the "
            "front median and that passes the production OOD gate; aggregates "
            "fixed at the front convention."
        ),
    }

    # Legacy cement-only reference retained for the audit trail.
    cement_only_legacy = 400.0 * float(GWP["OPC"])

    out = {
        "question": (
            "What is a system-boundary-consistent OPC reference GWP against "
            "which the exported fronts can be compared?"
        ),
        "front": {
            "gwp_min": front_gwp_min,
            "gwp_max": front_gwp_max,
            "gwp_median": front_gwp_median,
            "strength_median_mpa": front_strength_median,
            "fine_agg_ratio_median": front_fine_ratio,
        },
        "inventory": {k: float(v) for k, v in GWP.items()},
        "boundary": (
            "impact_from_mix: Production overhead + binder + water + "
            "fine/coarse aggregate + admixture (identical to exported fronts)"
        ),
        "superseded_cement_only_reference": {
            "gwp": cement_only_legacy,
            "why_superseded": (
                "Cement mass x cement factor alone; excludes water, aggregates, "
                "admixture and batching, so the boundary differs from the fronts."
            ),
        },
        "empirical_headline": empirical,
        "model_optimal_conservative": model_optimal,
        "relative_reductions": {
            "front_low_vs_empirical_median": float(
                1.0 - front_gwp_min / empirical["gwp_median"]
            ) if empirical["gwp_median"] == empirical["gwp_median"] else float("nan"),
            "front_median_vs_empirical_median": float(
                1.0 - front_gwp_median / empirical["gwp_median"]
            ) if empirical["gwp_median"] == empirical["gwp_median"] else float("nan"),
            "front_low_vs_model_optimal": float(
                1.0 - front_gwp_min / model_optimal["gwp"]
            ),
            "front_median_vs_model_optimal": float(
                1.0 - front_gwp_median / model_optimal["gwp"]
            ),
        },
    }

    (EXPORTS / "rev2_opc_reference_benchmark.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8",
    )
    pd.DataFrame(rows_out).to_csv(
        EXPORTS / "rev2_opc_reference_benchmark.csv", index=False,
    )
    matched[
        ["Cement", "Water", "w_b_ratio", "NFA", "NCA", "SP", "CS", "gwp_full_boundary", "source_id"]
    ].to_csv(EXPORTS / "rev2_opc_empirical_matched.csv", index=False)

    print(json.dumps({
        "empirical_n": empirical["n_strength_matched"],
        "empirical_gwp_median": round(empirical["gwp_median"], 2),
        "empirical_gwp_iqr": [
            round(empirical["gwp_q25"], 2), round(empirical["gwp_q75"], 2),
        ],
        "model_optimal_gwp": round(model_optimal["gwp"], 2),
        "model_optimal_mix": {
            "cement": round(model_optimal["cement"], 1),
            "w_b": round(model_optimal["w_b_ratio"], 3),
            "sp": round(model_optimal["sp"], 2),
            "pred_cs": round(model_optimal["predicted_cs"], 2),
        },
        "front_low_vs_empirical": round(
            out["relative_reductions"]["front_low_vs_empirical_median"] * 100, 1,
        ),
        "front_low_vs_model_optimal": round(
            out["relative_reductions"]["front_low_vs_model_optimal"] * 100, 1,
        ),
        "legacy_cement_only": cement_only_legacy,
        "ood_1nn": round(model_optimal["ood_1nn_distance"], 3),
        "passes_ood": model_optimal["passes_production_ood_gate"],
        "n_ood_feasible": model_optimal["n_grid_feasible_and_ood"],
        "unconstrained_min_gwp": round(model_optimal["unconstrained_min_gwp"], 2),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
