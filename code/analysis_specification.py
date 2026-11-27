"""Frozen analysis specification shared by the reproducibility scripts.

Locked choices for decision horizon, reference age, taxonomy, feasibility,
and production search settings live here. Other scripts read this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


REV = Path(__file__).resolve().parents[1]
EXPORTS = REV / "outputs"


@dataclass(frozen=True)
class ReconstructionSpec:
    # Primary decision definition
    reference_age_days: float = 35.0
    decision_horizon_years: float = 100.0
    freeze_sensitivity_years: float = 25.0

    # Numerical audits
    top_k: int = 10
    mc_draws: int = 4000
    mc_convergence_draws: tuple[int, ...] = (500, 1000, 2000, 4000)
    permutation_draws: int = 100_000
    optimization_seeds: tuple[int, ...] = (
        7, 21, 42, 57, 73, 99, 123, 211, 307, 401,
        503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409,
        1483, 1601, 1709, 1801, 1901, 1999, 2111, 2203, 2309, 2411,
    )
    # Production generation budget under the active 5-NN applicability gate.
    nsga2_generations: int = 400
    nsga2_population: int = 80
    hv_reference_point: float = 1.1
    plateau_trailing_generations: int = 10
    plateau_relative_increment: float = 0.01
    saa_grid: tuple[int, ...] = (24, 50, 100, 250)
    # Locked production SAA size.
    saa_production_n: int = 100
    lambda_sensitivity: tuple[float, ...] = (1.00, 1.53, 2.20)

    # Applicability domain: active gate uses 5-NN training self-distance quantile.
    ood_knn_neighbors: int = 5
    ood_training_quantile: float = 0.95

    # Empirical-support limits. Ratios are relative to total binder.
    max_scm_total_ratio: float = 0.60
    max_sf_ratio: float = 0.20
    max_mk_ratio: float = 0.25
    max_fa_ratio: float = 0.70
    max_ggbfs_ratio: float = 0.70
    min_opc_kg_m3: float = 250.0
    # Engineering strength target; active constraint is target + k * nested MAE.
    min_strength_mpa: float = 30.0
    strength_mae_mpa: float = 11.53
    strength_conservative_k: float = 1.0
    # Practical zero: doses below training MK-positive minimum snap to absent.
    mk_practical_zero_kg_m3: float = 20.0
    # Deployed ridge penalty from nested GroupKFold on the full corpus.
    strength_ridge_alpha: float = 100.0

    @property
    def active_strength_floor_mpa(self) -> float:
        return (
            self.min_strength_mpa
            + self.strength_conservative_k * self.strength_mae_mpa
        )

    # Taxonomy
    category_a: str = "volumetric cradle-to-gate declared unit"
    category_b: str = "gate-to-performance efficiency ratio using capped equivalent initiation time"
    category_c: str = "dimensionless transport-normalized screening surrogate"
    s1_name: str = "comparison-basis displacement"
    s2_name: str = "surrogate-form discrepancy"
    s3_name: str = "residual ranking risk"


SPEC = ReconstructionSpec()


def write_specification() -> Path:
    """Export the exact analysis specification for provenance."""
    EXPORTS.mkdir(parents=True, exist_ok=True)
    path = EXPORTS / "rev2_analysis_specification.json"
    payload = asdict(SPEC)
    payload["active_strength_floor_mpa"] = SPEC.active_strength_floor_mpa
    payload["primary_service_function"] = "t_eq = min(t_init, 100 years)"
    payload["aging_primary"] = (
        "piecewise D(t): D_ref through reference age, power-law aging thereafter; "
        "decision metric capped at 100 years"
    )
    payload["aging_sensitivity"] = "same aging law with diffusivity frozen after 25 years"
    payload["category_c_definition"] = (
        "J = GWP * D_tilde, D_tilde = D_app / (1e-12 m2/s)"
    )
    payload["strength_feasibility"] = (
        f"active constraint: predicted_fc - {SPEC.strength_mae_mpa} >= "
        f"{SPEC.min_strength_mpa} MPa (floor {SPEC.active_strength_floor_mpa} MPa)"
    )
    payload["mk_practical_zero_rule"] = (
        f"if MK < {SPEC.mk_practical_zero_kg_m3} kg/m3 then MK = 0 before "
        "feature construction, aging, GWP, and OOD"
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_specification())
