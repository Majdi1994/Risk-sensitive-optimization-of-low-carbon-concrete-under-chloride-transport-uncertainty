"""Reference-age-consistent chloride-initiation mapping for the current search.

The reference age and the decision horizon come from analysis_specification.py.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfinv

from analysis_specification import SPEC


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def equivalent_diffusion_age(
    time_years: np.ndarray | float,
    m: float,
    reference_age_days: float = SPEC.reference_age_days,
    freeze_years: float | None = None,
) -> np.ndarray:
    """Return integral_0^t D(u)/D_ref du in years.

    D is D_ref until the reference age, follows the power law afterwards, and
    optionally remains constant after ``freeze_years``. This avoids applying
    the aging exponent before the age at which D_ref is defined.
    """
    t = np.maximum(np.asarray(time_years, dtype=float), 0.0)
    tref = float(reference_age_days) / 365.25
    m = float(np.clip(m, 0.0, 0.95))

    if m < 1e-12:
        return t

    def prefreeze_age(x: np.ndarray) -> np.ndarray:
        return np.where(
            x <= tref,
            x,
            tref
            + (tref**m)
            * (np.power(x, 1.0 - m) - tref ** (1.0 - m))
            / (1.0 - m),
        )

    if freeze_years is None:
        return prefreeze_age(t)

    tf = max(float(freeze_years), tref)
    at_freeze = float(prefreeze_age(np.asarray(tf)))
    d_ratio_freeze = (tf / tref) ** (-m)
    return np.where(
        t <= tf,
        prefreeze_age(t),
        at_freeze + d_ratio_freeze * (t - tf),
    )


def invert_equivalent_diffusion_age(
    required_age_years: np.ndarray | float,
    m: float,
    reference_age_days: float = SPEC.reference_age_days,
    freeze_years: float | None = None,
) -> np.ndarray:
    """Invert ``equivalent_diffusion_age`` analytically."""
    b = np.maximum(np.asarray(required_age_years, dtype=float), 0.0)
    tref = float(reference_age_days) / 365.25
    m = float(np.clip(m, 0.0, 0.95))

    if m < 1e-12:
        return b

    rhs = tref ** (1.0 - m) + np.maximum(b - tref, 0.0) * (1.0 - m) / (tref**m)
    t_power = np.power(np.maximum(rhs, 0.0), 1.0 / (1.0 - m))
    t_unfrozen = np.where(b <= tref, b, t_power)

    if freeze_years is None:
        return t_unfrozen

    tf = max(float(freeze_years), tref)
    at_freeze = float(equivalent_diffusion_age(tf, m, reference_age_days, None))
    d_ratio_freeze = (tf / tref) ** (-m)
    t_after = tf + np.maximum(b - at_freeze, 0.0) / d_ratio_freeze
    return np.where(b <= at_freeze, t_unfrozen, t_after)


def initiation_time_years(
    d_ref_m2s: np.ndarray | float,
    m: float,
    cs_percent_binder: np.ndarray | float,
    ccrit_percent_binder: np.ndarray | float,
    cover_m: float = 0.050,
    c0_percent_binder: float = 0.0,
    freeze_years: float | None = None,
) -> np.ndarray:
    """Compute uncapped initiation time from D_ref at the declared age."""
    d = np.maximum(np.asarray(d_ref_m2s, dtype=float), 1e-20)
    cs = np.asarray(cs_percent_binder, dtype=float)
    ccrit = np.asarray(ccrit_percent_binder, dtype=float)
    beta = np.clip(
        (ccrit - c0_percent_binder)
        / np.maximum(cs - c0_percent_binder, 1e-12),
        1e-6,
        1.0 - 1e-6,
    )
    z = np.maximum(erfinv(1.0 - beta), 1e-12)
    required_age = (cover_m**2) / (4.0 * d * z**2) / SECONDS_PER_YEAR
    return invert_equivalent_diffusion_age(
        required_age,
        m,
        reference_age_days=SPEC.reference_age_days,
        freeze_years=freeze_years,
    )


def equivalent_service_time(
    *args,
    horizon_years: float = SPEC.decision_horizon_years,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw initiation time, decision-bounded equivalent time)."""
    raw = initiation_time_years(*args, **kwargs)
    return raw, np.minimum(raw, float(horizon_years))

