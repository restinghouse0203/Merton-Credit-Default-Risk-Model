"""Black-Scholes / Merton (BSM) structural default metrics.

This module computes:
- Distance to Default (DD)
- Default Probability (PD) under physical and risk-neutral measures
- Loss Given Default (LGD) model-implied

Formulas:
    DD = [log(A₀/D) + (μ - ½σ²)T] / (σ√T)
    PD_physical = Φ(-DD)
    PD_risk_neutral = Φ(-d₂^Q) where d₂^Q uses r instead of μ
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _safe_float(value: object) -> float:
    """Convert to float and check finiteness."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("Value is not finite.")
    return out


def bsm_distance_to_default(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    mu_physical: float,
) -> float:
    """Compute Distance to Default under physical measure.
    
    DD = [log(A₀/D) + (μ - ½σ²)T] / (σ√T)
    
    Parameters
    ----------
    asset_value : A₀
    debt_face_value : D (strike)
    sigma_asset : σ (asset volatility, annualized)
    maturity_years : T
    mu_physical : μ (physical drift, annualized)
    
    Returns
    -------
    DD : Distance to Default
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    sigma = _safe_float(sigma_asset)
    T = _safe_float(maturity_years)
    mu = _safe_float(mu_physical)
    
    if A0 <= 0 or D <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("A0, D, sigma, and T must all be strictly positive.")
    
    dd = (np.log(A0 / D) + (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return _safe_float(dd)


def bsm_default_probability(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    drift: float,
    *,
    measure: str = "physical",
) -> float:
    """Compute default probability P(A_T < D).
    
    PD = Φ(-d₂) where d₂ = [log(A₀/D) + (m - ½σ²)T] / (σ√T)
    
    Parameters
    ----------
    drift : μ for physical, r for risk-neutral
    measure : "physical" or "risk_neutral" (for labeling only)
    
    Returns
    -------
    PD : Default probability
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    sigma = _safe_float(sigma_asset)
    T = _safe_float(maturity_years)
    m = _safe_float(drift)
    
    if A0 <= 0 or D <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("A0, D, sigma, and T must all be strictly positive.")
    
    d2 = (np.log(A0 / D) + (m - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pd = float(norm.cdf(-d2))
    return _safe_float(np.clip(pd, 0.0, 1.0))


def bsm_lgd_model_implied(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    drift: float,
) -> float:
    """Model-implied Loss Given Default (LGD) under BSM.
    
    LGD = 1 - E[A_T | A_T < D] / D
    
    Where:
        E[A_T 1_{A_T<D}] = A₀ e^{mT} Φ(-d₁)
        E[A_T | A_T<D] = E[A_T 1_{A_T<D}] / PD
        d₁ = [log(A₀/D) + (m + ½σ²)T] / (σ√T)
        d₂ = d₁ - σ√T
    
    Parameters
    ----------
    drift : μ for physical measure, r for risk-neutral measure
    
    Returns
    -------
    LGD : Loss Given Default, clipped to [0, 1]
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    sigma = _safe_float(sigma_asset)
    T = _safe_float(maturity_years)
    m = _safe_float(drift)
    
    if A0 <= 0 or D <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("A0, D, sigma, and T must all be strictly positive.")
    
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(A0 / D) + (m + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    
    pd = float(norm.cdf(-d2))
    if pd < 1e-15:
        return 0.0
    
    # E[A_T 1_{A_T<D}]
    truncated_mean = A0 * np.exp(m * T) * norm.cdf(-d1)
    # E[A_T | A_T<D]
    conditional_mean = truncated_mean / pd
    
    lgd = 1.0 - conditional_mean / D
    return _safe_float(np.clip(lgd, 0.0, 1.0))


def bsm_metrics_all(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    mu_physical: float,
    r_risk_neutral: float,
) -> dict[str, float]:
    """Compute all BSM metrics: DD, PD (P & Q), LGD (P & Q).
    
    Returns
    -------
    dict with keys:
        Distance_to_Default
        PD_physical
        PD_risk_neutral
        LGD_model_physical
        LGD_model_risk_neutral
    """
    dd = bsm_distance_to_default(
        asset_value, debt_face_value, sigma_asset, maturity_years, mu_physical
    )
    pd_phys = bsm_default_probability(
        asset_value, debt_face_value, sigma_asset, maturity_years, mu_physical,
        measure="physical"
    )
    pd_rn = bsm_default_probability(
        asset_value, debt_face_value, sigma_asset, maturity_years, r_risk_neutral,
        measure="risk_neutral"
    )
    lgd_phys = bsm_lgd_model_implied(
        asset_value, debt_face_value, sigma_asset, maturity_years, mu_physical
    )
    lgd_rn = bsm_lgd_model_implied(
        asset_value, debt_face_value, sigma_asset, maturity_years, r_risk_neutral
    )
    
    return {
        "Distance_to_Default": dd,
        "PD_physical": pd_phys,
        "PD_risk_neutral": pd_rn,
        "LGD_model_physical": lgd_phys,
        "LGD_model_risk_neutral": lgd_rn,
    }


if __name__ == "__main__":
    # Example: AAPL-like parameters
    A0 = 3.5e12  # $3.5T asset value
    D = 1.0e11   # $100B debt
    sigma = 0.25  # 25% vol
    T = 1.0
    mu = 0.15
    r = 0.04
    
    metrics = bsm_metrics_all(A0, D, sigma, T, mu, r)
    print("BSM Metrics Example:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")
