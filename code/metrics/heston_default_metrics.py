"""Heston stochastic volatility structural default metrics.

The Heston model assumes asset dynamics:
    dA_t/A_t = μ dt + √v_t dW_1
    dv_t = κ(θ - v_t) dt + σ_v √v_t dW_2
    dW_1 dW_2 = ρ dt

Default probability P(A_T < D) is computed via characteristic function
and Gil-Pelaez inversion:
    P(X_T < a) = ½ - (1/π) ∫₀^∞ Im[exp(-iua) φ_X(u)] / u du

where X_T = log(A_T/A_0) and φ_X is the Heston characteristic function.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def _safe_float(value: object) -> float:
    """Convert to float and check finiteness."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("Value is not finite.")
    return out


def heston_characteristic_function(
    u: complex,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    drift: float,
) -> complex:
    """Heston characteristic function φ(u) for log(A_T/A_0).
    
    Using the standard formulation (Heston 1993, Gatheral 2006).
    
    Parameters
    ----------
    u : frequency (complex)
    T : time to maturity
    v0 : initial variance
    kappa : mean-reversion speed
    theta : long-run variance
    sigma_v : vol-of-vol
    rho : correlation between asset and variance Brownian motions
    drift : μ for physical, r for risk-neutral
    
    Returns
    -------
    φ(u) : characteristic function value
    """
    i = 1j
    
    # Heston CF parameters
    d = np.sqrt((rho * sigma_v * i * u - kappa)**2 + sigma_v**2 * (i * u + u**2))
    g = (kappa - rho * sigma_v * i * u - d) / (kappa - rho * sigma_v * i * u + d)
    
    # C and D terms
    C = drift * i * u * T + (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * i * u - d) * T - 2.0 * np.log((1.0 - g * np.exp(-d * T)) / (1.0 - g))
    )
    D = ((kappa - rho * sigma_v * i * u - d) / sigma_v**2) * (
        (1.0 - np.exp(-d * T)) / (1.0 - g * np.exp(-d * T))
    )
    
    return np.exp(C + D * v0)


def heston_log_return_cdf_cf(
    log_threshold: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    drift: float,
    *,
    u_max: float = 300.0,
) -> float:
    """P(log(A_T/A_0) < log_threshold) via Gil-Pelaez inversion.
    
    Parameters
    ----------
    log_threshold : log(D/A_0)
    drift : μ for physical, r for risk-neutral
    u_max : upper integration limit
    
    Returns
    -------
    CDF value, clipped to [0, 1]
    """
    a = float(log_threshold)
    
    def integrand(u: float) -> float:
        if u <= 0.0:
            return 0.0
        cf_val = heston_characteristic_function(u, T, v0, kappa, theta, sigma_v, rho, drift)
        val = np.exp(-1j * u * a) * cf_val
        return float(np.imag(val)) / u
    
    integral, _ = quad(integrand, 1e-9, u_max, limit=500, epsabs=1e-8, epsrel=1e-8)
    return float(np.clip(0.5 - integral / math.pi, 0.0, 1.0))


def heston_default_probability(
    asset_value: float,
    debt_face_value: float,
    maturity_years: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    drift: float,
    *,
    measure: str = "physical",
) -> float:
    """Compute default probability P(A_T < D) under Heston.
    
    Parameters
    ----------
    v0 : initial variance (σ²)
    kappa : mean-reversion speed
    theta : long-run variance
    sigma_v : vol-of-vol
    rho : correlation between asset and vol
    drift : μ for physical, r for risk-neutral
    measure : "physical" or "risk_neutral" (for labeling)
    
    Returns
    -------
    PD : Default probability
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    T = _safe_float(maturity_years)
    
    if A0 <= 0 or D <= 0 or T <= 0:
        raise ValueError("asset_value, debt_face_value, maturity_years must be positive.")
    if v0 <= 0 or theta <= 0 or sigma_v <= 0:
        raise ValueError("v0, theta, sigma_v must be positive.")
    
    log_threshold = math.log(D / A0)
    pd = heston_log_return_cdf_cf(log_threshold, T, v0, kappa, theta, sigma_v, rho, drift)
    return _safe_float(np.clip(pd, 0.0, 1.0))


def heston_distance_to_default(
    asset_value: float,
    debt_face_value: float,
    maturity_years: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    mu_physical: float,
) -> float:
    """Pseudo-DD for Heston: DD = -Φ⁻¹(PD_physical).
    
    Returns
    -------
    DD : Equivalent standard-normal distance
    """
    pd_phys = heston_default_probability(
        asset_value, debt_face_value, maturity_years,
        v0, kappa, theta, sigma_v, rho, mu_physical,
        measure="physical"
    )
    dd = float(-norm.ppf(max(pd_phys, 1e-15)))
    return _safe_float(dd)


def heston_lgd_model_implied(
    asset_value: float,
    debt_face_value: float,
    maturity_years: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    drift: float,
) -> float:
    """Model-implied LGD under Heston using share-measure CF.
    
    LGD = 1 - E[A_T | A_T < D] / D
    
    Uses the share-measure (Esscher transform) to compute E[A_T 1_{A_T<D}].
    
    Returns
    -------
    LGD : Loss Given Default, clipped to [0, 1]
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    T = _safe_float(maturity_years)
    
    if A0 <= 0 or D <= 0 or T <= 0:
        raise ValueError("asset_value, debt_face_value, maturity_years must be positive.")
    if v0 <= 0 or theta <= 0 or sigma_v <= 0:
        raise ValueError("v0, theta, sigma_v must be positive.")
    
    # Compute PD
    log_threshold = math.log(D / A0)
    pd = heston_log_return_cdf_cf(log_threshold, T, v0, kappa, theta, sigma_v, rho, drift)
    
    if pd < 1e-15:
        return 0.0
    
    # Share measure: φ^S(u) = φ(u - i) / e^{mT}
    # P^S(X < a) via Gil-Pelaez with share CF
    def share_cf(u: complex) -> complex:
        return heston_characteristic_function(u - 1j, T, v0, kappa, theta, sigma_v, rho, drift) / np.exp(drift * T)
    
    def integrand_share(u: float) -> float:
        if u <= 0.0:
            return 0.0
        cf_val = share_cf(u)
        val = np.exp(-1j * u * log_threshold) * cf_val
        return float(np.imag(val)) / u
    
    integral, _ = quad(integrand_share, 1e-9, 300.0, limit=500, epsabs=1e-8, epsrel=1e-8)
    prob_share = float(np.clip(0.5 - integral / math.pi, 0.0, 1.0))
    
    # E[A_T 1_{A_T<D}] = A0 e^{mT} P^S(X_T < log_threshold)
    truncated_mean = A0 * np.exp(drift * T) * prob_share
    conditional_mean = truncated_mean / pd
    
    lgd = 1.0 - conditional_mean / D
    return _safe_float(np.clip(lgd, 0.0, 1.0))


def heston_metrics_all(
    asset_value: float,
    debt_face_value: float,
    maturity_years: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    mu_physical: float,
    r_risk_neutral: float,
) -> dict[str, float]:
    """Compute all Heston metrics: DD, PD (P & Q), LGD (P & Q).
    
    Returns
    -------
    dict with keys:
        Distance_to_Default_Heston
        PD_Heston_physical
        PD_Heston_risk_neutral
        LGD_Heston_physical
        LGD_Heston_risk_neutral
        Heston_v0
        Heston_kappa
        Heston_theta
        Heston_sigma_v
        Heston_rho
    """
    dd = heston_distance_to_default(
        asset_value, debt_face_value, maturity_years,
        v0, kappa, theta, sigma_v, rho, mu_physical
    )
    pd_phys = heston_default_probability(
        asset_value, debt_face_value, maturity_years,
        v0, kappa, theta, sigma_v, rho, mu_physical,
        measure="physical"
    )
    pd_rn = heston_default_probability(
        asset_value, debt_face_value, maturity_years,
        v0, kappa, theta, sigma_v, rho, r_risk_neutral,
        measure="risk_neutral"
    )
    lgd_phys = heston_lgd_model_implied(
        asset_value, debt_face_value, maturity_years,
        v0, kappa, theta, sigma_v, rho, mu_physical
    )
    lgd_rn = heston_lgd_model_implied(
        asset_value, debt_face_value, maturity_years,
        v0, kappa, theta, sigma_v, rho, r_risk_neutral
    )
    
    return {
        "Distance_to_Default_Heston": dd,
        "PD_Heston_physical": pd_phys,
        "PD_Heston_risk_neutral": pd_rn,
        "LGD_Heston_physical": lgd_phys,
        "LGD_Heston_risk_neutral": lgd_rn,
        "Heston_v0": v0,
        "Heston_kappa": kappa,
        "Heston_theta": theta,
        "Heston_sigma_v": sigma_v,
        "Heston_rho": rho,
    }


if __name__ == "__main__":
    # Example with typical Heston parameters
    A0 = 3.5e12
    D = 1.0e11
    T = 1.0
    v0 = 0.25**2  # initial variance (σ₀²)
    kappa = 2.0    # mean reversion
    theta = 0.25**2  # long-run variance
    sigma_v = 0.3  # vol-of-vol
    rho = -0.7     # correlation
    mu = 0.15
    r = 0.04
    
    metrics = heston_metrics_all(A0, D, T, v0, kappa, theta, sigma_v, rho, mu, r)
    print("Heston Metrics Example:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")
