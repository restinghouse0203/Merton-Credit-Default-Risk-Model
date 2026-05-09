"""Merton Jump-Diffusion structural default metrics.

Under Merton's jump-diffusion model, conditional on N_T = n jumps:
    log(A_T/A_0) | N_T=n ~ N((drift_n - ½σ_n²)T, σ_n²T)

where:
    σ_n² = σ² + n·σ_J²/T
    drift_n = μ − λk + n·γ/T  (physical)
    drift_n = r − λk + n·γ/T  (risk-neutral)
    k = E[J] - 1 = exp(μ_J + ½σ_J²) - 1
    γ = log(1 + k)

Default probability is a Poisson mixture:
    PD = Σ_n P(N_T=n) Φ(-d₂_n)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _safe_float(value: object) -> float:
    """Convert to float and check finiteness."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("Value is not finite.")
    return out


def merton_jump_default_probability(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    drift: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    *,
    measure: str = "physical",
    n_terms: int = 50,
) -> float:
    """Compute PD under Merton jump-diffusion via Poisson mixture.
    
    Parameters
    ----------
    drift : μ for physical, r for risk-neutral
    jump_intensity : λ (annualized jump rate)
    jump_mean : μ_J (mean log-jump size)
    jump_vol : σ_J (std-dev of log-jump size)
    measure : "physical" or "risk_neutral"
    n_terms : Number of Poisson terms to sum
    
    Returns
    -------
    PD : Default probability
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    sigma = _safe_float(sigma_asset)
    T = _safe_float(maturity_years)
    m = _safe_float(drift)
    lam = float(jump_intensity)
    muJ = float(jump_mean)
    sigmaJ = float(jump_vol)
    
    if A0 <= 0 or D <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("A0, D, sigma, T must be strictly positive.")
    if lam < 0:
        raise ValueError("jump_intensity must be non-negative.")
    
    # Jump compensation
    k = float(np.exp(muJ + 0.5 * sigmaJ**2) - 1.0)
    gamma = float(np.log(1.0 + k))
    
    # Risk-neutral intensity
    if measure == "risk_neutral":
        lambda_eff = lam * (1.0 + k)
    else:
        lambda_eff = lam
    
    pd = 0.0
    for n in range(n_terms):
        # Conditional volatility
        sigma_n = float(np.sqrt(sigma**2 + n * sigmaJ**2 / T))
        sigma_n_sqrt_T = sigma_n * float(np.sqrt(T))
        if sigma_n_sqrt_T == 0.0:
            continue
        
        # Poisson weight: P(N_T = n)
        log_w = -lambda_eff * T + n * np.log(lambda_eff * T + 1e-300) - math.lgamma(n + 1)
        w = float(np.exp(log_w))
        
        # Conditional drift
        drift_n = m - lam * k + n * gamma / T
        d2_n = (np.log(A0 / D) + (drift_n - 0.5 * sigma_n**2) * T) / sigma_n_sqrt_T
        
        pd += w * float(norm.cdf(-d2_n))
    
    return _safe_float(np.clip(pd, 0.0, 1.0))


def merton_jump_distance_to_default(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    mu_physical: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    *,
    n_terms: int = 50,
) -> float:
    """Pseudo-DD for Merton Jump: DD = -Φ⁻¹(PD_physical).
    
    When λ=0 this reduces to the plain BSM DD.
    
    Returns
    -------
    DD : Equivalent standard-normal distance
    """
    pd_phys = merton_jump_default_probability(
        asset_value, debt_face_value, sigma_asset, maturity_years,
        mu_physical, jump_intensity, jump_mean, jump_vol,
        measure="physical", n_terms=n_terms
    )
    dd = float(-norm.ppf(max(pd_phys, 1e-15)))
    return _safe_float(dd)


def merton_jump_lgd_model_implied(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    drift: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    *,
    measure: str = "physical",
    n_terms: int = 50,
) -> float:
    """Model-implied LGD under Merton Jump-Diffusion.
    
    LGD = 1 - E[A_T | A_T < D] / D
    
    Computed as a Poisson mixture of truncated lognormal means.
    
    Returns
    -------
    LGD : Loss Given Default, clipped to [0, 1]
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    sigma = _safe_float(sigma_asset)
    T = _safe_float(maturity_years)
    m = _safe_float(drift)
    lam = float(jump_intensity)
    muJ = float(jump_mean)
    sigmaJ = float(jump_vol)
    
    if A0 <= 0 or D <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("A0, D, sigma, T must be strictly positive.")
    if lam < 0:
        raise ValueError("jump_intensity must be non-negative.")
    
    k = float(np.exp(muJ + 0.5 * sigmaJ**2) - 1.0)
    gamma = float(np.log(1.0 + k))
    
    if measure == "risk_neutral":
        lambda_eff = lam * (1.0 + k)
    else:
        lambda_eff = lam
    
    pd = 0.0
    truncated_mean = 0.0
    
    for n in range(n_terms):
        sigma_n = float(np.sqrt(sigma**2 + n * sigmaJ**2 / T))
        sigma_n_sqrt_T = sigma_n * float(np.sqrt(T))
        if sigma_n_sqrt_T == 0.0:
            continue
        
        log_w = -lambda_eff * T + n * np.log(lambda_eff * T + 1e-300) - math.lgamma(n + 1)
        w = float(np.exp(log_w))
        
        drift_n = m - lam * k + n * gamma / T
        d2_n = (np.log(A0 / D) + (drift_n - 0.5 * sigma_n**2) * T) / sigma_n_sqrt_T
        d1_n = d2_n + sigma_n_sqrt_T
        
        pd += w * float(norm.cdf(-d2_n))
        truncated_mean += w * A0 * np.exp(drift_n * T) * norm.cdf(-d1_n)
    
    if pd < 1e-15:
        return 0.0
    
    conditional_mean = truncated_mean / pd
    lgd = 1.0 - conditional_mean / D
    return _safe_float(np.clip(lgd, 0.0, 1.0))


def merton_jump_metrics_all(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    mu_physical: float,
    r_risk_neutral: float,
    jump_intensity: float = 5.0,
    jump_mean: float = -0.05,
    jump_vol: float = 0.10,
    *,
    n_terms: int = 50,
) -> dict[str, float]:
    """Compute all Merton Jump metrics: DD, PD (P & Q), LGD (P & Q).
    
    Returns
    -------
    dict with keys:
        Distance_to_Default_Merton_Jump
        PD_Merton_Jump_physical
        PD_Merton_Jump_risk_neutral
        LGD_Merton_Jump_physical
        LGD_Merton_Jump_risk_neutral
        MertonJump_lambda
        MertonJump_muJ
        MertonJump_sigmaJ
    """
    dd = merton_jump_distance_to_default(
        asset_value, debt_face_value, sigma_asset, maturity_years,
        mu_physical, jump_intensity, jump_mean, jump_vol, n_terms=n_terms
    )
    pd_phys = merton_jump_default_probability(
        asset_value, debt_face_value, sigma_asset, maturity_years,
        mu_physical, jump_intensity, jump_mean, jump_vol,
        measure="physical", n_terms=n_terms
    )
    pd_rn = merton_jump_default_probability(
        asset_value, debt_face_value, sigma_asset, maturity_years,
        r_risk_neutral, jump_intensity, jump_mean, jump_vol,
        measure="risk_neutral", n_terms=n_terms
    )
    lgd_phys = merton_jump_lgd_model_implied(
        asset_value, debt_face_value, sigma_asset, maturity_years,
        mu_physical, jump_intensity, jump_mean, jump_vol,
        measure="physical", n_terms=n_terms
    )
    lgd_rn = merton_jump_lgd_model_implied(
        asset_value, debt_face_value, sigma_asset, maturity_years,
        r_risk_neutral, jump_intensity, jump_mean, jump_vol,
        measure="risk_neutral", n_terms=n_terms
    )
    
    return {
        "Distance_to_Default_Merton_Jump": dd,
        "PD_Merton_Jump_physical": pd_phys,
        "PD_Merton_Jump_risk_neutral": pd_rn,
        "LGD_Merton_Jump_physical": lgd_phys,
        "LGD_Merton_Jump_risk_neutral": lgd_rn,
        "MertonJump_lambda": jump_intensity,
        "MertonJump_muJ": jump_mean,
        "MertonJump_sigmaJ": jump_vol,
    }


if __name__ == "__main__":
    # Example
    A0 = 3.5e12
    D = 1.0e11
    sigma = 0.25
    T = 1.0
    mu = 0.15
    r = 0.04
    
    metrics = merton_jump_metrics_all(
        A0, D, sigma, T, mu, r,
        jump_intensity=5.0, jump_mean=-0.05, jump_vol=0.10
    )
    print("Merton Jump Metrics Example:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")
