"""Structural default probability using Kou's jump-diffusion return density.

This mirrors the Merton pipeline idea (default when assets fall below debt at
horizon T) but replaces the lognormal closure with the Kou (2002) small-interval
approximation for the *simple* return G over Δt = T:

    (P_{t+Δt} - P_t) / P_t ≈ μ Δt + σ ε sqrt(Δt) + I X,

with I ~ Bernoulli(λ Δt) and X double-exponential (symmetric scale η, mean jump
κ in the return). The PDF g(x) is Eq. (6.31)-style:

    ω = x - μ Δt - κ,
    g(x) = (λ Δt)/(2η) exp(σ² Δt / (2η²)) [ exp(-ω/η) Φ(...) + exp(ω/η) Φ(...) ]
         + (1 - λ Δt) (1/(σ√Δt)) f((x - μ Δt)/(σ√Δt)).

Structural mapping (simple return on assets): approximate A_T = A_0 (1 + G), so

    PD ≈ P(G < D/A_0 - 1).

Caveats
-------
- The density is derived for small λ Δt (single-jump Bernoulli). If λ*T is not
  small, interpret results cautiously or compound/simulate.
- The Merton pipeline uses log-return volatility as a proxy; Kou's σ is the
  Gaussian component of *simple* returns. For small Δt they are close.
- Risk-neutral default here uses the same g with μ replaced by r in the
  diffusion/jump-mixture formula; a fully consistent Q measure also adjusts
  jump intensity and tail parameters (Esscher transform). Pass explicit Q
  parameters when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

import pandas as pd

from merton_default_pipeline import (
    DEFAULT_RF_PATH,
    FirmDateInput,
    compute_metrics_for_firm_date,
    load_risk_free_series,
)


def kou_simple_return_pdf(
    x: float | np.ndarray,
    mu: float,
    sigma: float,
    dt: float,
    lam: float,
    eta: float,
    kappa: float,
) -> float | np.ndarray:
    """Probability density g(x) for Kou's approximate simple return G (one interval)."""
    mu = float(mu)
    sigma = float(sigma)
    dt = float(dt)
    lam = float(lam)
    eta = float(eta)
    kappa = float(kappa)

    if sigma <= 0 or dt <= 0 or eta <= 0:
        raise ValueError("sigma, dt, and eta must be strictly positive.")
    if lam < 0:
        raise ValueError("lam (jump intensity) must be non-negative.")

    lam_dt = float(lam * dt)
    if lam_dt > 1.0:
        raise ValueError(
            "lam * dt must be <= 1 for the Bernoulli single-jump approximation. "
            "Reduce lambda, horizon dt, or compound many small intervals."
        )

    x_arr = np.asarray(x, dtype=float)
    sig_sqrt_dt = sigma * np.sqrt(dt)
    omega = x_arr - mu * dt - kappa

    denom_phi = sigma * eta * np.sqrt(dt)
    z1 = (omega * eta - sigma**2 * dt) / denom_phi
    # Kou (2002) / Tsay: second Φ uses minus on (ωη + σ²Δt) so tails decay (symmetric DE limit).
    z2 = -(omega * eta + sigma**2 * dt) / denom_phi

    # Log-domain jump piece avoids overflow in exp(σ²Δt/(2η²)) when η is small vs σ√Δt.
    if lam_dt <= 0.0:
        jump_part = np.zeros_like(x_arr, dtype=float)
    else:
        log_mix = np.log(lam_dt) - np.log(2.0 * eta) + (sigma**2 * dt) / (2.0 * eta**2)
        log_t1 = -omega / eta + norm.logcdf(z1)
        log_t2 = omega / eta + norm.logcdf(z2)
        stacked = np.stack([log_t1, log_t2], axis=0)
        log_jump = log_mix + logsumexp(stacked, axis=0)
        jump_part = np.exp(np.clip(log_jump, -745.0, 745.0))

    z = (x_arr - mu * dt) / sig_sqrt_dt
    diff_part = (1.0 - lam_dt) * norm.pdf(z) / sig_sqrt_dt

    out = jump_part + diff_part
    if np.ndim(out) == 0:
        return float(out)
    return out


def kou_simple_return_cdf(
    x: float,
    mu: float,
    sigma: float,
    dt: float,
    lam: float,
    eta: float,
    kappa: float,
    *,
    eps: float = 1e-12,
) -> float:
    """CDF of G at x: integral_{-inf}^{x} g(u) du via adaptive quadrature."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("x must be finite.")

    def integrand(u: float) -> float:
        return float(kou_simple_return_pdf(u, mu, sigma, dt, lam, eta, kappa))

    # Left tail: density is negligible far left for typical parameters.
    left = min(x - 50.0 * sigma * np.sqrt(dt), -10.0)
    val, err = quad(integrand, left, x, limit=200, epsabs=eps, epsrel=eps)
    if not np.isfinite(val):
        raise RuntimeError("Quadrature did not return a finite CDF value.")
    return float(np.clip(val, 0.0, 1.0))


def kou_structural_default_prob(
    asset_value: float,
    debt_face_value: float,
    mu: float,
    sigma: float,
    maturity_years: float,
    lam: float,
    eta: float,
    kappa: float,
    *,
    measure: Literal["physical", "risk_neutral"] = "physical",
    risk_free_rate: float | None = None,
) -> dict[str, float]:
    """Default prob P(A_T < D) with A_T ≈ A_0 (1 + G) and G ~ Kou density.

    measure
        ``physical`` uses ``mu``; ``risk_neutral`` uses ``risk_free_rate`` as
        the drift in g(·) (same λ, η, κ unless you pass a separate call with
        Q-calibrated jump parameters).
    """
    A0 = float(asset_value)
    D = float(debt_face_value)
    T = float(maturity_years)

    if A0 <= 0 or D <= 0 or T <= 0:
        raise ValueError("asset_value, debt_face_value, maturity_years must be positive.")

    if measure == "physical":
        drift = float(mu)
    else:
        if risk_free_rate is None:
            raise ValueError("risk_free_rate is required when measure='risk_neutral'.")
        drift = float(risk_free_rate)

    threshold = D / A0 - 1.0
    pd = kou_simple_return_cdf(
        threshold,
        drift,
        sigma,
        T,
        lam,
        eta,
        kappa,
    )
    return {
        "PD_Kou": float(np.clip(pd, 0.0, 1.0)),
        "simple_return_threshold": float(threshold),
        "lambda_times_dt": float(lam * T),
    }


def kou_log_return_cdf_cf(
    log_threshold: float,
    mu: float,
    sigma: float,
    T: float,
    lam: float,
    kappa: float,
    eta: float,
    u_max: float = 300.0,
) -> float:
    """P(X_T < log_threshold) via Gil-Pelaez inversion of the exact Kou CF.

    Computes the CDF of the *log*-return X_T = log(A_T/A_0) at a given
    threshold, using the exact characteristic function of the full
    Poisson–compound-jump model (no small-Δt Bernoulli approximation).

    The CF of X_T under the Tsay Laplace(kappa, eta) parameterisation is:

        φ_X(u) = exp( iu·μ_x·T  −  ½σ²u²T  +  λT·(φ_Y(u) − 1) )

    where:
        μ_x    = μ − ½σ² − λψ           (compensated drift)
        ψ      = exp(κ)/(1−η²) − 1      (Kou compensator = E[J]−1)
        φ_Y(u) = exp(iuκ)/(1 + η²u²)   (CF of Laplace(κ,η) log-jump)

    Gil-Pelaez inversion:
        P(X_T < a) = ½ − (1/π) ∫₀^∞ Im[exp(−iua)·φ_X(u)] / u  du

    Parameters
    ----------
    log_threshold : log(D/A_0)  (negative for in-the-money default)
    mu            : annualised drift (physical μ or risk-free r)
    kappa         : Laplace location (mean log-jump)
    eta           : Laplace scale (must satisfy 0 < eta < 1)
    """
    if eta <= 0.0 or eta >= 1.0:
        raise ValueError("eta must satisfy 0 < eta < 1.")
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma and T must be strictly positive.")

    psi = math.exp(kappa) / (1.0 - eta ** 2) - 1.0
    mu_x = mu - 0.5 * sigma ** 2 - lam * psi
    a = float(log_threshold)

    def phi_Y(u: complex) -> complex:
        return np.exp(1j * u * kappa) / (1.0 + eta ** 2 * u ** 2)

    def phi_X(u: complex) -> complex:
        return np.exp(
            1j * u * mu_x * T
            - 0.5 * sigma ** 2 * u ** 2 * T
            + lam * T * (phi_Y(u) - 1.0)
        )

    def integrand(u: float) -> float:
        if u <= 0.0:
            return 0.0
        val = np.exp(-1j * u * a) * phi_X(u)
        return float(np.imag(val)) / u

    integral, _ = quad(integrand, 1e-9, u_max, limit=500, epsabs=1e-8, epsrel=1e-8)
    return float(np.clip(0.5 - integral / math.pi, 0.0, 1.0))


def kou_structural_pd_cf(
    asset_value: float,
    debt_face_value: float,
    mu: float,
    sigma: float,
    maturity_years: float,
    lam: float,
    kappa: float,
    eta: float,
    *,
    measure: Literal["physical", "risk_neutral"] = "physical",
    risk_free_rate: float | None = None,
) -> dict[str, float]:
    """Structural default prob P(A_T < D) using the exact Kou log-return CF.

    Default occurs when the log asset return falls below the log leverage:
        A_T < D  ⟺  X_T = log(A_T/A_0) < log(D/A_0)

    This uses the full Poisson compound-jump CF (valid for any T and λ),
    unlike ``kou_structural_default_prob`` which uses the Bernoulli
    small-interval approximation (only valid when λ·T ≪ 1).

    Returns a dict with:
        PD_Kou            : default probability
        DD_Kou            : pseudo-DD = −Φ⁻¹(PD_Kou_physical), equivalent
                            normal distance (matches Merton DD when λ = 0)
        log_threshold     : log(D/A_0)
    """
    A0 = float(asset_value)
    D = float(debt_face_value)
    T = float(maturity_years)

    if A0 <= 0 or D <= 0 or T <= 0:
        raise ValueError("asset_value, debt_face_value, maturity_years must be positive.")

    drift = mu if measure == "physical" else (
        risk_free_rate if risk_free_rate is not None else
        (_ for _ in ()).throw(ValueError("risk_free_rate required for risk_neutral."))
    )
    if measure == "risk_neutral" and risk_free_rate is None:
        raise ValueError("risk_free_rate is required when measure='risk_neutral'.")
    drift = mu if measure == "physical" else float(risk_free_rate)  # type: ignore[arg-type]

    log_thr = math.log(D / A0)
    pd_val = kou_log_return_cdf_cf(log_thr, drift, sigma, T, lam, kappa, eta)

    return {
        "PD_Kou": float(np.clip(pd_val, 0.0, 1.0)),
        "log_threshold": log_thr,
        "lambda_times_T": float(lam * T),
    }


@dataclass(frozen=True)
class KouJumpParams:
    """Jump component of Kou-style return (annualized λ; per-interval λΔt in g).

    ``eta`` is the symmetric double-exponential scale in the *return* density
    (same units as simple returns). If ``sigma**2 * dt / (2*eta**2)`` is huge
    (very small ``eta`` vs ``sigma * sqrt(dt)``), the Kou prefactor blows up;
    calibrate ``eta`` to jump magnitudes or use a horizon ``dt`` consistent
    with how the density was fit (often one trading day).
    """

    lam: float = 0.05
    eta: float = 0.12
    kappa: float = 0.0


def compute_kou_metrics_for_firm_date(
    ticker: str,
    as_of_date: str,
    maturity_years: float,
    rf_df,
    kou: KouJumpParams | None = None,
    lookback_days: int = 252,
) -> dict[str, float | str]:
    """One row: same asset proxy as Merton pipeline + Kou PDs and DD.

    Uses the exact CF-based log-return CDF (``kou_structural_pd_cf``), which
    is valid for any T and λ.  The old Bernoulli simple-return approximation
    is kept in ``kou_structural_default_prob`` for small-interval use but is
    NOT used here.

    Distance to Default (Kou):
        DD_Kou = −Φ⁻¹(PD_Kou_physical)
    This is the equivalent standard-normal distance so that when λ=0 it
    equals the plain Merton DD exactly.
    """
    if kou is None:
        kou = KouJumpParams()

    base = compute_metrics_for_firm_date(
        ticker=ticker,
        as_of_date=as_of_date,
        maturity_years=maturity_years,
        rf_df=rf_df,
        lookback_days=lookback_days,
    )
    if "error" in base:
        return base

    A0 = float(base["asset_value_proxy"])
    D = float(base["debt_face_value"])
    mu = float(base["mu_physical"])
    sigma = float(base["sigma_asset_proxy"])
    r = float(base["risk_free_rate"])

    # Physical PD via exact CF inversion
    phys = kou_structural_pd_cf(
        A0, D, mu, sigma, maturity_years,
        kou.lam, kou.kappa, kou.eta,
        measure="physical",
    )
    # Risk-neutral PD via exact CF inversion
    risk_n = kou_structural_pd_cf(
        A0, D, mu, sigma, maturity_years,
        kou.lam, kou.kappa, kou.eta,
        measure="risk_neutral",
        risk_free_rate=r,
    )

    pd_phys = float(np.clip(phys["PD_Kou"], 0.0, 1.0))
    pd_rn = float(np.clip(risk_n["PD_Kou"], 0.0, 1.0))

    # Pseudo-DD: equivalent normal distance, matches Merton DD when λ=0
    dd_kou = float(-norm.ppf(max(pd_phys, 1e-15)))

    out = {**base}
    out["Distance_to_Default_Kou"] = dd_kou
    out["PD_Kou_physical"] = pd_phys
    out["PD_Kou_risk_neutral"] = pd_rn
    out["Kou_lambda"] = kou.lam
    out["Kou_eta"] = kou.eta
    out["Kou_kappa"] = kou.kappa
    out["Kou_lambda_T"] = float(kou.lam * maturity_years)
    return out


def run_kou_metrics_pipeline(
    firm_dates,
    risk_free_csv_path: str | Path = DEFAULT_RF_PATH,
    kou: KouJumpParams | None = None,
    lookback_days: int = 252,
    output_csv_path: str | Path | None = None,
):
    """Same inputs as ``run_metrics_pipeline`` but adds Kou PD columns."""
    rf_df = load_risk_free_series(risk_free_csv_path)
    rows: list[dict] = []
    for item in firm_dates:
        try:
            row = compute_kou_metrics_for_firm_date(
                ticker=item.ticker,
                as_of_date=item.as_of_date,
                maturity_years=item.maturity_years,
                rf_df=rf_df,
                kou=kou,
                lookback_days=lookback_days,
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "ticker": item.ticker,
                    "as_of_date": item.as_of_date,
                    "maturity_years": item.maturity_years,
                    "error": str(exc),
                }
            )
    out = pd.DataFrame(rows)
    if output_csv_path is not None:
        out.to_csv(output_csv_path, index=False)
    return out


if __name__ == "__main__":
    # Example: same universe as merton_default_pipeline.__main__
    tickers = ["AAPL", "JPM", "XOM", "T", "F", "UAL", "AMZN", "KO"]
    inputs = [
        FirmDateInput(ticker=t, as_of_date="2025-12-31", maturity_years=T)
        for t in tickers
        for T in (1.0, 3.0, 5.0)
    ]
    out_path = (
        Path(__file__).resolve().parents[2] / "results" / "tables" / "kou_pd_table.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = run_kou_metrics_pipeline(
        firm_dates=inputs,
        kou=KouJumpParams(lam=0.05, eta=0.12, kappa=0.0),
        output_csv_path=out_path,
    )
    print(f"Saved: {out_path}")
    cols = [
        c
        for c in [
            "ticker",
            "as_of_date",
            "maturity_years",
            "PD_physical",
            "PD_risk_neutral",
            "PD_Kou_physical",
            "PD_Kou_risk_neutral",
            "Kou_lambda_dt",
        ]
        if c in df.columns
    ]
    print(df[cols].head(12).to_string(index=False))
