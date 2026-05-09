"""Pipeline to compute Merton-style default metrics per firm-date.

Part 3 metrics implemented here:
- PD_physical
- PD_risk_neutral
- Distance_to_Default
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


DEFAULT_RF_PATH = Path(__file__).resolve().parents[2] / "data" / "US_treasury_yield3m.csv"


@dataclass(frozen=True)
class FirmDateInput:
    ticker: str
    as_of_date: str
    maturity_years: float = 1.0


def _safe_float(value: object) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("Value is not finite.")
    return out


def _get_price_series(hist: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """Extract a 1D price series even when yfinance returns MultiIndex columns."""
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker}.")

    series: pd.Series | pd.DataFrame
    if isinstance(hist.columns, pd.MultiIndex):
        # yfinance may return columns like ('Close', 'AAPL')
        if field in hist.columns.get_level_values(0):
            series = hist.xs(field, axis=1, level=0)
        else:
            raise ValueError(f"{field} not found in yfinance history for {ticker}.")
    else:
        if field not in hist.columns:
            raise ValueError(f"{field} not found in yfinance history for {ticker}.")
        series = hist[field]

    if isinstance(series, pd.DataFrame):
        # If multiple columns remain, use matching ticker when present, otherwise first.
        if ticker in series.columns:
            series = series[ticker]
        else:
            series = series.iloc[:, 0]

    return series.dropna()


def load_risk_free_series(csv_path: str | Path = DEFAULT_RF_PATH) -> pd.DataFrame:
    """Load 3M Treasury series used as risk-free proxy."""
    rf = pd.read_csv(csv_path, skiprows=[0], names=["Date", "US_treasury_yield3m"])
    rf["Date"] = pd.to_datetime(rf["Date"])
    rf["US_treasury_yield3m"] = pd.to_numeric(rf["US_treasury_yield3m"], errors="coerce")
    rf = rf.dropna(subset=["Date", "US_treasury_yield3m"]).sort_values("Date")
    return rf


def risk_free_rate_for_date(as_of_date: str, rf_df: pd.DataFrame) -> float:
    """Get nearest available treasury yield on or before as_of_date."""
    date = pd.to_datetime(as_of_date)
    subset = rf_df.loc[rf_df["Date"] <= date]
    if subset.empty:
        raise ValueError(f"No risk-free observation on or before {as_of_date}.")
    return _safe_float(subset.iloc[-1]["US_treasury_yield3m"]) / 100.0


def spot_price_on_or_before(ticker: str, as_of_date: str, buffer_days: int = 5) -> float:
    """Fetch latest close close to as_of_date from yfinance."""
    end = pd.to_datetime(as_of_date)
    start = end - pd.Timedelta(days=buffer_days)
    hist = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    close = _get_price_series(hist=hist, field="Close", ticker=ticker)
    if close.empty:
        raise ValueError(f"Close column empty for {ticker} around {as_of_date}.")
    return _safe_float(close.iloc[-1])


def latest_balance_sheet_values(ticker: str) -> tuple[float, float]:
    """Return shares outstanding and debt proxy from latest balance sheet."""
    yft = yf.Ticker(ticker)
    bs = yft.balance_sheet
    if bs.empty:
        raise ValueError(f"Empty balance sheet for {ticker}.")

    latest_col = bs.columns[0]
    shares_candidates = ["Ordinary Shares Number", "Share Issued", "Ordinary Shares Outstanding"]
    debt_candidates = ["Total Debt", "Total Liabilities Net Minority Interest"]

    shares_outstanding = np.nan
    total_debt = np.nan

    for row_name in shares_candidates:
        if row_name in bs.index:
            shares_outstanding = bs.loc[row_name, latest_col]
            break

    for row_name in debt_candidates:
        if row_name in bs.index:
            total_debt = bs.loc[row_name, latest_col]
            break

    if pd.isna(shares_outstanding):
        shares_outstanding = yft.info.get("sharesOutstanding", np.nan)

    if pd.isna(shares_outstanding) or pd.isna(total_debt):
        raise ValueError(f"Could not retrieve shares/debt for {ticker}.")

    return _safe_float(shares_outstanding), _safe_float(total_debt)


def estimate_equity_drift_and_vol(
    ticker: str,
    as_of_date: str,
    lookback_days: int = 252,
) -> tuple[float, float]:
    """Estimate annualized drift and vol from historical log returns."""
    end = pd.to_datetime(as_of_date)
    start = end - pd.Timedelta(days=int(np.ceil(lookback_days * 1.7)))
    hist = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    px = _get_price_series(hist=hist, field="Adj Close", ticker=ticker)
    rets = np.log(px / px.shift(1)).dropna()
    if len(rets) < 60:
        raise ValueError(f"Insufficient return history for {ticker} at {as_of_date}.")

    rets = rets.tail(lookback_days)
    mu = _safe_float(rets.mean() * 252.0)
    sigma = _safe_float(rets.std(ddof=1) * np.sqrt(252.0))
    if sigma <= 0.0:
        raise ValueError("Estimated volatility is non-positive.")
    return mu, sigma


def merton_pd_and_dd(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    mu_physical: float,
    r_risk_neutral: float,
) -> dict[str, float]:
    """Compute DD and default probabilities under P and Q for Merton model."""
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    sigma = _safe_float(sigma_asset)
    T = _safe_float(maturity_years)
    mu = _safe_float(mu_physical)
    r = _safe_float(r_risk_neutral)

    if A0 <= 0 or D <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("A0, D, sigma, and T must all be strictly positive.")

    denom = sigma * np.sqrt(T)
    dd_physical = (np.log(A0 / D) + (mu - 0.5 * sigma * sigma) * T) / denom
    d2_q = (np.log(A0 / D) + (r - 0.5 * sigma * sigma) * T) / denom

    return {
        "Distance_to_Default": _safe_float(dd_physical),
        "PD_physical": _safe_float(norm.cdf(-dd_physical)),
        "PD_risk_neutral": _safe_float(norm.cdf(-d2_q)),
    }


def merton_jump_pd_and_dd(
    asset_value: float,
    debt_face_value: float,
    sigma_asset: float,
    maturity_years: float,
    mu_physical: float,
    r_risk_neutral: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    n_terms: int = 50,
) -> dict[str, float]:
    """Merton jump-diffusion structural PD using the conditional lognormal sum.

    Under Merton's model, conditional on N_T = n jumps,

        log(A_T/A_0) | N_T=n  ~  N( (drift_n - ½σ_n²)T,  σ_n²T )

    where drift_n and σ_n are the n-jump adjusted parameters (same as the
    option pricing formula).  Averaging over N_T:

        Physical:      N_T ~ Poisson(λ·T),   drift = μ − λk
        Risk-neutral:  N_T ~ Poisson(λ'·T),  drift = r − λk,  λ' = λ(1+k)

    Parameters
    ----------
    jump_intensity : λ (annualised jump rate under the physical measure)
    jump_mean      : μ_J (mean of log-jump size, same as Merton's δ)
    jump_vol       : σ_J (std-dev of log-jump size)
    """
    A0 = _safe_float(asset_value)
    D = _safe_float(debt_face_value)
    sigma = _safe_float(sigma_asset)
    T = _safe_float(maturity_years)
    mu = _safe_float(mu_physical)
    r = _safe_float(r_risk_neutral)
    lam = float(jump_intensity)
    muJ = float(jump_mean)
    sigmaJ = float(jump_vol)

    if A0 <= 0 or D <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("A0, D, sigma, T must be strictly positive.")
    if lam < 0:
        raise ValueError("jump_intensity must be non-negative.")

    # k = E[J] - 1 = E[exp(Y)] - 1 where Y ~ N(muJ, sigmaJ^2)
    k = float(np.exp(muJ + 0.5 * sigmaJ ** 2) - 1.0)
    # gamma = log(1 + k) = muJ + 0.5*sigmaJ^2
    gamma = float(np.log(1.0 + k))
    lambda_prime = lam * (1.0 + k)   # risk-neutral intensity

    pd_phys = 0.0
    pd_rn = 0.0

    for n in range(n_terms):
        # Conditional vol (same under P and Q)
        sigma_n = float(np.sqrt(sigma ** 2 + n * sigmaJ ** 2 / T))
        sigma_n_sqrt_T = sigma_n * float(np.sqrt(T))
        if sigma_n_sqrt_T == 0.0:
            continue

        # ---- Physical measure ----
        # Poisson weight: P(N_T = n) = exp(-λT)(λT)^n / n!
        log_w_phys = -lam * T + n * np.log(lam * T + 1e-300) - math.lgamma(n + 1)
        w_phys = float(np.exp(log_w_phys))

        r_n_phys = mu - lam * k + n * gamma / T
        d2_n_phys = (np.log(A0 / D) + (r_n_phys - 0.5 * sigma_n ** 2) * T) / sigma_n_sqrt_T
        pd_phys += w_phys * float(norm.cdf(-d2_n_phys))

        # ---- Risk-neutral measure ----
        # Poisson weight: P(N_T = n) = exp(-λ'T)(λ'T)^n / n!
        log_w_rn = -lambda_prime * T + n * np.log(lambda_prime * T + 1e-300) - math.lgamma(n + 1)
        w_rn = float(np.exp(log_w_rn))

        r_n_rn = r - lam * k + n * gamma / T
        d2_n_rn = (np.log(A0 / D) + (r_n_rn - 0.5 * sigma_n ** 2) * T) / sigma_n_sqrt_T
        pd_rn += w_rn * float(norm.cdf(-d2_n_rn))

    pd_phys = float(np.clip(pd_phys, 0.0, 1.0))
    pd_rn = float(np.clip(pd_rn, 0.0, 1.0))

    # Pseudo-DD: equivalent standard-normal distance such that Φ(-DD) = PD_physical.
    # When λ=0 this reduces to the Merton DD exactly.
    dd = float(-norm.ppf(max(pd_phys, 1e-15)))

    return {
        "Distance_to_Default_Merton_Jump": _safe_float(dd),
        "PD_Merton_Jump_physical": _safe_float(pd_phys),
        "PD_Merton_Jump_risk_neutral": _safe_float(pd_rn),
    }


def compute_jump_metrics_for_firm_date(
    ticker: str,
    as_of_date: str,
    maturity_years: float,
    rf_df: pd.DataFrame,
    jump_intensity: float = 5.0,
    jump_mean: float = -0.05,
    jump_vol: float = 0.10,
    lookback_days: int = 252,
) -> dict[str, float | str]:
    """One row: Merton BS metrics + Merton Jump metrics."""
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

    jump_metrics = merton_jump_pd_and_dd(
        asset_value=A0,
        debt_face_value=D,
        sigma_asset=sigma,
        maturity_years=maturity_years,
        mu_physical=mu,
        r_risk_neutral=r,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_vol=jump_vol,
    )
    return {
        **base,
        **jump_metrics,
        "MertonJump_lambda": jump_intensity,
        "MertonJump_muJ": jump_mean,
        "MertonJump_sigmaJ": jump_vol,
    }


def run_merton_jump_pipeline(
    firm_dates: Sequence[FirmDateInput] | Iterable[FirmDateInput],
    risk_free_csv_path: str | Path = DEFAULT_RF_PATH,
    jump_intensity: float = 5.0,
    jump_mean: float = -0.05,
    jump_vol: float = 0.10,
    lookback_days: int = 252,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run Merton-BS + Merton-Jump pipeline for a list of firm-date entries."""
    rf_df = load_risk_free_series(risk_free_csv_path)
    rows: list[dict[str, float | str]] = []

    for item in firm_dates:
        try:
            row = compute_jump_metrics_for_firm_date(
                ticker=item.ticker,
                as_of_date=item.as_of_date,
                maturity_years=item.maturity_years,
                rf_df=rf_df,
                jump_intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_vol=jump_vol,
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


def compute_metrics_for_firm_date(
    ticker: str,
    as_of_date: str,
    maturity_years: float,
    rf_df: pd.DataFrame,
    lookback_days: int = 252,
) -> dict[str, float | str]:
    """Build one firm-date row for the empirical table."""
    shares, debt_face = latest_balance_sheet_values(ticker)
    spot = spot_price_on_or_before(ticker, as_of_date)
    mu, sigma = estimate_equity_drift_and_vol(ticker, as_of_date, lookback_days=lookback_days)
    r = risk_free_rate_for_date(as_of_date, rf_df)

    # Same setup used in pricing notebook: asset proxy = market cap + debt.
    asset_value = shares * spot + debt_face
    metrics = merton_pd_and_dd(
        asset_value=asset_value,
        debt_face_value=debt_face,
        sigma_asset=sigma,
        maturity_years=maturity_years,
        mu_physical=mu,
        r_risk_neutral=r,
    )
    return {
        "ticker": ticker,
        "as_of_date": pd.to_datetime(as_of_date).strftime("%Y-%m-%d"),
        "maturity_years": maturity_years,
        "asset_value_proxy": asset_value,
        "debt_face_value": debt_face,
        "mu_physical": mu,
        "sigma_asset_proxy": sigma,
        "risk_free_rate": r,
        **metrics,
    }


def run_metrics_pipeline(
    firm_dates: Sequence[FirmDateInput] | Iterable[FirmDateInput],
    risk_free_csv_path: str | Path = DEFAULT_RF_PATH,
    lookback_days: int = 252,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run pipeline for a list of firm-date entries."""
    rf_df = load_risk_free_series(risk_free_csv_path)
    rows: list[dict[str, float | str]] = []

    for item in firm_dates:
        try:
            row = compute_metrics_for_firm_date(
                ticker=item.ticker,
                as_of_date=item.as_of_date,
                maturity_years=item.maturity_years,
                rf_df=rf_df,
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
    # Example run aligned with the pricing notebook universe.
    tickers = ["AAPL", "JPM", "XOM", "T", "F", "UAL", "AMZN", "KO"]
    inputs = [
        FirmDateInput(ticker=t, as_of_date="2025-12-31", maturity_years=T)
        for t in tickers
        for T in (1.0, 3.0, 5.0)
    ]
    df_metrics = run_metrics_pipeline(
        firm_dates=inputs,
        output_csv_path=Path(__file__).resolve().parents[2] / "results" / "tables" / "merton_pd_dd_table.csv",
    )
    print(df_metrics.head())
