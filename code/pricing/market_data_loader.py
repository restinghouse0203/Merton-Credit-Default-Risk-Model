"""Market data loader for Merton structural credit risk pricing.

Generalizes the inline data-loading logic from ``pricing.ipynb`` so that any
ticker can be passed in. Falls back to ``yfinance`` when local CSV snapshots
are not available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from IV import implied_volatility


DATA_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_RISK_FREE_DATE = "2025-12-31"


@dataclass
class MertonInputs:
    """Inputs required to run Merton's structural credit-risk pricing.

    Attributes:
        ticker: Equity ticker symbol.
        spot: Per-share equity price.
        shares_outstanding: Common shares outstanding.
        total_debt: Face value of debt (used as strike K).
        sigma: Asset/equity volatility proxy (ATM implied volatility).
        risk_free_rate: Annualised continuously-compounded short rate.
        asset_value: ``shares_outstanding * spot + total_debt`` (Merton S).
        ref_date: Reference date used for spot / risk-free rate snapshot.
    """

    ticker: str
    spot: float
    shares_outstanding: float
    total_debt: float
    sigma: float
    risk_free_rate: float
    asset_value: float
    ref_date: str

    @property
    def S(self) -> float:
        return self.asset_value

    @property
    def K(self) -> float:
        return self.total_debt

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "spot": self.spot,
            "shares_outstanding": self.shares_outstanding,
            "total_debt": self.total_debt,
            "sigma": self.sigma,
            "risk_free_rate": self.risk_free_rate,
            "asset_value": self.asset_value,
            "ref_date": self.ref_date,
        }


def load_risk_free_rate(
    ref_date: str = DEFAULT_RISK_FREE_DATE,
    data_dir: str = DATA_DIR_DEFAULT,
    csv_name: str = "US_treasury_yield3m.csv",
) -> float:
    """Read continuously-compounded risk-free rate from local treasury CSV.

    The CSV is expected to follow the same layout used in ``pricing.ipynb``:
    a header row that we skip, then ``Date, yield_pct`` columns.
    """
    path = os.path.join(data_dir, csv_name)
    df = pd.read_csv(
        path,
        skiprows=[0],
        names=["Date", "US_treasury_yield3m"],
    )
    matches = df.loc[df["Date"] == ref_date, "US_treasury_yield3m"]
    if matches.empty:
        # Fall back to most recent observation on or before ref_date.
        df["Date"] = pd.to_datetime(df["Date"])
        ref_dt = pd.to_datetime(ref_date)
        prior = df.loc[df["Date"] <= ref_dt]
        if prior.empty:
            raise ValueError(
                f"No risk-free rate observation found on or before {ref_date}"
            )
        rate_pct = float(prior.iloc[-1]["US_treasury_yield3m"])
    else:
        rate_pct = float(matches.iloc[0])
    return rate_pct / 100.0


def load_spot_price(
    ticker: str,
    ref_date: str = DEFAULT_RISK_FREE_DATE,
    data_dir: str = DATA_DIR_DEFAULT,
) -> float:
    """Read the most recent adjusted-close price from a local CSV.

    Falls back to ``yfinance`` when the local CSV does not exist. The CSV
    layout matches the snapshots stored under ``data/{TICKER}_stock_price.csv``.
    """
    candidates = [
        os.path.join(data_dir, f"{ticker.upper()}_stock_price.csv"),
        os.path.join(data_dir, f"{ticker.lower()}_stock_price.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            stock_df = pd.read_csv(
                path,
                skiprows=[0, 1, 2],
                names=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
            )
            stock_df["Date"] = pd.to_datetime(stock_df["Date"])
            ref_dt = pd.to_datetime(ref_date)
            prior = stock_df.loc[stock_df["Date"] <= ref_dt]
            if prior.empty:
                # Use most recent row available.
                return float(stock_df.iloc[0]["Adj Close"])
            return float(prior.sort_values("Date").iloc[-1]["Adj Close"])

    return _yfinance_spot_price(ticker, ref_date)


def _yfinance_spot_price(ticker: str, ref_date: str) -> float:
    end_dt = pd.to_datetime(ref_date) + pd.Timedelta(days=4)
    start_dt = pd.to_datetime(ref_date) - pd.Timedelta(days=10)
    try:
        hist = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise ValueError(f"yfinance download failed for {ticker}: {e}")
    
    if hist is None or hist.empty:
        raise ValueError(f"No yfinance price data available for {ticker} near {ref_date}")
    
    # Handle MultiIndex columns that yfinance sometimes returns
    if isinstance(hist.columns, pd.MultiIndex):
        # Try to get ('Adj Close', ticker)
        try:
            series = hist[("Adj Close", ticker)]
        except KeyError:
            try:
                series = hist[("Close", ticker)]
            except KeyError:
                raise ValueError(f"No Adj Close or Close columns for {ticker} in MultiIndex")
    else:
        # Simple columns
        if "Adj Close" in hist.columns:
            series = hist["Adj Close"]
        elif "Close" in hist.columns:
            series = hist["Close"]
        else:
            raise ValueError(f"No price columns in yfinance data for {ticker}")
    
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No usable Adj Close prices for {ticker} near {ref_date}")
    
    # Extract the scalar value safely
    last_val = float(series.iloc[-1])
    return last_val


def load_balance_sheet(ticker: str) -> tuple[float, float]:
    """Return ``(shares_outstanding, total_debt)`` from yfinance balance sheet.

    Mirrors the candidate-row search from the original notebook so we
    gracefully handle different yfinance label conventions.
    """
    bs = yf.Ticker(ticker).balance_sheet
    if bs is None or bs.empty:
        raise ValueError(f"yfinance returned no balance sheet for {ticker}")

    latest_col = bs.columns[0]
    shares_candidates = [
        "Ordinary Shares Number",
        "Share Issued",
        "Ordinary Shares Outstanding",
    ]
    debt_candidates = [
        "Total Debt",
        "Total Liabilities Net Minority Interest",
    ]

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
        shares_outstanding = yf.Ticker(ticker).info.get("sharesOutstanding", np.nan)

    if pd.isna(shares_outstanding) or pd.isna(total_debt):
        raise ValueError(
            f"Could not retrieve shares/debt for {ticker} from balance sheet."
        )

    return float(shares_outstanding), float(total_debt)


def get_implied_volatility(ticker: str) -> float:
    """ATM implied volatility (mean of call/put IVs) via ``IV.implied_volatility``."""
    _, _, iv_atm, *_ = implied_volatility(ticker)
    if iv_atm is None or np.isnan(iv_atm):
        raise ValueError(f"Implied volatility unavailable for {ticker}")
    return float(iv_atm)


def compute_merton_inputs(
    ticker: str,
    ref_date: str = DEFAULT_RISK_FREE_DATE,
    data_dir: str = DATA_DIR_DEFAULT,
    sigma_override: float | None = None,
    spot_override: float | None = None,
    shares_override: float | None = None,
    debt_override: float | None = None,
) -> MertonInputs:
    """Bundle together every input needed for Merton structural pricing.

    Override arguments are useful for special cases (e.g. distressed names,
    historical snapshots like Credit Suisse pre-2023) where the standard
    yfinance pull is unavailable or stale.
    """
    r = load_risk_free_rate(ref_date=ref_date, data_dir=data_dir)

    spot = spot_override if spot_override is not None else load_spot_price(
        ticker, ref_date=ref_date, data_dir=data_dir
    )

    if shares_override is not None and debt_override is not None:
        shares_outstanding, total_debt = float(shares_override), float(debt_override)
    else:
        shares_outstanding, total_debt = load_balance_sheet(ticker)
        if shares_override is not None:
            shares_outstanding = float(shares_override)
        if debt_override is not None:
            total_debt = float(debt_override)

    sigma = sigma_override if sigma_override is not None else get_implied_volatility(ticker)

    asset_value = shares_outstanding * spot + total_debt

    return MertonInputs(
        ticker=ticker,
        spot=spot,
        shares_outstanding=shares_outstanding,
        total_debt=total_debt,
        sigma=sigma,
        risk_free_rate=r,
        asset_value=asset_value,
        ref_date=ref_date,
    )


def compute_merton_inputs_batch(
    tickers: Iterable[str],
    ref_date: str = DEFAULT_RISK_FREE_DATE,
    data_dir: str = DATA_DIR_DEFAULT,
    overrides: dict | None = None,
    skip_on_error: bool = True,
) -> dict[str, MertonInputs]:
    """Compute Merton inputs for many tickers, optionally skipping failures.

    ``overrides`` maps ``ticker`` -> kwargs forwarded to
    :func:`compute_merton_inputs` (e.g. ``{"CS": {"sigma_override": 1.2}}``).
    """
    overrides = overrides or {}
    out: dict[str, MertonInputs] = {}
    for ticker in tickers:
        kwargs = overrides.get(ticker, {})
        try:
            out[ticker] = compute_merton_inputs(
                ticker, ref_date=ref_date, data_dir=data_dir, **kwargs
            )
        except Exception as exc:
            if not skip_on_error:
                raise
            print(f"[market_data_loader] Skipping {ticker}: {exc}")
    return out


def historical_spot_from_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> float:
    """Return the last adjusted-close price within [start, end) from yfinance."""
    try:
        hist = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise ValueError(f"yfinance download failed for {ticker} between {start} and {end}: {e}")
    
    if hist is None or hist.empty:
        raise ValueError(f"No yfinance prices for {ticker} between {start} and {end}")
    
    # Handle MultiIndex columns that yfinance sometimes returns
    if isinstance(hist.columns, pd.MultiIndex):
        # Try to get ('Adj Close', ticker)
        try:
            series = hist[("Adj Close", ticker)]
        except KeyError:
            try:
                series = hist[("Close", ticker)]
            except KeyError:
                raise ValueError(f"No Adj Close or Close columns for {ticker} in MultiIndex")
    else:
        # Simple columns
        if "Adj Close" in hist.columns:
            series = hist["Adj Close"]
        elif "Close" in hist.columns:
            series = hist["Close"]
        else:
            raise ValueError(f"No price columns in yfinance data for {ticker}")
    
    series = series.dropna()
    if series.empty:
        raise ValueError(f"All Adj Close prices NaN for {ticker} between {start} and {end}")
    
    # Extract the scalar value safely
    last_val = float(series.iloc[-1])
    return last_val
