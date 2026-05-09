"""Unified Merton-framework pricing across BSM, Heston, Merton-Jump and Kou-Jump.

Each ``price_merton_*`` function returns a one-row DataFrame with columns
``equity_{T}y`` and ``debt_{T}y`` for every requested maturity. This makes it
trivial to concatenate per-ticker per-model results in the notebook.

Merton framework
----------------
Asset value (Merton ``S``)  = shares_outstanding * spot + total_debt
Strike      (Merton ``K``)  = total_debt
Equity      = European Call(S, K, T, r, sigma)
Debt        = K * exp(-rT) - European Put(S, K, T, r, sigma)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from bs_pricing import black_scholes_call, black_scholes_put
from kou_jump_pricing import kou_jump_call_fourier, kou_jump_put_fourier
from market_data_loader import MertonInputs, compute_merton_inputs
from merton_jump_pricing import (
    merton_jump_pricing_call,
    merton_jump_pricing_put,
)

try:  # Heston Monte Carlo is optional during import-time so static analyzers don't fail.
    from Heston_EulerAndMilstein import (
        GeneratePathsHestonEuler,
    )
except Exception:  # pragma: no cover
    GeneratePathsHestonEuler = None  # type: ignore[assignment]


DEFAULT_MATURITIES: tuple[int, ...] = (1, 3, 5)


# ---------------------------------------------------------------------------
# Default model parameters (literature-standard for demonstrations).
# ---------------------------------------------------------------------------


@dataclass
class HestonParams:
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.30
    rho: float = -0.7
    n_paths: int = 20_000
    n_steps_per_year: int = 252
    seed: int = 42

    def with_atm_iv(self, sigma: float) -> "HestonParams":
        """Anchor the long-run variance to the equity ATM IV.

        Without anchoring, the same constant Heston parameters would ignore the
        ticker-specific volatility level entirely. We re-use the structural
        kappa/xi/rho from literature but recenter v0 and theta to ``sigma**2``.
        """
        var = float(sigma) ** 2
        return HestonParams(
            v0=var,
            kappa=self.kappa,
            theta=var,
            xi=self.xi,
            rho=self.rho,
            n_paths=self.n_paths,
            n_steps_per_year=self.n_steps_per_year,
            seed=self.seed,
        )


@dataclass
class MertonJumpParams:
    Lambda: float = 1.0
    muJ: float = -0.05
    sigmaJ: float = 0.10
    n_terms: int = 50


@dataclass
class KouJumpParams:
    lam: float = 2.0
    p: float = 0.4
    eta1: float = 10.0
    eta2: float = 5.0
    u_max: float = 200.0


@dataclass
class ModelParamSet:
    heston: HestonParams = field(default_factory=HestonParams)
    merton_jump: MertonJumpParams = field(default_factory=MertonJumpParams)
    kou_jump: KouJumpParams = field(default_factory=KouJumpParams)


# ---------------------------------------------------------------------------
# Per-model pricers. Each accepts ``MertonInputs`` for one ticker.
# ---------------------------------------------------------------------------


def _empty_row(ticker: str, maturities: Sequence[int]) -> dict:
    row: dict = {"ticker": ticker}
    for T in maturities:
        row[f"equity_{T}y"] = np.nan
        row[f"debt_{T}y"] = np.nan
    return row


def price_merton_bsm(
    inputs: MertonInputs,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
) -> pd.DataFrame:
    """Black-Scholes-Merton structural pricing.

    Equity from ``black_scholes_call``; debt via call-put parity.
    """
    S, K, r, sigma = inputs.S, inputs.K, inputs.risk_free_rate, inputs.sigma
    row = {"ticker": inputs.ticker}
    for T in maturities:
        equity = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        debt = K * math.exp(-r * T) - put
        row[f"equity_{T}y"] = float(equity)
        row[f"debt_{T}y"] = float(debt)
    return pd.DataFrame([row])


def price_merton_jump(
    inputs: MertonInputs,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    params: MertonJumpParams | None = None,
) -> pd.DataFrame:
    """Merton (1976) jump-diffusion structural pricing using closed-form series."""
    p = params or MertonJumpParams()
    S, K, r, sigma = inputs.S, inputs.K, inputs.risk_free_rate, inputs.sigma
    row = {"ticker": inputs.ticker}
    for T in maturities:
        call = merton_jump_pricing_call(
            S, K, T, r, sigma, p.Lambda, p.muJ, p.sigmaJ, p.n_terms
        )
        put = merton_jump_pricing_put(
            S, K, T, r, sigma, p.Lambda, p.muJ, p.sigmaJ, p.n_terms
        )
        debt = K * math.exp(-r * T) - put
        row[f"equity_{T}y"] = float(call)
        row[f"debt_{T}y"] = float(debt)
    return pd.DataFrame([row])


def price_merton_kou(
    inputs: MertonInputs,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    params: KouJumpParams | None = None,
) -> pd.DataFrame:
    """Kou (2002) double-exponential jump-diffusion via Fourier inversion."""
    p = params or KouJumpParams()
    S, K, r, sigma = inputs.S, inputs.K, inputs.risk_free_rate, inputs.sigma
    row = {"ticker": inputs.ticker}
    for T in maturities:
        call = kou_jump_call_fourier(
            S, K, T, r, sigma, p.lam, p.p, p.eta1, p.eta2, u_max=p.u_max
        )
        put = kou_jump_put_fourier(
            S, K, T, r, sigma, p.lam, p.p, p.eta1, p.eta2, u_max=p.u_max
        )
        debt = K * math.exp(-r * T) - put
        row[f"equity_{T}y"] = float(call)
        row[f"debt_{T}y"] = float(debt)
    return pd.DataFrame([row])


def price_merton_heston(
    inputs: MertonInputs,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    params: HestonParams | None = None,
) -> pd.DataFrame:
    """Heston stochastic-volatility structural pricing via Monte Carlo.

    Uses the same Euler full-truncation scheme as ``GeneratePathsHestonEuler``.
    Long-run variance is anchored to ATM IV through :meth:`HestonParams.with_atm_iv`.
    """
    if GeneratePathsHestonEuler is None:
        raise ImportError("Heston_EulerAndMilstein.GeneratePathsHestonEuler not available")

    base = params or HestonParams()
    p = base.with_atm_iv(inputs.sigma)

    S, K, r = inputs.S, inputs.K, inputs.risk_free_rate
    row = {"ticker": inputs.ticker}

    for T in maturities:
        np.random.seed(p.seed)
        paths = GeneratePathsHestonEuler(
            NoOfPaths=p.n_paths,
            NoOfSteps=int(p.n_steps_per_year * T),
            T=T,
            r=r,
            S_0=S,
            v0=p.v0,
            kappa=p.kappa,
            theta=p.theta,
            xi=p.xi,
            rho=p.rho,
        )
        A_T = paths["S"][:, -1]
        equity = math.exp(-r * T) * float(np.mean(np.maximum(A_T - K, 0.0)))
        debt = math.exp(-r * T) * float(np.mean(np.minimum(A_T, K)))
        row[f"equity_{T}y"] = equity
        row[f"debt_{T}y"] = debt
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Convenience: price all four models for a single ticker / batch of tickers.
# ---------------------------------------------------------------------------


MODEL_PRICERS = {
    "BSM": price_merton_bsm,
    "Heston": price_merton_heston,
    "Merton-Jump": price_merton_jump,
    "Kou-Jump": price_merton_kou,
}


def price_all_models(
    inputs: MertonInputs,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    param_set: ModelParamSet | None = None,
) -> pd.DataFrame:
    """Price one ticker under every supported model.

    Returns a DataFrame with one row per (ticker, model) pair.
    """
    p = param_set or ModelParamSet()
    rows = []
    for model_name, pricer in MODEL_PRICERS.items():
        if model_name == "BSM":
            df = pricer(inputs, maturities=maturities)
        elif model_name == "Heston":
            df = pricer(inputs, maturities=maturities, params=p.heston)
        elif model_name == "Merton-Jump":
            df = pricer(inputs, maturities=maturities, params=p.merton_jump)
        elif model_name == "Kou-Jump":
            df = pricer(inputs, maturities=maturities, params=p.kou_jump)
        else:  # pragma: no cover
            continue
        df = df.copy()
        df.insert(1, "model", model_name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def price_all_tickers(
    inputs_by_ticker: dict[str, MertonInputs],
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    param_set: ModelParamSet | None = None,
) -> pd.DataFrame:
    """Price every ticker under every model.

    Returns a long-form DataFrame indexed by ``(ticker, model)``.
    """
    frames = []
    for ticker, inputs in inputs_by_ticker.items():
        try:
            frames.append(price_all_models(inputs, maturities=maturities, param_set=param_set))
        except Exception as exc:
            print(f"[credit_pricing_models] Pricing failed for {ticker}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["ticker", "model"])
    return pd.concat(frames, ignore_index=True)


def to_wide_table(long_df: pd.DataFrame, maturities: Sequence[int] = DEFAULT_MATURITIES) -> pd.DataFrame:
    """Pivot long-form (ticker, model) results into a wide comparison table.

    Output columns are ``equity_{T}y_{model}`` and ``debt_{T}y_{model}``.
    """
    if long_df.empty:
        return long_df

    pivots = []
    value_cols = [f"equity_{T}y" for T in maturities] + [f"debt_{T}y" for T in maturities]
    for col in value_cols:
        wide = long_df.pivot(index="ticker", columns="model", values=col)
        wide.columns = [f"{col}_{m}" for m in wide.columns]
        pivots.append(wide)
    return pd.concat(pivots, axis=1).reset_index()


def credit_spread_table(
    long_df: pd.DataFrame,
    risk_free_rate: float,
    face_value_by_ticker: dict[str, float],
    maturities: Sequence[int] = DEFAULT_MATURITIES,
) -> pd.DataFrame:
    """Implied credit spread = -ln(D / (F * e^{-rT})) / T, per ticker/model/maturity.

    ``D`` is the model's risky debt price and ``F`` is its face value.
    Returns a long-form table with column ``credit_spread_{T}y``.
    """
    out = long_df[["ticker", "model"]].copy()
    for T in maturities:
        col = f"debt_{T}y"
        if col not in long_df.columns:
            continue
        spreads = []
        for _, row in long_df.iterrows():
            face = face_value_by_ticker.get(row["ticker"])
            debt = row[col]
            if face is None or face <= 0 or debt <= 0:
                spreads.append(np.nan)
                continue
            risk_free_pv = face * math.exp(-risk_free_rate * T)
            ratio = debt / risk_free_pv
            if ratio <= 0:
                spreads.append(np.nan)
            else:
                spreads.append(-math.log(ratio) / T)
        out[f"credit_spread_{T}y"] = spreads
    return out
