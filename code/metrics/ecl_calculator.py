"""Expected Credit Loss (ECL) calculation helpers.

ECL = PD × LGD × EAD

Components:
- EAD (Exposure at Default): Defaults to debt face value D
- LGD (Loss Given Default): 1 - Recovery Rate
- PD (Probability of Default): From structural models

Recovery rate assumptions:
- Investment grade firms: 40% recovery → LGD = 60%
- High-risk firms: 25% recovery → LGD = 75%
"""

from __future__ import annotations

import numpy as np


# Recovery rate constants (market averages)
RECOVERY_RATE_INVESTMENT_GRADE = 0.40  # 40% recovery
RECOVERY_RATE_HIGH_RISK = 0.25  # 25% recovery

# Ticker classifications
INVESTMENT_GRADE_TICKERS = {"AAPL", "JPM", "XOM", "KO", "AMZN", "ORCL"}
HIGH_RISK_TICKERS = {"T", "F", "UAL", "GME", "BBBY"}


def _safe_float(value: object) -> float:
    """Convert to float and check finiteness."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("Value is not finite.")
    return out


def ead_face_value(debt_face_value: float) -> float:
    """Exposure at Default (EAD) = Face value of debt.
    
    In the zero-coupon Merton setup, EAD equals the debt face value D.
    
    Parameters
    ----------
    debt_face_value : D
    
    Returns
    -------
    EAD : Exposure at default
    """
    return _safe_float(debt_face_value)


def lgd_constant(recovery_rate: float) -> float:
    """Loss Given Default from a constant recovery rate.
    
    LGD = 1 - recovery_rate
    
    Parameters
    ----------
    recovery_rate : Recovery rate in [0, 1]
    
    Returns
    -------
    LGD : Loss given default
    """
    rr = float(recovery_rate)
    if not (0.0 <= rr <= 1.0):
        raise ValueError("recovery_rate must be in [0, 1].")
    return float(1.0 - rr)


def get_recovery_rate(ticker: str) -> float:
    """Get recovery rate for a ticker based on classification.
    
    Parameters
    ----------
    ticker : Stock ticker symbol
    
    Returns
    -------
    recovery_rate : 0.40 for investment grade, 0.25 for high-risk
    """
    ticker_upper = ticker.upper()
    if ticker_upper in INVESTMENT_GRADE_TICKERS:
        return RECOVERY_RATE_INVESTMENT_GRADE
    elif ticker_upper in HIGH_RISK_TICKERS:
        return RECOVERY_RATE_HIGH_RISK
    else:
        # Default to investment grade if unknown
        return RECOVERY_RATE_INVESTMENT_GRADE


def get_lgd_constant(ticker: str) -> float:
    """Get constant LGD for a ticker.
    
    Returns
    -------
    LGD : 0.60 for investment grade, 0.75 for high-risk
    """
    rr = get_recovery_rate(ticker)
    return lgd_constant(rr)


def ecl(pd: float, lgd: float, ead: float) -> float:
    """Expected Credit Loss (undiscounted).
    
    ECL = PD × LGD × EAD
    
    Parameters
    ----------
    pd : Probability of default [0, 1]
    lgd : Loss given default [0, 1]
    ead : Exposure at default (currency)
    
    Returns
    -------
    ECL : Expected credit loss (currency)
    """
    pd_val = float(pd)
    lgd_val = float(lgd)
    ead_val = float(ead)
    
    if not (0.0 <= pd_val <= 1.0):
        raise ValueError("pd must be in [0, 1].")
    if not (0.0 <= lgd_val <= 1.0):
        raise ValueError("lgd must be in [0, 1].")
    if ead_val < 0:
        raise ValueError("ead must be non-negative.")
    
    # Guard against tiny PD to avoid numerical issues
    if pd_val < 1e-15:
        return 0.0
    
    return _safe_float(pd_val * lgd_val * ead_val)


def ecl_pv(pd: float, lgd: float, ead: float, r: float, T: float) -> float:
    """Present value of Expected Credit Loss.
    
    PV(ECL) = exp(-rT) × PD × LGD × EAD
    
    When used with risk-neutral PD and LGD, this equals the structural
    put option value (up to model consistency).
    
    Parameters
    ----------
    pd : Probability of default [0, 1]
    lgd : Loss given default [0, 1]
    ead : Exposure at default (currency)
    r : Risk-free rate (annualized)
    T : Time to maturity (years)
    
    Returns
    -------
    PV_ECL : Present value of expected credit loss (currency)
    """
    pd_val = float(pd)
    lgd_val = float(lgd)
    ead_val = float(ead)
    r_val = float(r)
    T_val = float(T)
    
    if not (0.0 <= pd_val <= 1.0):
        raise ValueError("pd must be in [0, 1].")
    if not (0.0 <= lgd_val <= 1.0):
        raise ValueError("lgd must be in [0, 1].")
    if ead_val < 0:
        raise ValueError("ead must be non-negative.")
    if T_val < 0:
        raise ValueError("T must be non-negative.")
    
    if pd_val < 1e-15:
        return 0.0
    
    discount = float(np.exp(-r_val * T_val))
    return _safe_float(discount * pd_val * lgd_val * ead_val)


def ecl_summary(
    ticker: str,
    pd_physical: float,
    pd_risk_neutral: float,
    debt_face_value: float,
    r: float,
    T: float,
    *,
    lgd_model_physical: float | None = None,
    lgd_model_risk_neutral: float | None = None,
) -> dict[str, float]:
    """Compute comprehensive ECL summary for a firm.
    
    Computes both constant-LGD and model-implied LGD ECL values.
    
    Returns
    -------
    dict with keys:
        EAD
        LGD_constant
        recovery_rate
        ECL_const_physical
        ECL_const_risk_neutral
        PV_ECL_const_risk_neutral
        [if lgd_model provided:]
        LGD_model_physical
        LGD_model_risk_neutral
        ECL_model_physical
        ECL_model_risk_neutral
        PV_ECL_model_risk_neutral
    """
    ead_val = ead_face_value(debt_face_value)
    rr = get_recovery_rate(ticker)
    lgd_const = lgd_constant(rr)
    
    result = {
        "EAD": ead_val,
        "LGD_constant": lgd_const,
        "recovery_rate": rr,
        "ECL_const_physical": ecl(pd_physical, lgd_const, ead_val),
        "ECL_const_risk_neutral": ecl(pd_risk_neutral, lgd_const, ead_val),
        "PV_ECL_const_risk_neutral": ecl_pv(pd_risk_neutral, lgd_const, ead_val, r, T),
    }
    
    # Add model-implied LGD metrics if provided
    if lgd_model_physical is not None:
        result["LGD_model_physical"] = lgd_model_physical
        result["ECL_model_physical"] = ecl(pd_physical, lgd_model_physical, ead_val)
    
    if lgd_model_risk_neutral is not None:
        result["LGD_model_risk_neutral"] = lgd_model_risk_neutral
        result["ECL_model_risk_neutral"] = ecl(pd_risk_neutral, lgd_model_risk_neutral, ead_val)
        result["PV_ECL_model_risk_neutral"] = ecl_pv(
            pd_risk_neutral, lgd_model_risk_neutral, ead_val, r, T
        )
    
    return result


if __name__ == "__main__":
    # Example
    ticker = "AAPL"
    D = 1.0e11
    pd_p = 1e-6
    pd_q = 1e-5
    r = 0.04
    T = 1.0
    
    summary = ecl_summary(ticker, pd_p, pd_q, D, r, T)
    print(f"ECL Summary for {ticker}:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")
    
    print("\nClassifications:")
    for t in ["AAPL", "JPM", "F", "UAL", "UNKNOWN"]:
        rr = get_recovery_rate(t)
        lgd = get_lgd_constant(t)
        print(f"  {t}: RR={rr:.0%}, LGD={lgd:.0%}")
