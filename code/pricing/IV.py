# Implied Volatility for the ticker

import yfinance as yf
import numpy as np
import pandas as pd

def implied_volatility(ticker, start_date = "2025-12-31", end_date = "2026-01-01"):
    tick = yf.Ticker(ticker)
    spot = tick.history(start=start_date, end=end_date)["Close"].iloc[0]
    expiry = tick.options[0]
    opt = tick.option_chain(expiry)
    calls = opt.calls.copy()
    puts = opt.puts.copy()
    calls["dist"] = (calls["strike"] - spot).abs()
    puts["dist"] = (puts["strike"] - spot).abs()
    atm_call = calls.loc[calls["dist"].idxmin()]
    atm_put = puts.loc[puts["dist"].idxmin()]
    # print(atm_call)
    # print(atm_put)
    iv_call = atm_call["impliedVolatility"]
    iv_put = atm_put["impliedVolatility"]
    iv_atm = np.nanmean([iv_call, iv_put])
    call_strike = atm_call["strike"]
    put_strike = atm_put["strike"]
    return iv_call, iv_put, iv_atm, call_strike, put_strike, spot, expiry
