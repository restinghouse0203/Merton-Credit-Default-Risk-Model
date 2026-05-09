# Merton jump-diffusion closed-form call pricing
import math
import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def merton_jump_pricing_call(S, K, T, r, sigma, Lambda, muJ, sigmaJ, N):
    """
    Price a European call under Merton jump diffusion with a truncated series.
    Args:
        S: initial stock price
        K: strike price
        T: time to maturity
        r: risk-free rate
        sigma: asset volatility
        Lambda: jump intensity
        muJ: mean of jump size
        sigmaJ: jump size volatility (aka delta)
        N: number of terms to approximate the infinite series

    Closed-form relationship used:
      C = sum_{n=0}^{N-1} p_n * BS(S, K, T, r_n, sigma_n)
      p_n = exp(-lambda' T) * (lambda' T)^n / n!
      lambda' = Lambda * (1 + k),   k = E[e^J] - 1
      r_n = r - Lambda*k + n*log(1+k)/T
      sigma_n^2 = sigma^2 + n*delta^2/T

    Here k is computed from muJ and sigmaJ:
      k = E[e^J] - 1 = exp(muJ + 0.5*sigmaJ^2) - 1.
    """
    if T <= 0:
        return max(S - K, 0.0)
    if N <= 0:
        raise ValueError("N must be positive.")

    if muJ is None or sigmaJ is None:
        raise ValueError("muJ and sigmaJ must be provided.")
    k = np.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0

    lambda_prime = Lambda * (1.0 + k)
    gamma = np.log(1.0 + k)

    merton_jump_call = 0.0
    for n in range(N):
        r_n = r - Lambda * k + (n * gamma) / T
        sigma_n = np.sqrt(sigma * sigma + (n * sigmaJ * sigmaJ) / T)
        bs_call_n = black_scholes_call(S, K, T, r_n, sigma_n)
        poisson_weight = np.exp(-lambda_prime * T) * (lambda_prime * T) ** n / math.factorial(n)
        merton_jump_call += poisson_weight * bs_call_n
    return merton_jump_call

def merton_jump_pricing_put(S, K, T, r, sigma, Lambda, muJ, sigmaJ, N):
    """
    Use put-call parity to price a European put under Merton jump diffusion with a truncated series.
    Args:
        S: initial stock price
        K: strike price
        T: time to maturity
        r: risk-free rate
        sigma: asset volatility
        Lambda: jump intensity
        muJ: mean of jump size
        sigmaJ: jump size volatility (aka delta)
        N: number of terms to approximate the infinite series
    """

    return merton_jump_pricing_call(S, K, T, r, sigma, Lambda, muJ, sigmaJ, N) - S + K * np.exp(-r * T)