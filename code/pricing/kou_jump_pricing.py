#%%
"""
European options under Kou's double-exponential jump–diffusion (risk-neutral).

- ``kou_jump_call_hh``: Tsay-style series (Eq.~6.33) with Hh recursion; intended for
  the symmetric parameter case p=0.5, eta1=eta2>1 (Tsay ``η`` = 1/η_jump).
- ``kou_jump_call_fourier``: Gil–Pelaez / two-integral formula from the characteristic
  function (general p, η₁, η₂).
- ``kou_jump_call_mc``: terminal-time Monte Carlo (same Q as paths).

Put prices use put–call parity. λ→0 matches Black–Scholes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

SQRT2PI = math.sqrt(2.0 * math.pi)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S - K * math.exp(-r * T), 0.0)
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / st
    d2 = d1 - st
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def kou_psi(p: float, eta1: float, eta2: float) -> float:
    return p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0) - 1.0


def Hh(n: int, x: float) -> float:
    """Hermite–type integrals Hh_n(x) used in Tsay/Kou (recursion Eq.~6.35)."""
    if n < -1:
        return 0.0
    if n == -1:
        return math.exp(-0.5 * x * x)
    if n == 0:
        return SQRT2PI * norm.cdf(-x)
    h_nm1 = Hh(0, x)
    h_nm2 = Hh(-1, x)
    for k in range(1, n + 1):
        h_n = (h_nm2 - x * h_nm1) / float(k)
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def _tsay_eta_kappa(psi: float, eta_jump: float) -> tuple[float, float]:
    """Map Kou symmetric jump rates η>1 to Tsay (η_ts, κ) with ψ = e^κ/(1-η_ts²)-1."""
    eta_ts = 1.0 / eta_jump
    inside = max((1.0 + psi) * (1.0 - eta_ts * eta_ts), 1e-300)
    return eta_ts, math.log(inside)


def kou_jump_call_hh(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    N_terms: int = 25,
) -> float:
    """
    Tsay (2005) symmetric double-exponential series with Hh functions.

    Uses η_ts = 1/η_jump with η_jump = η₁ = η₂ and p = 1/2 (checked approximately).
    For strongly asymmetric parameters, use ``kou_jump_call_fourier`` instead.
    """
    if T <= 0.0:
        return max(S - K, 0.0)
    if abs(p - 0.5) > 1e-6 or abs(eta1 - eta2) > 1e-6:
        raise ValueError("kou_jump_call_hh supports symmetric DE (p=0.5, eta1=eta2).")
    if eta1 <= 1.0:
        raise ValueError("Require eta1>1 for Kou jump transform used here.")

    psi = kou_psi(p, eta1, eta2)
    eta_ts, kappa = _tsay_eta_kappa(psi, eta1)
    tau = T
    sig_sqrt_t = sigma * math.sqrt(tau)

    h_plus = (math.log(S / K) + (r + 0.5 * sigma * sigma - lam * psi) * tau) / sig_sqrt_t
    h_minus = (math.log(S / K) + (r - 0.5 * sigma * sigma - lam * psi) * tau) / sig_sqrt_t

    base = math.exp(-lam * tau) * (
        S * math.exp(-lam * psi * tau) * norm.cdf(h_plus) - K * math.exp(-r * tau) * norm.cdf(h_minus)
    )

    ssum = 0.0
    for n in range(1, N_terms + 1):
        pois_w = math.exp(-lam * tau) * (lam * tau) ** n / math.factorial(n)
        omega = math.log(K / S) + lam * psi * tau - (r - 0.5 * sigma * sigma) * tau - n * kappa
        c_plus = sig_sqrt_t / eta_ts + omega / sig_sqrt_t
        c_minus = sig_sqrt_t / eta_ts - omega / sig_sqrt_t

        b_plus = (math.log(S / K) + (r + 0.5 * sigma * sigma - lam * psi) * tau + n * kappa) / sig_sqrt_t
        b_minus = (math.log(S / K) + (r - 0.5 * sigma * sigma - lam * psi) * tau + n * kappa) / sig_sqrt_t

        for j in range(1, n + 1):
            geom = (2.0**j) / (2.0 ** (2 * n - 1)) * math.comb(2 * n - j - 1, n - 1)

            a1 = S * math.exp(-lam * psi * tau + n * kappa)
            a1 *= 0.5 * ((1.0 / (1.0 - eta_ts) ** j) + (1.0 / (1.0 + eta_ts) ** j)) * norm.cdf(b_plus)
            a1 -= math.exp(-r * tau) * K * norm.cdf(b_minus)

            fac2 = 0.5 * math.exp(-r * tau - omega / eta_ts + 0.5 * sigma * sigma * tau / (eta_ts * eta_ts)) * K
            s2 = 0.0
            for i in range(j):
                term = ((1.0 / (1.0 - eta_ts) ** (j - i)) - 1.0) * ((sig_sqrt_t / eta_ts) ** i) * (1.0 / SQRT2PI)
                s2 += term * Hh(i, c_minus)
            a2 = fac2 * s2

            fac3 = 0.5 * math.exp(-r * tau + omega / eta_ts + 0.5 * sigma * sigma * tau / (eta_ts * eta_ts)) * K
            s3 = 0.0
            for i in range(j):
                term = (1.0 - (1.0 / (1.0 + eta_ts) ** (j - i))) * ((sig_sqrt_t / eta_ts) ** i) * (1.0 / SQRT2PI)
                s3 += term * Hh(i, c_plus)
            a3 = fac3 * s3

            ssum += pois_w * geom * (a1 + a2 + a3)

    return base + ssum


def _phi_jump(u: complex, p: float, eta1: float, eta2: float) -> complex:
    """Characteristic function of one Kou DE jump Y (finite part)."""
    return p * eta1 / (eta1 - 1j * u) + (1.0 - p) * eta2 / (eta2 + 1j * u)


def _phi_X(
    u: complex,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
) -> complex:
    """Characteristic function of X = log(S_T/S_0) under risk-neutral Kou."""
    psi = kou_psi(p, eta1, eta2)
    mu_x = r - lam * psi - 0.5 * sigma * sigma
    phi_y = _phi_jump(u, p, eta1, eta2)
    return np.exp(1j * u * mu_x * T - 0.5 * sigma * sigma * u * u * T + lam * T * (phi_y - 1.0))


def kou_jump_call_fourier(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    u_max: float = 200.0,
) -> float:
    """
    European call via Gil–Pelaez: C = S·Π₁ − K·e^{-rT}·Π₂ with k = ln(K/S₀),
    Π₂ = P(X>k), Π₁ from φ_X(u−i)/φ_X(−i). Works for general asymmetric Kou.
    """
    if T <= 0.0:
        return max(S - K, 0.0)
    k = math.log(K / S)

    def phi(u: complex) -> complex:
        return _phi_X(u, T, r, sigma, lam, p, eta1, eta2)

    phi_m_i = phi(-1j)
    if abs(phi_m_i) < 1e-14:
        phi_m_i = complex(math.exp(r * T), 0.0)

    def integrand_pi2(u: float) -> float:
        if u <= 0.0:
            return 0.0
        return float(np.imag(np.exp(-1j * u * k) * phi(u)) / u)

    def integrand_pi1(u: float) -> float:
        if u <= 0.0:
            return 0.0
        z = np.exp(-1j * u * k) * phi(u - 1j) / (1j * u * phi_m_i)
        return float(np.real(z))

    pi2, _ = quad(integrand_pi2, 1e-9, u_max, limit=500)
    pi1, _ = quad(integrand_pi1, 1e-9, u_max, limit=500)
    pi2 = 0.5 + pi2 / math.pi
    pi1 = 0.5 + pi1 / math.pi
    return S * pi1 - K * math.exp(-r * T) * pi2


def kou_jump_put_fourier(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    u_max: float = 200.0,
) -> float:
    c = kou_jump_call_fourier(S, K, T, r, sigma, lam, p, eta1, eta2, u_max=u_max)
    return c + K * math.exp(-r * T) - S


def kou_jump_call_tsay(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    kappa: float,
    eta: float,
    u_max: float = 500.0,
) -> float:
    """
    European call under Kou's model using the **Tsay (textbook) parameterisation**:
      log-jump  Y ~ Laplace(kappa, eta),  i.e.  f_Y(y) = 1/(2*eta)*exp(-|y-kappa|/eta)
      psi = E[J] - 1 = exp(kappa)/(1 - eta^2) - 1,  requires eta < 1

    This matches the parameterisation of Tsay's *Analysis of Financial Time Series*
    (Eq. 6.27–6.33) and replicates Example 6.8 (ct ≈ 3.92, pt ≈ 3.31) with
    S=80, K=81, T=0.25, r=0.08, sigma=0.2, lam=10, kappa=-0.02, eta=0.02.

    Characteristic function of one jump:
      phi_Y(u) = exp(i*u*kappa) / (1 + eta^2 * u^2)

    Pricing: Gil–Pelaez two-integral inversion.
    """
    if T <= 0.0:
        return max(S - K, 0.0)
    if eta >= 1.0 or eta <= 0.0:
        raise ValueError("eta must satisfy 0 < eta < 1 for E[J] to be finite.")

    psi = math.exp(kappa) / (1.0 - eta * eta) - 1.0
    mu_x = r - 0.5 * sigma * sigma - lam * psi
    k_log = math.log(K / S)

    def phi_Y(u: complex) -> complex:
        return np.exp(1j * u * kappa) / (1.0 + eta * eta * u * u)

    def phi_X(u: complex) -> complex:
        return np.exp(
            1j * u * mu_x * T
            - 0.5 * sigma * sigma * u * u * T
            + lam * T * (phi_Y(u) - 1.0)
        )

    phi_mi = phi_X(-1j)

    def integrand_pi2(u: float) -> float:
        if u <= 0.0:
            return 0.0
        return float(np.imag(np.exp(-1j * u * k_log) * phi_X(u)) / u)

    def integrand_pi1(u: float) -> float:
        if u <= 0.0:
            return 0.0
        return float(
            np.real(np.exp(-1j * u * k_log) * phi_X(u - 1j) / (1j * u * phi_mi))
        )

    pi2, _ = quad(integrand_pi2, 1e-9, u_max, limit=1000)
    pi1, _ = quad(integrand_pi1, 1e-9, u_max, limit=1000)
    pi2 = 0.5 + pi2 / math.pi
    pi1 = 0.5 + pi1 / math.pi
    return S * pi1 - K * math.exp(-r * T) * pi2


def kou_jump_put_tsay(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    kappa: float,
    eta: float,
    u_max: float = 500.0,
) -> float:
    """European put via put–call parity using Tsay parameterisation."""
    c = kou_jump_call_tsay(S, K, T, r, sigma, lam, kappa, eta, u_max=u_max)
    return c + K * math.exp(-r * T) - S


def kou_jump_put_hh(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    N_terms: int = 25,
) -> float:
    c = kou_jump_call_hh(S, K, T, r, sigma, lam, p, eta1, eta2, N_terms=N_terms)
    return c + K * math.exp(-r * T) - S


@dataclass
class MonteCarloResult:
    price: float
    stderr: float
    ci95_low: float
    ci95_high: float
    n_paths: int


def kou_jump_call_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    n_paths: int = 200_000,
    seed: int | None = 42,
) -> MonteCarloResult:
    """
    Terminal-time exact simulation: X_T = mu*T + sigma*sqrt(T)*Z + sum_{i=1}^N Y_i,
    N ~ Poisson(lam*T), Y_i i.i.d. Kou DE.
    """
    if seed is not None:
        np.random.seed(seed)
    if T <= 0.0:
        return MonteCarloResult(float(max(S - K, 0.0)), 0.0, 0.0, 0.0, n_paths)

    psi = kou_psi(p, eta1, eta2)
    mu = (r - lam * psi - 0.5 * sigma * sigma) * T
    sig_sqrt_T = sigma * math.sqrt(T)

    N = np.random.poisson(lam * T, size=n_paths)
    Z = np.random.normal(0.0, 1.0, size=n_paths)
    X = mu + sig_sqrt_T * Z

    n_max = int(N.max()) if N.size else 0
    if n_max > 0:
        u = np.random.uniform(0.0, 1.0, size=(n_paths, n_max))
        pos = u < p
        pos_exp = np.random.exponential(1.0 / eta1, size=(n_paths, n_max))
        neg_exp = -np.random.exponential(1.0 / eta2, size=(n_paths, n_max))
        Y = np.where(pos, pos_exp, neg_exp)
        mask = np.arange(n_max)[None, :] < N[:, None]
        X = X + np.sum(Y * mask, axis=1)

    ST = S * np.exp(X)
    payoff = np.maximum(ST - K, 0.0)
    disc = math.exp(-r * T)
    est = disc * payoff
    mean_p = float(np.mean(est))
    std_p = float(np.std(est, ddof=1)) if n_paths > 1 else 0.0
    se = std_p / math.sqrt(n_paths)
    z = 1.96
    return MonteCarloResult(mean_p, se, mean_p - z * se, mean_p + z * se, n_paths)


def _demo() -> None:
    S0, K, T, r, sig = 100.0, 100.0, 0.5, 0.05, 0.15
    lam, p, eta1, eta2 = 2.0, 0.5, 10.0, 10.0

    cf = kou_jump_call_fourier(S0, K, T, r, sig, lam, p, eta1, eta2)
    hh = kou_jump_call_hh(S0, K, T, r, sig, lam, p, eta1, eta2, N_terms=30)
    mc = kou_jump_call_mc(S0, K, T, r, sig, lam, p, eta1, eta2, n_paths=150_000, seed=1)

    bs = black_scholes_call(S0, K, T, r, sig)
    cf0 = kou_jump_call_fourier(S0, K, T, r, sig, 0.0, p, eta1, eta2)

    print("Kou symmetric demo (S=K=100, T=0.5):")
    print(f"  Fourier call: {cf:.6f}")
    print(f"  Hh-series call: {hh:.6f}")
    print(f"  MC call: {mc.price:.6f}  (95% CI [{mc.ci95_low:.6f}, {mc.ci95_high:.6f}])")
    print(f"  Black–Scholes (no jumps): {bs:.6f}")
    print(f"  Fourier with λ=0: {cf0:.6f}  (should match BS ~ {bs:.6f})")

    cf_as = kou_jump_call_fourier(S0, K, T, r, sig, lam, 0.4, 12.0, 8.0)
    mc_as = kou_jump_call_mc(S0, K, T, r, sig, lam, 0.4, 12.0, 8.0, n_paths=300_000, seed=2)
    print("\nAsymmetric Kou (p=0.4, η₁=12, η₂=8):")
    print(f"  Fourier: {cf_as:.6f}, MC: {mc_as.price:.6f} ± {1.96*mc_as.stderr:.6f}")


if __name__ == "__main__":
    _demo()
