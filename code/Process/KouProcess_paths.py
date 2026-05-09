#%%
"""
Kou double-exponential jump–diffusion paths (log-price + spot).
Risk-neutral drift: X_{t+dt} = X_t + (r - lambda*psi - sigma^2/2)*dt + sigma*sqrt(dt)*Z + sum of jumps,
with i.i.d. jump sizes Y ~ DE(p, eta1, eta2) and Poisson(lambda*dt) jump counts per step.

Requires eta1 > 1 and eta2 > 0 so that E[e^Y] < infty and martingale compensator psi = E[e^Y] - 1.
@author: course materials (pattern after Grzelak MertonProcess_paths)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def kou_psi(p: float, eta1: float, eta2: float) -> float:
    """Risk-neutral compensator psi = E[e^Y] - 1 for Kou DE jump Y."""
    return p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0) - 1.0


def _sample_kou_jump_size(p: float, eta1: float, eta2: float, shape: tuple[int, ...]) -> np.ndarray:
    """One Kou jump: +Exp(eta1) with prob p, -Exp(eta2) with prob 1-p."""
    u = np.random.uniform(0.0, 1.0, shape)
    pos = u < p
    out = np.empty(shape, dtype=float)
    out[pos] = np.random.exponential(1.0 / eta1, np.sum(pos))
    out[~pos] = -np.random.exponential(1.0 / eta2, np.sum(~pos))
    return out


def GeneratePathsKou(
    NoOfPaths: int,
    NoOfSteps: int,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
) -> dict:
    """
    Simulate Kou jump–diffusion under Q (discounted stock is a martingale).

    Args:
        NoOfPaths, NoOfSteps: Monte Carlo paths and Euler steps.
        S0: initial spot.
        T: horizon.
        r: risk-free rate.
        sigma: diffusion vol.
        lam: jump intensity.
        p: prob jump is nonnegative (upward exponential).
        eta1: rate of upward exponential ( > 1 ).
        eta2: rate of downward exponential ( > 0 ).
    """
    if eta1 <= 1.0 or eta2 <= 0.0:
        raise ValueError("Require eta1 > 1 and eta2 > 0 for finite E[e^Y].")
    if not (0.0 < p < 1.0):
        raise ValueError("Require p in (0,1).")

    psi = kou_psi(p, eta1, eta2)
    dt = T / float(NoOfSteps)
    X = np.zeros((NoOfPaths, NoOfSteps + 1))
    S = np.zeros((NoOfPaths, NoOfSteps + 1))
    time = np.zeros(NoOfSteps + 1)

    X[:, 0] = np.log(S0)
    S[:, 0] = S0

    drift = r - lam * psi - 0.5 * sigma * sigma
    ZPois = np.random.poisson(lam * dt, size=(NoOfPaths, NoOfSteps))
    Z = np.random.normal(0.0, 1.0, size=(NoOfPaths, NoOfSteps))

    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        counts = ZPois[:, i].astype(int)
        jumps = np.zeros(NoOfPaths)
        n_max = int(np.max(counts)) if counts.size else 0
        if n_max > 0:
            extras = _sample_kou_jump_size(p, eta1, eta2, (NoOfPaths, n_max))
            for m in range(n_max):
                jumps += extras[:, m] * (counts > m)
        X[:, i + 1] = X[:, i] + drift * dt + sigma * np.sqrt(dt) * Z[:, i] + jumps
        time[i + 1] = time[i] + dt

    S = np.exp(X)
    return {"time": time, "X": X, "S": S}


def mainCalculation(
    NoOfPaths: int = 50,
    NoOfSteps: int = 500,
    S0: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    lam: float = 3.0,
    p: float = 0.5,
    eta1: float = 10.0,
    eta2: float = 10.0,
) -> None:
    paths = GeneratePathsKou(NoOfPaths, NoOfSteps, S0, T, r, sigma, lam, p, eta1, eta2)
    time_grid = paths["time"]
    X = paths["X"]
    S = paths["S"]

    plt.figure(1, figsize=(8, 5))
    plt.plot(time_grid, np.transpose(X))
    plt.title("Kou jump–diffusion log-price X(t)")
    plt.grid(True, alpha=0.3)
    plt.xlabel("time")
    plt.ylabel("X(t)")

    plt.figure(2, figsize=(8, 5))
    plt.plot(time_grid, np.transpose(S))
    plt.title("Kou jump–diffusion spot S(t)")
    plt.grid(True, alpha=0.3)
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.show()


if __name__ == "__main__":
    mainCalculation()
