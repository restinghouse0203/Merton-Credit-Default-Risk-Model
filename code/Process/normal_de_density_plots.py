#%%
"""
Double-exponential (Kou-style) density vs standard normal — four views.
Default: symmetric unit-variance Laplace (p=0.5, eta1=eta2=sqrt(2)) vs N(0,1).
"""
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def de_pdf(y: np.ndarray, p: float, eta1: float, eta2: float) -> np.ndarray:
    """
    Kou double-exponential density on R:
    f(y) = p * eta1 * exp(-eta1*y) 1_{y>=0} + (1-p) * eta2 * exp(eta2*y) 1_{y<0}.
    Requires p in (0,1), eta1>0, eta2>0.
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    pos = y >= 0
    neg = ~pos
    out[pos] = p * eta1 * np.exp(-eta1 * y[pos])
    out[neg] = (1.0 - p) * eta2 * np.exp(eta2 * y[neg])
    return out


def de_variance(p: float, eta1: float, eta2: float) -> float:
    """Variance of Y ~ DE(p, eta1, eta2)."""
    return 2.0 * p / (eta1**2) + 2.0 * (1.0 - p) / (eta2**2)


def plot_four_views(
    p: float = 0.5,
    eta1: float = math.sqrt(2.0),
    eta2: float = math.sqrt(2.0),
    save_prefix: str | None = None,
    show: bool = True,
) -> None:
    x_full = np.linspace(-8.0, 8.0, 4000)
    y_full = de_pdf(x_full, p, eta1, eta2)
    z_full = norm.pdf(x_full)

    var_de = de_variance(p, eta1, eta2)
    title_suffix = f" (DE: p={p:.3g}, η₁={eta1:.4g}, η₂={eta2:.4g}, Var(Y)={var_de:.4g})"

    # plot in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    
    # 1) Full density
    axs[0, 0].plot(x_full, z_full, label="Normal(0,1)", lw=2)
    axs[0, 0].plot(x_full, y_full, label="Double-exponential", lw=2)
    axs[0, 0].set_title("Full density" + title_suffix)
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("density")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    if save_prefix:
        fig.savefig(save_prefix + "_full.png", dpi=150)

    # 2) Peak zoom
    x_peak = np.linspace(-1.2, 1.2, 2000)
    axs[0, 1].plot(x_peak, norm.pdf(x_peak), label="Normal(0,1)", lw=2)
    axs[0, 1].plot(x_peak, de_pdf(x_peak, p, eta1, eta2), label="Double-exponential", lw=2)
    axs[0, 1].set_title("Peak")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("density")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    if save_prefix:
        fig.savefig(save_prefix + "_peak.png", dpi=150)

    # 3) Left tail
    x_left = np.linspace(-12.0, -4.0, 2000)
    axs[1, 0].semilogy(x_left, norm.pdf(x_left) + 1e-300, label="Normal(0,1)", lw=2)
    axs[1, 0].semilogy(x_left, de_pdf(x_left, p, eta1, eta2) + 1e-300, label="Double-exponential", lw=2)
    axs[1, 0].set_title("Left tail")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("density (log)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    fig.tight_layout()
    if save_prefix:
        fig.savefig(save_prefix + "_left_tail.png", dpi=150)

    # 4) Right tail
    x_right = np.linspace(4.0, 12.0, 2000)
    axs[1, 1].semilogy(x_right, norm.pdf(x_right) + 1e-300, label="Normal(0,1)", lw=2)
    axs[1, 1].semilogy(x_right, de_pdf(x_right, p, eta1, eta2) + 1e-300, label="Double-exponential", lw=2)
    axs[1, 1].set_title("Right tail")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("density (log)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    if save_prefix:
        fig.savefig(save_prefix + "_right_tail.png", dpi=150)

    if show:
        plt.show()


def _parse_args() -> argparse.Namespace:
    # p = argparse.ArgumentParser(description="Normal vs Kou double-exponential densities")
    p = argparse.ArgumentParser(description="")
    p.add_argument("--p", type=float, default=0.5, help="probability of nonnegative jump")
    p.add_argument("--eta1", type=float, default=math.sqrt(2.0), help="up-tail rate eta1")
    p.add_argument("--eta2", type=float, default=math.sqrt(2.0), help="down-tail rate eta2")
    p.add_argument("--save-prefix", type=str, default=None, help="if set, save PNGs as prefix_{full,peak,left_tail,right_tail}.png")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not (0.0 < args.p < 1.0) or args.eta1 <= 0 or args.eta2 <= 0:
        raise SystemExit("require 0<p<1 and eta1,eta2>0")
    do_show = args.save_prefix is None
    plot_four_views(
        p=args.p,
        eta1=args.eta1,
        eta2=args.eta2,
        save_prefix=args.save_prefix,
        show=do_show,
    )
