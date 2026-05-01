# Merton Credit Risk Model and Jump-Diffusion Extensions

A stochastic-calculus project on structural credit risk, centered on Merton’s model and extended with jump-diffusion frameworks for improved tail-risk and default-probability modeling.

## Overview

This repository studies corporate default risk under structural models where equity is modeled as a contingent claim on firm assets.  
It compares:

- Black–Scholes / Merton structural baseline
- Merton jump-diffusion extension
- Kou double-exponential jump-diffusion extension
- (Optional benchmark) Heston volatility model components

Main outputs include:

- Distance to Default (DD)
- Physical default probability (PD under P)
- Risk-neutral default probability (PD under Q)
- Cross-model empirical comparison tables and plots

## Mathematical Context

Under the classical Merton framework:

- Firm defaults at horizon \( T \) if \( A_T < D \) (asset value below debt face value)
- Equity resembles a call option on assets
- Debt resembles risk-free debt minus a put option

Jump extensions are included to capture discontinuities and heavy tails that continuous diffusions miss.

## Repository Layout

- `src/credit_risk/models/` — pricing model implementations
- `src/credit_risk/structural/` — PD/DD structural default pipelines
- `src/credit_risk/simulation/` — path generation and Monte Carlo utilities
- `scripts/` — reproducible entry points for pipelines and figure generation
- `notebooks/` — exploratory and presentation notebooks
- `data/` — raw and processed inputs
- `results/` — generated tables and figures
- `tests/` — unit and smoke tests
- `docs/slides/` — presentation slides and documentation artifacts

## Data

Expected inputs include:

- Equity price history (e.g., AAPL, JPM, XOM, etc.)
- Balance-sheet debt proxies
- Risk-free rate series (e.g., 3M Treasury)

Raw datasets live in `data/raw/`; transformed datasets are stored in `data/processed/`.

## Installation
TBA

## Quick Start

Run baseline Merton structural metrics:
TBA

Run Kou jump-based structural metrics:
TBA

Merge model outputs for comparison:
TBA

Outputs are written to:

- `results/tables/`
- `results/figures/`

## Reproducibility

1. Fix random seeds where simulation is used.
2. Keep model parameter sets under version control (e.g., `src/credit_risk/calibration/params.py`).
3. Regenerate all figures/tables via scripts, not manual notebook edits.

## Testing

```bash
pytest -q
```

Recommended tests include:

- pricing sanity checks against known limits
- pipeline smoke tests on small ticker universes
- consistency tests between CSV outputs and in-memory pipeline DataFrames

## Current Scope and Limitations

- Asset value is proxied from market cap + debt conventions
- Balance-sheet frequency limits asset observability
- Risk-neutral jump calibration can be model-dependent
- Multi-debt/coupon structures are only partially explored

## References

- Merton, R. C. (1974). On the Pricing of Corporate Debt
- Merton, R. C. (1976). Option Pricing When Underlying Stock Returns Are Discontinuous
- Kou, S. G. (2002). A Jump-Diffusion Model for Option Pricing
- Lando, D. (2004). Credit Risk Modeling: Theory and Applications
- Oosterlee, C. W., & Grzelak, L. A. (2019). Mathematical Modeling and Computation in Finance
