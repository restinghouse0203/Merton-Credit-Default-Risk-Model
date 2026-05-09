# Jump Diffusion Models for Structural Credit Risk

(Stochastic Calculus project) comparing stochastic models for **structural credit risk** under Merton’s framework, with emphasis on **jump-diffusion** specifications versus simpler benchmarks.
- Completed under ORIE 5610 — Stochastic Calculus II, Cornell University.

## Overview

This project studies **jump-diffusion models** in a credit-risk setting. We compare **investment-grade (“safe”)** firms with **default-risky** names to evaluate how different stochastic-calculus models behave in theory and in empirical credit metrics.

**Data sources**

- **[yfinance](https://pypi.org/project/yfinance/)** — equity prices and balance-sheet inputs used for Merton-style structural inputs.
- **FRED / Treasury yields** — short risk-free rates (e.g., 3-month Treasury); local CSV snapshots are stored under `data/` (e.g., `US_treasury_yield3m.csv`, `risk_free_rate.csv`, SOFR and longer yields where applicable).

**Code acknowledgments**

Some utilities and numerical experiments follow patterns from Lech A. Grzelak’s *Computational Finance* materials; see the book’s companion repository:  
[QuantFinanceBook / PythonCodes](https://github.com/LechGrzelak/QuantFinanceBook/tree/master/PythonCodes).

## Main takeaways

- **Kou’s jump-diffusion model** produced the strongest **theoretical** behavior along the dimensions we studied (e.g., **volatility smile**, **heavy left tails**, **delta hedging** under jumps).
- In **empirical** exercises, the gains over a simple benchmark were often **modest**: **Black–Scholes–Merton (BSM)** was frequently **adequate** for practical credit metrics such as **probability of default (PD)**, **credit spread**, and **credit valuation**, given the data and implementation choices used here.
- The written report also discusses **limitations of Merton’s structural default framework**, including **unobserved asset value and volatility**, and **default drivers not captured in market or accounting data** (for example, risks revealed only through events such as regulatory or operational issues rather than prices alone).

## Repository layout

```text
project/
├── README.md
├── data/                    # Downloaded / cached inputs (prices, yields, balance sheets)
├── results/
│   └── tables/              # Exported CSV summaries (model comparisons, PD/DD tables)
└── code/
    ├── Process/             # Path simulation (GBM, jumps, Kou/Merton jump processes)
    ├── Jump/                # Jump-model figures / IV effects (book-style experiments)
    ├── heston/              # Heston/CIR helpers and discretization utilities
    ├── Hedging/             # Delta hedging (BS and jump-augmented paths)
    ├── pricing/             # Pricing, IV, cross-model comparison notebooks/scripts
    └── metrics/             # Default metrics, ECL, pipelines per model
```

Run notebooks from the `code/` subfolders they live in (or adjust paths) so imports resolve correctly.

## Report layout

The full write-up is organized as follows:

1. **Background and Motivation**
2. **Merton’s Structural Default Risk Model**
3. **Pricing Models**
   1. Black–Scholes–Merton  
   2. Heston  
   3. Merton’s Jump Diffusion  
   4. Kou’s Jump Diffusion  
4. **Application to Finance**
   1. Volatility Smile  
   2. Delta Hedging  
   3. Theoretical Pricing Comparison  
   4. Implementation Note  
5. **Empirical Results**
   1. Credit Valuation  
   2. Credit Spread  
   3. Probability of Default  
6. **Conclusion**
7. **Appendix**
8. **References**

## Dependencies

Python 3 with typical scientific stack (**NumPy**, **Pandas**, **Matplotlib**, etc.) and **`yfinance`**. Exact versions are not pinned in this repository; create a virtual environment and install packages as needed for your setup.

## License and academic use

If you reuse this work, cite the course/report appropriately and retain attribution to third-party code (e.g., Grzelak’s companion repository where adapted).
