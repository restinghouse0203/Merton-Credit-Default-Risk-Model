# pricing of Kou's jump diffusion model and Merton's jump diffusion model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import math
import random
import os
import sys
import warnings

# parameters
S = 80 # initial stock price
K = 81 # strike price
T = 0.25 # time to maturity
r = 0.08 # risk-free rate
sigma = 0.2 # volatility
Lambda = 10 # jump intensity
muJ = -0.02 # mean of jump size
sigmaJ = 0.02 # jump size volatility





# Merton's jump diffusion model
from merton_jump_pricing import black_scholes_call, merton_jump_pricing_call, merton_jump_pricing_put
bs = black_scholes_call(S, K, T, r, sigma)
merton = merton_jump_pricing_call(S, K, T, r, sigma, Lambda, muJ, sigmaJ, N)

# print the results
print(f"Black-Scholes price: {bs}")
print(f"Merton's jump diffusion price: {merton}")
print(f"Kou's jump diffusion price: {kou}")

