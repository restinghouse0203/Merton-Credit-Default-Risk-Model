#%%
"""
Created on Thu Dec 12 2018
Hedging Jumps with the Black Scholes model
@author: Lech A. Grzelak

Extended: Kou jump-diffusion paths via ``Process/KouProcess_paths.py``;
side-by-side comparison of Merton vs Kou delta-hedging PnL under BS hedge.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum

# Kou paths live alongside other process modules
_CODE_DIR = Path(__file__).resolve().parent.parent
_PROCESS_DIR = _CODE_DIR / "Process"
if str(_PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESS_DIR))

from KouProcess_paths import GeneratePathsKou

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsMerton(NoOfPaths,NoOfSteps,S0, T,lambdaP,muJ,sigmaJ,r,sigma):    

    # Create empty matrices for Poisson process and for compensated Poisson process

    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
                
    dt = T / float(NoOfSteps)
    X[:,0] = np.log(S0)
    S[:,0] = S0
    
    # Expectation E(e^J) for J~N(muJ,sigmaJ^2)

    EeJ = np.exp(muJ + 0.5*sigmaJ*sigmaJ)
    ZPois = np.random.poisson(lambdaP*dt,[NoOfPaths,NoOfSteps]) 
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    J = np.random.normal(muJ,sigmaJ,[NoOfPaths,NoOfSteps])
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])

        # Making sure that samples from a normal have mean 0 and variance 1

        X[:,i+1]  = X[:,i] + (r - lambdaP*(EeJ-1) - 0.5*sigma*sigma)*dt +sigma*np.sqrt(dt)* Z[:,i]\
                    + J[:,i] * ZPois[:,i]
        time[i+1] = time[i] +dt
        
    S = np.exp(X)
    paths = {"time":time,"X":X,"S":S}
    return paths

def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = np.log(S_0)
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma * sigma) * dt + sigma * (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Compute exponent of ABM

    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths

# Black-Scholes call option price

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * (T-t)) / (sigma * np.sqrt(T-t))
    d2    = d1 - sigma * np.sqrt(T-t)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t))
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t)) - st.norm.cdf(-d1)*S_0
    return value

def BS_Delta(CP,S_0,K,sigma,t,T,r):

    # When defining a time grid it may happen that the last grid point 
    # is slightly behind the maturity time

    if t-T>10e-20 and T-t<10e-7:
        t=T
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * \
             (T-t)) / (sigma * np.sqrt(T-t))
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1)
    elif CP == OptionType.PUT:
       value = st.norm.cdf(d1)-1.0
    return value


def run_bs_delta_hedge(CP, K, sigma, T, r, s0, time, S):
    """
    Black–Scholes delta hedge with ``sigma`` (fine-grid, higher-jump-intensity loop).
    Returns terminal-time PnL paths and time series for call value and delta.
    """
    NoOfPaths, n_nodes = S.shape
    NoOfSteps = n_nodes - 1

    C = lambda t,K,S0: BS_Call_Put_Option_Price(CP,S0,K,sigma,t,T,r)
    Delta = lambda t,K,S0: BS_Delta(CP,S0,K,sigma,t,T,r)

    PnL = np.zeros([NoOfPaths, NoOfSteps+1])
    delta_init = Delta(0.0,K,s0)
    PnL[:,0] = C(0.0,K,s0) - delta_init * s0

    CallM = np.zeros([NoOfPaths, NoOfSteps+1])
    CallM[:,0] = C(0.0,K,s0)
    DeltaM = np.zeros([NoOfPaths, NoOfSteps+1])
    DeltaM[:,0] = Delta(0,K,s0)

    for i in range(1, NoOfSteps+1):
        dt = time[i] - time[i-1]
        delta_old = Delta(time[i-1],K,S[:,i-1])
        delta_curr = Delta(time[i],K,S[:,i])

        PnL[:,i] = PnL[:,i-1]*np.exp(r*dt) - (delta_curr-delta_old)*S[:,i]
        CallM[:,i] = C(time[i],K,S[:,i])
        DeltaM[:,i] = Delta(time[i],K,S[:,i])

    PnL[:,-1] = PnL[:,-1] - np.maximum(S[:,-1]-K,0) + DeltaM[:,-1]*S[:,-1]

    return PnL, CallM, DeltaM


def mainCalculation():
    NoOfPaths = 1000
    NoOfSteps = 2000
    T         = 1.0
    r         = 0.1
    sigma     = 0.2
    lambdaP       = 3.0
    muJ       = 0.0
    sigmaJ    = 0.25
    s0        = 1.0
    K         = [0.95]
    CP        = OptionType.CALL

    # Kou parameters (risk-neutral DE jumps); intensity aligned with Merton lambdaP for comparability
    lam_kou = lambdaP
    p_kou = 0.5
    eta1_kou = 10.0
    eta2_kou = 5.0

    path_id = 10

    np.random.seed(7)
    Paths_merton = GeneratePathsMerton(NoOfPaths,NoOfSteps,s0, T,lambdaP,muJ,sigmaJ,r,sigma)
    time_m = Paths_merton["time"]
    S_m    = Paths_merton["S"]

    np.random.seed(7)
    Paths_kou = GeneratePathsKou(
        NoOfPaths, NoOfSteps, s0, T, r, sigma, lam_kou, p_kou, eta1_kou, eta2_kou
    )
    time_k = Paths_kou["time"]
    S_k    = Paths_kou["S"]

    PnL_m, CallM_m, DeltaM_m = run_bs_delta_hedge(CP, K, sigma, T, r, s0, time_m, S_m)
    PnL_k, CallM_k, DeltaM_k = run_bs_delta_hedge(CP, K, sigma, T, r, s0, time_k, S_k)

    # --- Single-path time series: Merton vs Kou

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(time_m, S_m[path_id,:], label="Stock S(t)")
    ax1.plot(time_m, CallM_m[path_id,:], label="BS call value")
    ax1.plot(time_m, DeltaM_m[path_id,:], label="BS delta")
    ax1.plot(time_m, PnL_m[path_id,:], label="Hedging PnL")
    ax1.set_title(
        "Merton jump–diffusion: Black–Scholes delta hedge "
        f"({NoOfSteps} steps, {NoOfPaths} paths, path {path_id}; λ={lambdaP})"
    )
    ax1.set_ylabel("value")
    ax1.legend(loc="best")
    ax1.grid(True)

    ax2.plot(time_k, S_k[path_id,:], label="Stock S(t)")
    ax2.plot(time_k, CallM_k[path_id,:], label="BS call value")
    ax2.plot(time_k, DeltaM_k[path_id,:], label="BS delta")
    ax2.plot(time_k, PnL_k[path_id,:], label="Hedging PnL")
    ax2.set_title(
        "Kou jump–diffusion (double-exponential): Black–Scholes delta hedge "
        f"(λ={lam_kou}, p={p_kou}, η₁={eta1_kou}, η₂={eta2_kou})"
    )
    ax2.set_xlabel("time")
    ax2.set_ylabel("value")
    ax2.legend(loc="best")
    ax2.grid(True)

    plt.tight_layout()

    # --- Histograms of terminal PnL (original script used 200 bins)

    fig2, (axh1, axh2) = plt.subplots(1, 2, figsize=(11, 4))

    axh1.hist(PnL_m[:,-1], bins=200, density=False, alpha=0.85)
    axh1.set_title("Terminal hedging PnL — Merton jump–diffusion")
    axh1.set_xlabel("PnL at T")
    axh1.grid(True)

    axh2.hist(PnL_k[:,-1], bins=200, density=False, alpha=0.85, color="C1")
    axh2.set_title("Terminal hedging PnL — Kou jump–diffusion")
    axh2.set_xlabel("PnL at T")
    axh2.grid(True)

    low = min(PnL_m[:,-1].min(), PnL_k[:,-1].min())
    high = max(PnL_m[:,-1].max(), PnL_k[:,-1].max())
    pad = 0.05 * (high - low + 1e-12)
    axh1.set_xlim(low - pad, high + pad)
    axh2.set_xlim(low - pad, high + pad)

    plt.tight_layout()

    print(
        "Merton — path {0}, S0={1}, PnL(T-)={2}, S(T)={3}, max(S(T)-K,0)={4}, PnL(T)={5}".format(
            path_id, s0, PnL_m[path_id,-2], S_m[path_id,-1],
            np.maximum(S_m[path_id,-1]-K,0.0), PnL_m[path_id,-1],
        )
    )
    print(
        "Kou    — path {0}, S0={1}, PnL(T-)={2}, S(T)={3}, max(S(T)-K,0)={4}, PnL(T)={5}".format(
            path_id, s0, PnL_k[path_id,-2], S_k[path_id,-1],
            np.maximum(S_k[path_id,-1]-K,0.0), PnL_k[path_id,-1],
        )
    )


mainCalculation()
