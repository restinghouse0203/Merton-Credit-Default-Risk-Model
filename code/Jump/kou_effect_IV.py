#%%
# Task: change the merton's parameter to kou's parameter and plot the effect of eta1, eta2, p on implied volatility

"""
Created on May 8 2026
Kou Jump-Diffusion Model and implied volatilities obtained with the COS method
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
import scipy.optimize as optimize
import math
from scipy.integrate import quad

# Set i= imaginary number

i   = np.complex128(0.0,1.0)

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
# Black-Scholes call option price

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

# Implied volatility method

def ImpliedVolatility(CP,marketPrice,K,T,S_0,r,initialVol = 0.4):
    func = lambda sigma: np.power(BS_Call_Option_Price(CP,S_0,K,sigma,T,r) \
                                  - marketPrice, 1.0)
    impliedVol = optimize.newton(func, initialVol, tol=1e-7)
    
    return impliedVol

# Kou model helper functions

def kou_psi(p, eta1, eta2):
    """Compensator for Kou model: E[e^Y] - 1"""
    return p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0) - 1.0

def _phi_jump(u, p, eta1, eta2):
    """Characteristic function of one Kou DE jump Y"""
    return p * eta1 / (eta1 - 1j * u) + (1.0 - p) * eta2 / (eta2 + 1j * u)

def _phi_X(u, T, r, sigma, lam, p, eta1, eta2):
    """Characteristic function of X = log(S_T/S_0) under risk-neutral Kou"""
    psi = kou_psi(p, eta1, eta2)
    mu_x = r - lam * psi - 0.5 * sigma * sigma
    phi_y = _phi_jump(u, p, eta1, eta2)
    return np.exp(1j * u * mu_x * T - 0.5 * sigma * sigma * u * u * T + lam * T * (phi_y - 1.0))

def KouCallPrice(S, K, T, r, sigma, lam, p, eta1, eta2, u_max=200.0):
    """
    European call price under Kou model via Gil-Pelaez Fourier inversion.
    Works for vectorized strikes K.
    """
    K = np.array(K).reshape([len(K),1])
    
    # Vectorized computation for each strike
    prices = np.zeros_like(K)
    
    for idx in range(len(K)):
        K_val = float(K[idx, 0])
        
        if T <= 0.0:
            prices[idx] = max(S - K_val, 0.0)
            continue
            
        k = math.log(K_val / S)
        
        def phi(u):
            return _phi_X(u, T, r, sigma, lam, p, eta1, eta2)
        
        phi_m_i = phi(-1j)
        if abs(phi_m_i) < 1e-14:
            phi_m_i = complex(math.exp(r * T), 0.0)
        
        def integrand_pi2(u):
            if u <= 0.0:
                return 0.0
            return float(np.imag(np.exp(-1j * u * k) * phi(u)) / u)
        
        def integrand_pi1(u):
            if u <= 0.0:
                return 0.0
            z = np.exp(-1j * u * k) * phi(u - 1j) / (1j * u * phi_m_i)
            return float(np.real(z))
        
        pi2, _ = quad(integrand_pi2, 1e-9, u_max, limit=500)
        pi1, _ = quad(integrand_pi1, 1e-9, u_max, limit=500)
        pi2 = 0.5 + pi2 / math.pi
        pi1 = 0.5 + pi1 / math.pi
        prices[idx] = S * pi1 - K_val * math.exp(-r * T) * pi2
    
    return prices


def mainCalculation():
    CP  = OptionType.CALL
    S0  = 100
    r   = 0.0
    tau = 2
    
    K = np.linspace(40,180,25)
    K = np.array(K).reshape([len(K),1])

    sigma  = 0.25
    lambdaP = 0.1  # jump intensity
    
    # Baseline Kou parameters
    p_base = 0.5
    eta1_base = 10.0
    eta2_base = 10.0
     
    # Effect of eta1 (upward jump rate)

    plt.figure(1)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    plt.title(f"Effect of eta1 (upward jump rate) on implied volatility")
    eta1_V = [5.0, 8.0, 10.0, 15.0, 20.0]
    legend = []
    for eta1_temp in eta1_V:    

        # Evaluate the Kou model

        valueExact = KouCallPrice(S0, K, tau, r, sigma, lambdaP, p_base, eta1_temp, eta2_base)
        
        # Implied volatilities

        IV = np.zeros([len(K),1])
        for idx in range(0,len(K)):
            IV[idx] = ImpliedVolatility(CP, valueExact[idx], K[idx], tau, S0, r)
        plt.plot(K, IV)
        legend.append('eta1={0}'.format(eta1_temp))
    plt.legend(legend)
    
    # Effect of eta2 (downward jump rate)

    plt.figure(2)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    plt.title(f"Effect of eta2 (downward jump rate) on implied volatility")
    eta2_V = [5.0, 8.0, 10.0, 15.0, 20.0]
    legend = []
    for eta2_temp in eta2_V:    

        # Evaluate the Kou model

        valueExact = KouCallPrice(S0, K, tau, r, sigma, lambdaP, p_base, eta1_base, eta2_temp)
        
        # Implied volatilities

        IV = np.zeros([len(K),1])
        for idx in range(0,len(K)):
            IV[idx] = ImpliedVolatility(CP, valueExact[idx], K[idx], tau, S0, r, 0.3)
        
        plt.plot(K, IV)
        legend.append('eta2={0}'.format(eta2_temp))
    plt.legend(legend)
    
    # Effect of p (probability of upward jump)

    plt.figure(3)
    plt.grid()
    plt.title(f"Effect of p (upward jump probability) on implied volatility")
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    p_V = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]
    legend = []
    for p_temp in p_V:    

        # Evaluate the Kou model

        valueExact = KouCallPrice(S0, K, tau, r, sigma, lambdaP, p_temp, eta1_base, eta2_base)
        
        # Implied volatilities

        IV = np.zeros([len(K),1])
        for idx in range(0,len(K)):
            IV[idx] = ImpliedVolatility(CP, valueExact[idx], K[idx], tau, S0, r)
        plt.plot(K, IV)
        legend.append('p={0}'.format(p_temp))
    plt.legend(legend)
    
      
mainCalculation()
