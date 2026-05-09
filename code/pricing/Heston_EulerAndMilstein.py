#%%
"""
Created on Jan 20 2019
Convergence of option prices for Euler and Milsten schemes
@author: Lech A. Grzelak
"""
import numpy as np
import scipy.stats as st
import enum 

# set i= imaginary number
i   = np.complex128(0.0,1.0)

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
# Black-Scholes Call option price
def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / float(sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Euler Approximation
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i]* dt + sigma * S1[:,i] * (W[:,i+1] - W[:,i])
        time[i+1] = time[i] +dt
        
    # Retun S1 and S2
    paths = {"time":time,"S":S1}
    return paths

def BS_Cash_Or_Nothing_Price(CP,S_0,K,sigma,tau,r):
    #Black-Scholes for cash or nothing option
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / float(sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = K * np.exp(-r * tau) * st.norm.cdf(d2)
    if CP == OptionType.PUT:
        value = K * np.exp(-r * tau) *(1.0 - st.norm.cdf(d2))
    return value

def GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Milstein Approximation
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
       
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i] 
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i]* dt + sigma * S1[:,i] * (W[:,i+1] - W[:,i]) \
                    + 0.5 * sigma **2.0 * S1[:,i] * (np.power((W[:,i+1] - W[:,i]),2) - dt)
        time[i+1] = time[i] +dt
        
    # Retun S1 and S2
    paths = {"time":time,"S":S1}
    return paths

def EUOptionPriceFromMCPaths(CP,S,K,T,r):
    # S is a vector of Monte Carlo samples at T
    if CP == OptionType.CALL:
        return np.exp(-r*T)*np.mean(np.maximum(S-K,0.0))
    elif CP == OptionType.PUT:
        return np.exp(-r*T)*np.mean(np.maximum(K-S,0.0))

def CashofNothingPriceFromMCPaths(CP,S,K,T,r):
    # S is a vector of Monte Carlo samples at T
    if CP == OptionType.CALL:
        return np.exp(-r*T)*K*np.mean((S>K))
    elif CP == OptionType.PUT:
        return np.exp(-r*T)*K*np.mean((S<=K))


def GeneratePathsHestonEuler(
    NoOfPaths,
    NoOfSteps,
    T,
    r,
    S_0,
    v0,
    kappa,
    theta,
    xi,
    rho,
):
    """
    Euler full-truncation simulation of Heston model:
        dS_t = r S_t dt + sqrt(v_t) S_t dW_1
        dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_2
        corr(dW_1, dW_2) = rho
    """
    dt = T / float(NoOfSteps)
    sqrt_dt = np.sqrt(dt)

    z1 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    z2 = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])

    # Brownian increments with target correlation rho
    dW1 = sqrt_dt * z1
    dW2 = sqrt_dt * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)

    S = np.zeros([NoOfPaths, NoOfSteps + 1])
    V = np.zeros([NoOfPaths, NoOfSteps + 1])
    time = np.linspace(0.0, T, NoOfSteps + 1)

    S[:, 0] = S_0
    V[:, 0] = v0

    for t in range(NoOfSteps):
        v_pos = np.maximum(V[:, t], 0.0)
        S[:, t + 1] = S[:, t] + r * S[:, t] * dt + np.sqrt(v_pos) * S[:, t] * dW1[:, t]
        V[:, t + 1] = (
            V[:, t]
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos) * dW2[:, t]
        )
        V[:, t + 1] = np.maximum(V[:, t + 1], 0.0)

    return {"time": time, "S": S, "V": V}


def MertonPricesUnderHestonMC(
    asset_value,
    debt_face_value,
    T,
    r,
    NoOfPaths=20000,
    NoOfSteps=252,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    xi=0.30,
    rho=-0.7,
    seed=42,
):
    """
    Merton interpretation under Heston asset dynamics:
      Equity = discounted E[(A_T - D)^+]
      Debt   = discounted E[min(A_T, D)] = D*exp(-rT) - Put(A_0, D)
    """
    np.random.seed(seed)
    paths = GeneratePathsHestonEuler(
        NoOfPaths=NoOfPaths,
        NoOfSteps=NoOfSteps,
        T=T,
        r=r,
        S_0=asset_value,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
    )

    A_T = paths["S"][:, -1]
    equity = np.exp(-r * T) * np.mean(np.maximum(A_T - debt_face_value, 0.0))
    debt = np.exp(-r * T) * np.mean(np.minimum(A_T, debt_face_value))

    return {"equity": equity, "debt": debt}


def heston_european_mc(
    S_0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    is_call=True,
    NoOfPaths=50_000,
    NoOfSteps=252,
    seed=42,
):
    """
    European call or put under Heston dynamics using the same Euler full-truncation
    scheme as GeneratePathsHestonEuler (risk-neutral square-root variance process).

    Parameters xi matches vol-of-vol in dV = kappa(theta - V)dt + xi*sqrt(V)dW2.
    """
    np.random.seed(seed)
    cp = OptionType.CALL if is_call else OptionType.PUT
    paths = GeneratePathsHestonEuler(
        NoOfPaths=NoOfPaths,
        NoOfSteps=NoOfSteps,
        T=T,
        r=r,
        S_0=S_0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
    )
    S_T = paths["S"][:, -1]
    return EUOptionPriceFromMCPaths(cp, S_T, K, T, r)


def heston_quantlib_vanilla(
    S_0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    q=0.0,
    reference_date=None,
    is_call=True,
):
    """
    Semi-analytic European call or put under Heston (QuantLib AnalyticHestonEngine).

    Uses the same variance dynamics as ``GeneratePathsHestonEuler`` (square-root process,
    ``xi`` is vol-of-vol). Requires ``pip install QuantLib``.
    """
    import QuantLib as ql

    if reference_date is None:
        reference_date = ql.Date(8, 5, 2026)
    ql.Settings.instance().evaluationDate = reference_date
    day_count = ql.Actual365Fixed()
    spot_handle = ql.SimpleQuote(S_0)
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(reference_date, r, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(reference_date, q, day_count)
    )
    exercise_date = reference_date + ql.Period(int(round(T * 365)), ql.Days)
    exercise = ql.EuropeanExercise(exercise_date)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if is_call else ql.Option.Put, K
    )
    option = ql.VanillaOption(payoff, exercise)
    heston_process = ql.HestonProcess(
        risk_free_ts,
        dividend_ts,
        ql.QuoteHandle(spot_handle),
        v0,
        kappa,
        theta,
        xi,
        rho,
    )
    option.setPricingEngine(
        ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
    )
    return option.NPV()


def mainCalculation():
    CP= OptionType.CALL
    T = 1
    r = 0.06
    sigma = 0.3
    S_0 = 5
    K = [S_0]
    NoOfSteps =1000
    
    # Simulated paths
    NoOfPathsV = [100,1000,5000,10000]
    
    # Call price
    exactPrice = BS_Call_Option_Price(CP,S_0,K,sigma,T,r)[0]
    print("EUROPEAN OPTION PRICING")
    print("Exact option price = {0}".format(exactPrice))
    for NoOfPathsTemp in NoOfPathsV:
        np.random.seed(1)
        PathsEuler    = GeneratePathsGBMEuler(NoOfPathsTemp,NoOfSteps,T,r,sigma,S_0)
        np.random.seed(1)
        PathsMilstein = GeneratePathsGBMMilstein(NoOfPathsTemp,NoOfSteps,T,r,sigma,S_0)
        S_Euler = PathsEuler["S"]
        S_Milstein = PathsMilstein["S"]
        priceEuler = EUOptionPriceFromMCPaths(CP,S_Euler[:,-1],K,T,r)
        priceMilstein = EUOptionPriceFromMCPaths(CP,S_Milstein[:,-1],K,T,r)
        print("For N = {0} Euler scheme yields option price = {1} and Milstein {2}"\
              .format(NoOfPathsTemp,priceEuler,priceMilstein))
        print("For N = {0} Euler error = {1} and Milstein  error {2}"\
              .format(NoOfPathsTemp,priceEuler-exactPrice,priceMilstein-exactPrice))
    
    # Cash or nothing price
    print("CASH OR NOTHING PRICING")
    exactPrice = BS_Cash_Or_Nothing_Price(CP,S_0,K,sigma,T,r)
    print("Exact option price = {0}".format(exactPrice))
    for NoOfPathsTemp in NoOfPathsV:
        np.random.seed(1)
        PathsEuler    = GeneratePathsGBMEuler(NoOfPathsTemp,NoOfSteps,T,r,sigma,S_0)
        np.random.seed(1)
        PathsMilstein = GeneratePathsGBMMilstein(NoOfPathsTemp,NoOfSteps,T,r,sigma,S_0)
        S_Euler = PathsEuler["S"]
        S_Milstein = PathsMilstein["S"]
        priceEuler = CashofNothingPriceFromMCPaths(CP,S_Euler[:,-1],K[0],T,r)
        priceMilstein = CashofNothingPriceFromMCPaths(CP,S_Milstein[:,-1],K[0],T,r)
        print("For N = {0} Euler scheme yields option price = {1} and Milstein {2}"\
              .format(NoOfPathsTemp,priceEuler,priceMilstein))
        print("For N = {0} Euler error = {1} and Milstein  error {2}"\
              .format(NoOfPathsTemp,priceEuler-exactPrice,priceMilstein-exactPrice))
if __name__ == "__main__":
    mainCalculation()