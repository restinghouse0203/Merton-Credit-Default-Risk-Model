#%%
"""
Created on Thu Jan 11 2019
Paths for the Jump diffusion proces of Merton
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsMerton(NoOfPaths,NoOfSteps,S0, T,xiP,muJ,sigmaJ,r,sigma):    
    # Create empty matrices for Poisson process and for compensated Poisson process
    X = np.zeros([NoOfPaths, NoOfSteps+1]) # Merton process
    S = np.zeros([NoOfPaths, NoOfSteps+1]) # stock price
    time = np.zeros([NoOfSteps+1]) # time
                
    dt = T / float(NoOfSteps)
    X[:,0] = np.log(S0) # log of the initial stock price
    S[:,0] = S0 # initial stock price
    
    # Expectation E(e^J) for J~N(muJ,sigmaJ^2)
    EeJ = np.exp(muJ + 0.5*sigmaJ*sigmaJ)
    
    ZPois = np.random.poisson(xiP*dt,[NoOfPaths,NoOfSteps]) # Poisson process
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps]) # Normal process
    
    J = np.random.normal(muJ,sigmaJ,[NoOfPaths,NoOfSteps]) # Jump size, normally distributed
    
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        # making sure that samples from normal have mean 0 and variance 1
        X[:,i+1]  = X[:,i] + (r - xiP*(EeJ-1) - 0.5*sigma*sigma)*dt +sigma*np.sqrt(dt)* Z[:,i]\
                    + J[:,i] * ZPois[:,i] # jump process
        time[i+1] = time[i] +dt # time step
        
    S = np.exp(X) # stock price
    paths = {"time":time,"X":X,"S":S} # paths
    return paths

def mainCalculation(NoOfPaths, NoOfSteps, T, xiP, muJ, sigmaJ, sigma, S0, r):

    Paths = GeneratePathsMerton(NoOfPaths,NoOfSteps,S0, T,xiP,muJ,sigmaJ,r,sigma)
    timeGrid = Paths["time"]
    X = Paths["X"]
    S = Paths["S"]
           
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(timeGrid, np.transpose(X))  
    ax[0].set_title("Merton Process log-price X(t) Paths")
    ax[0].grid()
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("X(t)")
    ax[1].plot(timeGrid, np.transpose(S))   
    ax[1].set_title("Merton Process spot price S(t) Paths")
    ax[1].grid()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("S(t)")
    

# mainCalculation()