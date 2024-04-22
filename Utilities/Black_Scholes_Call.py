import numpy as np
from scipy.stats import norm


def Black_Scholes_Call(S, K, r, sigma, q, T):
    """Black-Scholes Call option price.

    Args:
        S (float): initial stock price
        K (float): strike price
        r (float): risk-free rate
        sigma (float): volatility
        q (float): dividend yield
        T (float): time to maturity

    Returns:
        float: call option price
    """
    if sigma == 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    else:
        d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
