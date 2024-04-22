import numpy as np
from scipy.stats import norm


def Black_Scholes_Put(S, K, r, sigma, q, T):
    """Black-Scholes Put option price.

    Args:
        S (float): initial stock price
        K (float): strike price
        r (float): risk-free rate
        sigma (float): volatility
        q (float): dividend yield
        T (float): time to maturity

    Returns:
        float: put option price
    """
    if sigma == 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    else:
        d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
