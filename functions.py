
import numpy as np

def optimal_portfolio(m, sigma, mu_b):
    '''
    m: row vector of expected returns
    sigma: covariance matrix of returns
    mu_b: expected portfolio return
    
    Returns the weights and variance of the minimum variance portfolio
    '''
    m = np.matrix(m).T
    sigma = np.matrix(sigma)
    e = np.matrix(np.ones(len(m))).T
    sigma_inv = sigma.I
    # Calculate the minimum variance and market porfolio weights
    w_min_var = (sigma_inv * e) / (e.T * sigma_inv * e)
    w_mk = (sigma_inv * m) / (e.T * sigma_inv * m)
    # Calculate the return-constrained minimum variance portfolio
    v = w_mk - w_min_var
    alpha = ((mu_b - m.T * w_min_var) / (m.T * v))[0, 0]
    w = w_min_var + alpha * v
    return w, w.T * sigma * w

def min_var_portfolio(m, sigma):
    '''
    m: row vector of expected returns
    sigma: covariance matrix of returns
    
    Returns the weights and variance of the minimum variance portfolio
    '''
    m = np.matrix(m).T
    sigma = np.matrix(sigma)
    e = np.matrix(np.ones(len(m))).T
    sigma_inv = sigma.I
    w_min_var = (sigma_inv * e) / (e.T * sigma_inv * e)
    return w_min_var, w_min_var.T * sigma * w_min_var
