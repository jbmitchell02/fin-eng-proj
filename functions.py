
import numpy as np
import pandas as pd

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
    return w.T[0].tolist()[0], w.T * sigma * w

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
    return w_min_var.T[0].tolist()[0], w_min_var.T * sigma * w_min_var

def rebalancing_strategy(asset_returns, target_returns, rebalance_threshold, data_lookback):
    '''
    asset_returns: DataFrame of historical asset returns
    target_returns: target annualized returns
    rebalance_threshold: threshold for rebalancing
    data_lookback: number of days to look back for calculating mean and covariance

    Returns a list of weights at each time step
    '''
    num_assets = len(asset_returns.columns)
    num_periods = len(asset_returns)
    optimal_weights = []
    w_prev = [1/num_assets] * num_assets
    for i in range(data_lookback, num_periods):
        # Calculate mean and covariance over the lookback period
        lookback_returns = asset_returns.iloc[i-data_lookback:i-1]
        m = lookback_returns.mean()
        sigma = lookback_returns.cov()
        # Calculate optimal weights
        w = optimal_portfolio(m, sigma, target_returns/252)[0]
        # If any of the weights deviate from their previous weight by more than the threshold, rebalance
        if any([abs(w[i] - w_prev[i]) > rebalance_threshold for i in range(num_assets)]):
            optimal_weights.append(w)
            w_prev = w
        else:
            optimal_weights.append(w_prev)
    return optimal_weights

def strategy_performance(asset_returns, weights, dates):
    '''
    asset_returns: DataFrame of historical asset returns over the same time period as the weights
    weights: list of weights at each time step
    dates: list of dates corresponding to the weights

    Returns a dataframe of portfolio returns, the mean returns, standard deviation, and Sharpe ratio
    '''
    num_periods = len(asset_returns)
    portfolio_returns = []
    for i in range(num_periods):
        portfolio_returns.append(np.dot(weights[i], asset_returns.iloc[i]))
    portfolio_returns = pd.DataFrame(portfolio_returns, index=dates, columns=['Return'])
    mean = portfolio_returns.mean() * 252
    std = portfolio_returns.std() * np.sqrt(252)
    sharpe = mean/std
    return portfolio_returns, mean.iloc[0], std.iloc[0], sharpe.iloc[0]