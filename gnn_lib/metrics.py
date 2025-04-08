import numpy as np
from scipy.stats import linregress


def mae(y, y_hat):
    """mean absolute error"""
    return abs(np.array(y) - y_hat).mean()



def rmse(y, y_hat):
    """root mean squared error"""
    return np.sqrt(np.mean((np.array(y)-y_hat)**2))



def r2_score(y, y_hat):
    """
    coefficient of determination
    """
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum((y - np.array(y_hat)) ** 2)
    r2 = 1 - (rss / tss)
    return r2



def get_metrics(y, y_hat):

    """
    calculate MAE, RMSE, Rp, slope, and R2 metrics
    """

    res = linregress(y, y_hat)
    metrics = {
        'MAE': mae(y, y_hat),
        'RMSE': rmse(y, y_hat),
        'Rp': res.rvalue,
        'Slope': res.slope,
        'R2': r2_score(y, y_hat)
    }
    return metrics


