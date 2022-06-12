import numpy as np


def get_random_mos(y1m, y1s, y2m, y2s, y3m, y3s, y_rho=None, corr_trgt=0):
    if corr_trgt == 0:
        # simulate uncorrelated random scores
        y1 = np.random.normal(loc=y1m, scale=y1s, size=1)
        y2 = np.random.normal(loc=y2m, scale=y2s, size=1)
        y3 = np.random.normal(loc=y3m, scale=y3s, size=1)
        y1 = np.maximum(1.0, np.minimum(7.0, y1))[0]
        y2 = np.maximum(1.0, np.minimum(7.0, y2))[0]
        y3 = np.maximum(1.0, np.minimum(7.0, y3))[0]
    elif corr_trgt == 1:
        # simulate correlated random scores
        Y = np.random.standard_normal((1, 3))
        Y = np.dot(Y, np.linalg.cholesky(y_rho).T)[0]
        y1 = np.maximum(1.0, np.minimum(7.0, Y[0] * y1s + y1m))
        y2 = np.maximum(1.0, np.minimum(7.0, Y[1] * y2s + y2m))
        y3 = np.maximum(1.0, np.minimum(7.0, Y[2] * y3s + y3m))
    return y1, y2, y3

