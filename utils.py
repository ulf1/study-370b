import numpy as np


def get_random_mos(y1m, y1s, y2m, y2s, y3m, y3s,
                   adjust=1.0, rho=None, corr_trgt=0):
    num = 1 if isinstance(y1m, float) else len(y1m)
    Y = np.random.standard_normal((num, 3))
    Y = adjust * Y
    if (rho is not None) and (corr_trgt == 1):
        Y = np.dot(Y, np.linalg.cholesky(rho).T)
    y1 = np.maximum(1.0, np.minimum(7.0, Y[:, 0] * y1s + y1m))
    y2 = np.maximum(1.0, np.minimum(7.0, Y[:, 1] * y2s + y2m))
    y3 = np.maximum(1.0, np.minimum(7.0, Y[:, 2] * y3s + y3m))
    return y1, y2, y3

