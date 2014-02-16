#encoding:utf-8


import numpy as np
import scipy.spatial.distance as ssd


def euclidean_distances(X, Y, squared=False, inverse=True):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    if squared:
        return ssd.cdist(X, Y, 'sqeuclidean')
    XY = ssd.cdist(X, Y)
    return np.divide(1.0, (1.0+XY)) if inverse else XY

euclidean_distances = euclidean_distances

def pearson_distances(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    XY = ssd.cdist(X, Y, 'correlation', 2)
    return 1 - XY


