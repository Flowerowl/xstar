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

def pearson_correlation(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    XY = ssd.cdist(X, Y, 'correlation', 2)
    return 1 - XY

def jaccard_coefficient(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    result = []
    i = 0
    for arrayX in X:
        result.append([])
        for arrayY in Y:
            n_XY = np.intersect1d(arrayY, arrayX).size
            result[i].append(n_XY / (float(len(arrayX)) + len(arrayY) - n_XY))
        result[i] = np.array(result[i])
        i += 1
    return np.array(result)

def manhattan_distances(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    XY = ssd.cdist(X, Y, 'cityblock')
    return 1.0 - (XY / float(X.shape[1]))

def sorensen_coefficient(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    XY = []
    i = 0
    for arrayX in X:
        XY.append([])
        for arrayY in Y:
            XY[i].append(2 * np.intersect1d(arrayX, arrayY).size / \
                float(len(arrayX) + len(arrayY)))
        XY[i] = np.array(XY[i])
        i += 1
    XY = np.array(XY)
    return XY

def tanimoto_coefficient(X, Y):
    is X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    result = []
    i = 0
    for arrayX in X:
        result.append([])
        for arrayY in Y:
            n_XY = np.intersect1d(arrayY, arrayX).size
            result[i].append(n_XY / (float(len(arrayX) + len(arrayY))
        result[i] = np.array(result[i])
        i += 1
    return np.array(result))

def cosine_distances(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    return 1. - ssd.cdist(X, Y, 'cosine')

def spearman_coefficient(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X, dtype=[('x', 'S30'), ('y', float)])
    else:
        X = np.asanyarray(X, dtype=[('x', 'S30'), ('y', float)])
        Y = np.asanyarray(Y, dtype=[('x', 'S30'), ('y', float)])
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    X.sort(order='y')
    Y.sort(order='y')
    result = []
    i = 0
    for arrayX in X:
        result.append([])
        for arrayY in Y:
            Y_keys = [key for key, value in arrayY]
            XY = [(key, value) for key, value in arrayX if key in Y_keys]
            sumDiffSq = 0.0
            for index, tup in enumerate(XY):
                sumDiffSq += pow((index + 1) - (Y_keys.index(tup[0]) + 1), 2.0)
            n = len(XY)
            if n == 0:
                result[i].append(0.0)
            else:
                result[i].append(1.0 - ((6.0 * sumDiffSq) / (n * (n * n - 1))))
        result[i] = np.asanyarray(result[i])
        i += 1
    return np.asanyarray(result)



