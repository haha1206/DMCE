from sklearn.metrics.pairwise import euclidean_distances as EuDist2
import numpy as np

def KNN(X,knn):
    eps = 2.2204e-16
    n, dim = X.shape
    D = EuDist2(X, X, squared=True)
    NN_full = np.argsort(D, axis=1)
    W = np.zeros((n,n))
    for i in range(n):
        id = NN_full[i,1:(knn+2)]
        di = D[i,id]
        W[i,id] = (di[-1]-di)/(knn*di[-1]-sum(di[:-1])+eps)
    A = (W+W.T)/2
    return A

def norm_W(A):
    d = np.sum(A, 1)
    d[d == 0] = 1e-6
    d_inv = 1 / np.sqrt(d)
    tmp = A * np.outer(d_inv, d_inv)
    A2 = np.maximum(tmp, tmp.T)
    return A2







