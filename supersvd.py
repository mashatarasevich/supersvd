# vim: ts=4 sts=4 et
import numpy as _np
from scipy.sparse import linalg as _spla
from collections import namedtuple

SuperSvdResult = namedtuple('SuperSvdResult', [
    'x_coeff', 'y_coeff', 'corrcoeff', 
    'x_variance_fraction', 'y_variance_fraction',
    'x_vect', 'y_vect', 'eigenvalue_fraction', 'eigenvalues',
    ])

def supersvd(X, Y, k=3, eliminate_mean=True):
    """
    X and Y - the input data for which correlation is seeked
    dim(X) = nT x nX
    dim(Y) = nT x nY
    nX and nY may be multidimensional

    k is the number of seeked pairs

    Each array is decomposed into

    X[t, i] = Xm[i] + sqrt(size(X)) sum_e XC[e, t] XV[e, i]
    Y[t, j] = Ym[j] + sqrt(size(Y)) sum_e YC[e, t] YV[e, j]
    Xm[i] = mean(X[t, i], t)
    Ym[j] = mean(Y[t, j], t)

    ||YC[e, :]||_2 = ||XC[e, :]||_2 = 1 for each e

    XV and YV form an orthogonal basis, i.e.

    sum_i XV[e, i] XV[e', i] = 0 
    sum_j YV[e, j] YV[e', j] = 0
    when e != e'

    Returns 
        XC: k x nT
        YC: k x nT
        XV: k x nX
        YV: k x nY

        EF: k, fraction of covariance, explained by k-th pair
    """
    nTX, *nX = X.shape
    nTY, *nY = Y.shape
    assert nTX == nTY

    nT = nTX
    Xorig = X
    Yorig = Y
    X = Xorig.reshape(nT, -1)
    Y = Yorig.reshape(nT, -1)
    if eliminate_mean:
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

    # Norming makes eigenvalues ~O(1)
    COV = (X.T @ Y) / nT / (X.shape[1] * Y.shape[1])**0.25

    U, S, Vt = _spla.svds(COV, k=k)
    perm = _np.argsort(-S)
    S = S[perm]
    XV = U.T[perm, :]
    YV = Vt [perm, :]

    XC = (XV @ X.T) / _np.sqrt(X.size)
    YC = (YV @ Y.T) / _np.sqrt(Y.size)

    xnorm = _np.linalg.norm(XC, axis=1)
    ynorm = _np.linalg.norm(YC, axis=1)

    XC /= xnorm.reshape(-1, 1)
    YC /= ynorm.reshape(-1, 1)

    XV *= xnorm.reshape(-1, 1) 
    YV *= ynorm.reshape(-1, 1)

    S2 = S**2
    EF = S2 / _np.linalg.norm(COV, 'fro')**2

    CORR = _np.einsum('ij,ij->i', XC, YC)

    Xvar = _np.var(X)
    Yvar = _np.var(Y)

    Xvar_frac = _np.zeros_like(CORR)
    Yvar_frac = _np.zeros_like(CORR)

    for e in range(k):
        Xvar_frac[e] = _np.var(_np.sqrt(X.size) * _np.outer(XC[e, :], XV[e, :]))
        Yvar_frac[e] = _np.var(_np.sqrt(Y.size) * _np.outer(YC[e, :], YV[e, :]))

    Xvar_frac /= Xvar
    Yvar_frac /= Yvar

    return SuperSvdResult(
        x_coeff=XC,
        y_coeff=YC,
        corrcoeff=CORR,
        x_variance_fraction=Xvar_frac,
        y_variance_fraction=Yvar_frac,
        x_vect=XV.reshape(k, *nX),
        y_vect=YV.reshape(k, *nY),
        eigenvalue_fraction=EF,
        eigenvalues=S2,
    )
