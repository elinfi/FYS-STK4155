import numpy as np
from numba import jit
from sklearn.utils import resample

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

# @jit
def design_matrix(x, y, p):
    # make sure x and y  are 1D
    if (len(x.shape)) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((p + 1)*(p + 2)/2)      # number of columns/terms
    X = np.ones((N, l))

    for i in range(1, p + 1):
        q = int(i*(i + 1)/2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)
    return X

def OLS(X, z):
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ z)
    return beta

def Ridge(X, z, lmbda, degree):
    # number of combinations of x and y
    l = int((degree + 1)*(degree + 2)/2)
    beta_ridge = np.linalg.inv(X.T @ X + np.identity(l)*lmbda) @ (X.T @ z)
    return beta_ridge

# def Lasso()

def bootstrap(n_bootstrap, X_train, X_test, z_test, z_train, method):
    z_tilde_bootstrap = np.zeros((np.shape(z_test)[0], n_bootstrap))
    for b in range(n_bootstrap):
        z_ = resample(z_train)
        beta_bootstrap = method(X_train, z_)
        z_tilde_bootstrap[:, b] = np.ravel(X_test @ beta_bootstrap)
    return z_tilde_bootstrap

def variance_beta(beta, X, noise):
    var_beta = 0.8*np.diag(np.linalg.pinv(X.T @ X))
    return var_beta

def MSE(z, ztilde):
    MSE = np.mean((z - ztilde)**2)
    return MSE

def R2Score(z, ztilde):
    R2Score = 1 - np.sum((z - ztilde)**2)/np.sum((z - np.mean(z))**2)
    return R2Score
