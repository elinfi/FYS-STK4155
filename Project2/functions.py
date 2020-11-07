import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def FrankeFunction(x,y):
    """Computes the franke function.

    Keyword arguments:
    x -- x coordinates
    y -- y coordinates

    Return value:
    term1 + term2 + term3 + term4 -- Franke function
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def MSE(z, ztilde):
    """Computes mean squared error.

    Keyword arguments:
    z -- true value
    z_tilde -- predicted value

    Return value:
    MSE -- mean squared error
    """
    MSE = np.mean((z - ztilde)**2)
    return MSE

def OLS(X, z):
    """Computed ordinary least squares.

    Keyword arguments:
    X -- design matrix
    z -- true value

    Return value:
    beta -- ordinary least squared coefficients
    """
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ z)
    return beta
