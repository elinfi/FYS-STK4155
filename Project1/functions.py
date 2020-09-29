import numpy as np
from numba import jit
from imageio import imread
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

def FrankeDataBootstrap(n, noise):
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x, y = np.meshgrid(x, y)

    # Franke Function
    z = np.ravel(FrankeFunction(x, y) + noise*np.random.randn(n, n))

    return x, y, z

def FrankeDataCV(n, noise, test_size=0.3):
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))

    x, y = np.meshgrid(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

    # Franke Function
    z_train = np.ravel(FrankeFunction(x_train, y_train) + noise*np.random.randn(x_train.shape[0], x_train.shape[1]))
    z_test = np.ravel(FrankeFunction(x_test, y_test) + noise*np.random.randn(x_test.shape[0], x_test.shape[1]))

    return x_train, x_test, y_train, y_test, z_train, z_test

def TerrainDataBootstrap(n, filename):
    # Load terrain data
    terrain = imread('SRTM_data_Minneapolis.tif')

    # Normalize data
    scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
    scaler.fit(terrain)                         # compute the mean and std to be used for later scaling
    terrain_scaled = scaler.transform(terrain)  # perform standardization by centering and scaling
    terrain_scaled = terrain_scaled[:n, :n]

    x = np.sort(np.linspace(0, 1, terrain_scaled.shape[0]))
    y = np.sort(np.linspace(0, 1, terrain_scaled.shape[1]))

    x, y = np.meshgrid(x, y)
    z = np.ravel(terrain_scaled)

    return x, y, z

def TerrainDataCV(n, filename, test_size=0.3):
    # Load terrain data
    terrain = imread('SRTM_data_Minneapolis.tif')

    # Normalize data
    scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
    scaler.fit(terrain)                         # compute the mean and std to be used for later scaling
    terrain_scaled = scaler.transform(terrain)  # perform standardization by centering and scaling

    # Fixing a set of points
    terrain_scaled = terrain_scaled[:n, :n]

    # Create mesh of image pixel
    x = np.sort(np.linspace(0, 1, terrain_scaled.shape[0]))
    y = np.sort(np.linspace(0, 1, terrain_scaled.shape[1]))
    x, y = np.meshgrid(x, y)

    # Split data in train and test
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, terrain_scaled, test_size = 0.3)
    z_train = np.ravel(z_train)
    z_test = np.ravel(z_test)

    print(x_train.shape)
    print(z_train.shape)
    print(z_test.shape)

    return x_train, x_test, y_train, y_test, z_train, z_test


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

def variance_beta(X, noise):
    var_beta = noise*np.diag(np.linalg.pinv(X.T @ X))
    return var_beta

def MSE(z, ztilde):
    MSE = np.mean((z - ztilde)**2)
    return MSE

def R2Score(z, ztilde):
    R2Score = 1 - np.sum((z - ztilde)**2)/np.sum((z - np.mean(z))**2)
    return R2Score
