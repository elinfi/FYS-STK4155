import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

def FrankeData(n, noise, degree, scaling=False, test_size=0.3):
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))

    x, y = np.meshgrid(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size, \
                                                        random_state=7053)

    # Franke Function
    z_train = np.ravel(FrankeFunction(x_train, y_train) \
                       + noise*np.random.randn(x_train.shape[0], \
                                               x_train.shape[1]))
    z_test = np.ravel(FrankeFunction(x_test, y_test) \
                      + noise*np.random.randn(x_test.shape[0], x_test.shape[1]))
    z_train = z_train.reshape(-1, 1)
    z_test = z_test.reshape(-1, 1)

    # Create design matrix
    X_train = design_matrix(x_train, y_train, degree)
    X_test = design_matrix(x_test, y_test, degree)

    if scaling:
        scaler = StandardScaler()

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaler.fit(z_train)
        z_train_scaled = scaler.transform(z_train)
        z_test_scaled = scaler.transform(z_test)

        # Set the first column to 1 since StandardScaler sets it to 0
        X_train_scaled[:, 0] = 1
        X_test_scaled[:, 0] = 1

        return X_train, X_test, z_train_scaled, z_test_scaled

    return X_train, X_test, z_train, z_test

def design_matrix(x, y, degree):
    # make sure x and y  are 1D
    if (len(x.shape)) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((degree + 1)*(degree + 2)/2)      # number of columns/terms
    X = np.ones((N, l))

    for i in range(1, degree + 1):
        q = int(i*(i + 1)/2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)
    return X

def MSE(z, ztilde):
    MSE = np.mean((z - ztilde)**2)
    return MSE

def OLS(X, z):
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ z)
    return beta
