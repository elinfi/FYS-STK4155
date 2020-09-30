import numpy as np
import functions as f
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def bootstrap(n, maxdegree, n_bootstrap, noise, method, seed=130, datatype='Franke', lmbda=0, filename='SRTM_data_Minneapolis.tif'):
    MSE_bootstrap_test = np.zeros(maxdegree)
    MSE_bootstrap_train = np.zeros(maxdegree)
    bias_bootstrap = np.zeros(maxdegree)
    variance_bootstrap = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)

    # Make data
    print(f"n = {n}")
    np.random.seed(seed)

    if datatype == 'Franke':
        x, y, z = f.FrankeDataBootstrap(n, noise)
        print(max(z))

    elif datatype == 'Terrain':
        x, y, z = f.TerrainDataBootstrap(n, filename)


    for degree in range(maxdegree):
        # print(f"degree = {degree}")
        polydegree[degree] = degree

        #Create design matrix
        X = f.design_matrix(x, y, degree)

        # Split in training and test data
        X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size = 0.3)

        # Scale data by subtracting the mean
        scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
        scaler.fit(X_train)                         # compute the mean and std to be used for later scaling
        X_train_scaled = scaler.transform(X_train)  # perform standardization by centering and scaling
        X_test_scaled = scaler.transform(X_test)    # fit to data, then transform it
        scaler.fit(z_train)
        z_train_scaled = scaler.transform(z_train)
        z_test_scaled = scaler.transform(z_test)

        # Set the first column to 1 since StandardScaler sets it to 0
        X_train_scaled[:, 0] = 1
        X_test_scaled[:, 0] = 1

        # Bootstrap
        z_tilde_test = np.zeros((np.shape(z_test_scaled)[0], n_bootstrap))
        z_tilde_train = np.zeros((np.shape(z_train_scaled)[0], n_bootstrap))
        index = np.arange(0, X_train.shape[0], 1)
        for b in range(n_bootstrap):
            idx_ = resample(index)
            z_train_scaled_ = z_train_scaled[idx_]
            X_train_scaled_ = X_train_scaled[idx_, :]

            if method == f.OLS:
                beta_bootstrap = method(X_train_scaled_, z_train_scaled_)
                z_tilde_test[:, b] = np.ravel(X_test_scaled @ beta_bootstrap)
                z_tilde_train[:, b] = np.ravel(X_train_scaled @ beta_bootstrap)

            elif method == f.Ridge:
                beta_bootstrap = method(X_train_scaled_, z_train_scaled_, lmbda, degree)
                z_tilde_test[:, b] = np.ravel(X_test_scaled @ beta_bootstrap)
                z_tilde_train[:, b] = np.ravel(X_train_scaled @ beta_bootstrap)

            elif method == 'Lasso':
                clf_lasso = skl.Lasso(alpha = lmbda, fit_intercept=False).fit(X_train_scaled, z_train_scaled_)
                z_tilde_test[:, b] = clf_lasso.predict(X_test_scaled)
                z_tilde_train[:, b] = clf_lasso.predict(X_train_scaled)

        MSE_bootstrap_test[degree] = np.mean(np.mean((z_test_scaled - z_tilde_test)**2, axis=1, keepdims=True))
        MSE_bootstrap_train[degree] = np.mean(np.mean((z_train_scaled - z_tilde_train)**2, axis=1, keepdims=True))
        bias_bootstrap[degree] = np.mean((z_test - np.mean(z_tilde_test, axis=1, keepdims=True))**2)
        variance_bootstrap[degree] = np.mean( np.var(z_tilde_test, axis=1, keepdims=True))

    return polydegree, MSE_bootstrap_test, MSE_bootstrap_train, \
           bias_bootstrap, variance_bootstrap

if __name__ == '__main__':
    # initial data
    n = 50
    maxdegree = 30
    n_bootstrap = 5
    noise = 0.1
    method = f.OLS
    lmbda = 0
    seed = 130

    polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, \
    variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed)

    # plt.plot(polydegree, MSE_bootstrap_test, label='MSE test')
    # plt.plot(polydegree, MSE_bootstrap_train, label='MSE train')
    plt.plot(polydegree, bias_bootstrap, label='bias')
    plt.plot(polydegree, variance_bootstrap, label='variance')
    plt.xlabel("Model complexity")
    plt.ylabel("bias-variance trade-off")
    plt.legend()
    plt.show()
