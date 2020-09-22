import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions import MSE, R2Score
from functions import OLS, Ridge, variance_beta
from functions import FrankeFunction, design_matrix



def cross_validation(n, maxdegree, noise, n_folds, method):
    polydegree = np.zeros(maxdegree)
    MSE_mean = np.zeros(maxdegree)

    # Make data
    print(f"n = {n}")
    np.random.seed(101)
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    x_train, y_train = np.meshgrid(x_train, y_train)
    x_test, y_test = np.meshgrid(x_test, y_test)

    # Franke Function
    z_train = np.ravel(FrankeFunction(x_train, y_train) + noise*np.random.randn(x_train.shape[0], x_train.shape[0]))
    z_test = np.ravel(FrankeFunction(x_test, y_test) + noise*np.random.randn(np.shape(x_test)[0], np.shape(x_test)[0]))

    for degree in range(maxdegree):
        polydegree[degree] = degree

        # Create design matrix
        X_train = design_matrix(x_train, y_train, degree)

        # Scale data
        scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
        scaler.fit(X_train)                         # compute the mean and std to be used for later scaling
        X_train_scaled = scaler.transform(X_train)  # perform standardization by centering and scaling
        # X_test_scaled = scaler.transform(X_test)    # fit to data, then transform it

        # Set the first column to 1 since StandardScaler sets it to 0
        X_train_scaled[:, 0] = 1
        # X_test_scaled[:, 0] = 1

        # Shuffle data to get random folds
        index = np.arange(0, np.shape(X_train_scaled)[0], 1)
        np.random.shuffle(index)
        X_train_scaled_random = X_train_scaled[index]
        z_train_random = z_train[index]

        # Split data in n_folds folds
        X_folds = np.array(np.array_split(X_train_scaled_random, n_folds))
        z_folds = np.array(np.array_split(z_train_random, n_folds))

        # cross validation
        for k in range(n_folds):
            # Validation data
            X_val = X_folds[k]
            z_val = z_folds[k]

            # Training data
            idx = np.ones(n_folds, dtype=bool)
            idx[k] = False
            X_train_fold = X_folds[idx]
            X_train_fold = np.reshape(X_train_fold, (X_train_fold.shape[0]*X_train_fold.shape[1], X_train_fold.shape[2]))
            z_train_fold = np.ravel(z_folds[idx])


            beta_fold = method(X_train_fold, z_train_fold)
            z_tilde_fold = X_val @ beta_fold
            MSE_mean[degree] = MSE(z_val, z_tilde_fold)
            print(MSE_mean[degree])
        MSE_mean[degree] /= n_folds

    best_degree = list(MSE_mean).index(np.min(MSE_mean))
    print(best_degree)

    X_train_best = design_matrix(x_train, y_train, best_degree)
    scaler.fit(X_train_best)
    X_train_best_scaled = scaler.transform(X_train_best)
    X_test_best = design_matrix(x_test, y_test, best_degree)
    # scaler.fit(X_test_best)



    # SJEKK HVA DU SKAL SKALERE MED



    X_test_best_scaled = scaler.transform(X_test_best)
    beta_best = OLS(X_train_best_scaled, z_train)
    z_tilde_best = X_test_best_scaled @ beta_best
    MSE_best = MSE(z_test, z_tilde_best)
    print(MSE_best)

    return polydegree, MSE_mean, MSE_best

if __name__ == '__main__':
    # initial data
    n = 50            # number of data points
    maxdegree = 30
    noise = 0.1
    n_folds = 5               # number of folds

    polydegree, MSE_mean, MSE_best = cross_validation(n, maxdegree, noise, n_folds, method)
    plt.plot(polydegree, MSE_mean)
    plt.show()