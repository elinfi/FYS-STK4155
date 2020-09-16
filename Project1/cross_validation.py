import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions import MSE, R2Score
from functions import OLS, Ridge, variance_beta
from functions import FrankeFunction, design_matrix

# initial data
n = 4              # number of data points
maxdegree = 15
n_bootstrap = 100
noise = 0.1
n_folds = 5               # number of folds

polydegree = np.zeros(maxdegree)
MSE_folds = np.zeros(maxdegree)

# Make data
print(f"n = {n}")
np.random.seed(101)
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)

# Franke Function
z = np.ravel(FrankeFunction(x, y) + noise*np.random.randn(n, n))

for degree in range(maxdegree):
    # print(f"degree = {degree}")
    polydegree[degree] = degree

    #Create design matrix
    X = design_matrix(x, y, degree)

    print(f"X: {np.shape(X)}")
    # Split in training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size = 0.3)
    print(f"X_test: {np.shape(X_test)}")
    print(f"X_train {np.shape(X_train)}")
    # Scale data by subtracting the mean
    scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
    scaler.fit(X_train)                         # compute the mean and std to be used for later scaling
    X_train_scaled = scaler.transform(X_train)  # perform standardization by centering and scaling
    X_test_scaled = scaler.transform(X_test)    # fit to data, then transform it

    # Set the first column to 1 since StandardScaler sets it to 0
    X_train_scaled[:, 0] = 1
    X_test_scaled[:, 0] = 1

    index = np.arange(0, np.shape(X_train_scaled)[0], 1)
    np.random.shuffle(index)
    X_train_scaled_random = X_train_scaled[index]
    z_train_random = z_train[index]
    X_folds = np.array(np.array_split(X_train_scaled_random, n_folds))
    z_folds = np.array(np.array_split(z_train_random, n_folds))

    for k in range(n_folds):
        X_val = X_folds[k]
        z_val = z_folds[k]

        idx = np.ones(n_folds, dtype=bool)
        idx[k] = False
        X_train_fold = X_folds[idx]
        X_train_fold.flatten
        z_train_fold = z_folds[idx]
        z_train_fold.flatten

        beta_fold = OLS(X_train_fold, z_train_fold)
        z_tilde_fold = X_val @ beta_fold
        MSE_fold[degree] += MSE(z_val, z_tilde_fold)
MSE_fold /= n_folds
