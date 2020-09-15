import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions import MSE, R2Score
from functions import OLS, Ridge, variance_beta
from functions import FrankeFunction, design_matrix

# initial data
n = 100
maxdegree = 30
n_bootstrap = 10

MSE_bootstrap = np.zeros(maxdegree)
bias_bootstrap = np.zeros(maxdegree)
variance_bootstrap = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

# Make data
print(f"n = {n}")
np.random.seed(101)
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)

# Franke Function
noise = 0.8*np.random.randn(n, n)
z = np.ravel(FrankeFunction(x, y) + noise)


for degree in range(maxdegree):
    print(f"degree = {degree}")
    polydegree[degree] = degree

    #Create design matrix
    X = design_matrix(x, y, degree)

    # Split in training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size = 0.3)

    # Scale data by subtracting the mean
    scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
    scaler.fit(X_train)                         # compute the mean and std to be used for later scaling
    X_train_scaled = scaler.transform(X_train)  # perform standardization by centering and scaling
    X_test_scaled = scaler.transform(X_test)    # fit to data, then transform it

    # Set the first column to 1 since StandardScaler sets it to 0
    X_train_scaled[:, 0] = 1
    X_test_scaled[:, 0] = 1

    # Bootstrap
    z_tilde_bootstrap = np.zeros((np.shape(z_test)[0], n_bootstrap))
    for b in range(n_bootstrap):
        z_ = resample(z_train)
        beta_bootstrap = OLS(X_train_scaled, z_)
        z_tilde_bootstrap[:, b] = np.ravel(X_test_scaled @ beta_bootstrap)

    MSE_bootstrap[degree] = np.mean(np.mean((z_test - z_tilde_bootstrap)**2, axis=1, keepdims=True))
    bias_bootstrap[degree] = np.mean((z_test - np.mean(z_tilde_bootstrap, axis=1, keepdims=True))**2)
    variance_bootstrap[degree] = np.mean( np.var(z_tilde_bootstrap, axis=1, keepdims=True))

plt.plot(polydegree, MSE_bootstrap, label='MSE')
plt.plot(polydegree, bias_bootstrap, label='bias')
plt.plot(polydegree, variance_bootstrap, label='variance')
plt.show()
