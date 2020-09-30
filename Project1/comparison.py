import numpy as np
import matplotlib.pyplot as plt
from bootstrap import bootstrap
from cross_validation import cross_validation
from functions import OLS
from time import time

# initial data
n = 30                    # number of data points
maxdegree = 23
noise = 0.1
n_folds = 5              # number of folds
n_bootstrap = 100
method = OLS
lmbda = 0
seed = 4018

start = time()
polydegree_cv, MSE_mean, MSE_best, R2Score_skl, R2Score_mean = cross_validation(n, maxdegree, noise, n_folds, method, seed, lmbda)
end = time()
print(f"cv: {end - start}")

start = time()
polydegree_b, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed)
end = time()
print(f"bootstrap: {end - start}")
plt.style.use("ggplot")
plt.plot(polydegree_cv, MSE_mean, label='cross validation')
plt.plot(polydegree_b, MSE_bootstrap_test, label='bootstrap')
plt.xlabel('Model complexity', size=12)
plt.ylabel('MSE', size=12)
plt.title('Error comparison for OLS', size=18)
plt.legend()
plt.show()

plt.plot(polydegree_b, MSE_bootstrap_test, label='Test data')
plt.plot(polydegree_b, MSE_bootstrap_train, label='Train data')
plt.xlabel('Model complexity', size=12)
plt.ylabel('MSE', size=12)
plt.title('OLS with bootstrap', size=18)
plt.legend()
plt.show()
