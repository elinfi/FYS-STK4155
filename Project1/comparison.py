import numpy as np
import functions as f
import matplotlib.pyplot as plt
from p1_a import no_resampling
from bootstrap import bootstrap
from cross_validation import cross_validation
from time import time

# initial data
n = 30                   # number of data points
maxdegree = 23
noise = 0.8
n_folds = 5              # number of folds
n_bootstrap = 20
method = f.Ridge
lmbda = 0.001
seed = 7053

polydegree, MSE_train, MSE_test, MSE_train_scaled, MSE_test_scaled, R2Score_scaled = no_resampling(n, maxdegree, noise, method, seed=seed, lmbda=lmbda)
polydegree, MSE_train, MSE_test, MSE_train_scaled, MSE_test_scaled_OLS, R2Score_scaled = no_resampling(n, maxdegree, noise, method=f.OLS, seed=seed, lmbda=lmbda)

start = time()
polydegree_cv, MSE_mean, MSE_best, R2Score_skl, R2Score_mean, beta_best, best_degree = cross_validation(n, maxdegree, noise, n_folds, method, seed, lmbda)
end = time()
print(f"cv: {end - start}")

start = time()
polydegree_b, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed, lmbda=lmbda)
end = time()
print(f"bootstrap: {end - start}")

plt.style.use("ggplot")

plt.plot(polydegree, MSE_test_scaled, label='Ridge')
plt.plot(polydegree, MSE_test_scaled_OLS, label='OLS')
plt.xlabel('Model complexity', size=12)
plt.ylabel('MSE', size=12)
plt.title('Ridge vs OLS on noisy data', size=18)
plt.legend()
plt.show()

plt.plot(polydegree, MSE_test_scaled, label='no resampling')
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
