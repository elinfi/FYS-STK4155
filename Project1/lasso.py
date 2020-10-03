import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from bootstrap import bootstrap
from cross_validation import cross_validation

# initial data
n = 30            # number of data points
maxdegree = 25
noise = 0.1
n_folds = 5             # number of folds
n_bootstrap = 5
method = 'Lasso'
# lmbda_arr = np.array([0.005, 0.01, 0.05, 0.1, 0.5])
lmbda_arr = np.array([0.0001])
seed = 130

for lmbda in lmbda_arr:
    polydegree, MSE_mean, MSE_best, R2Score_skl, R2Score_mean = cross_validation(n, maxdegree, noise, n_folds, method, seed, lmbda)
    # polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed, lmbda=lmbda)
    plt.plot(polydegree, MSE_mean, label=f"CV, $\lambda$ = {lmbda}")
    # plt.plot(polydegree, MSE_bootstrap_test, label=f"bootstrap, $\lambda$ = {lmbda}")
plt.legend()
plt.show()

for lmbda in lmbda_arr:
    polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed, lmbda=lmbda)
    # plt.subplot(311)
    plt.plot(polydegree, MSE_bootstrap_test, label=f"MSE, $\lambda$ = {lmbda}")
    plt.xlabel("Model complexity")
    plt.ylabel("MSE")
    plt.legend()

    # plt.subplot(211)
    # plt.plot(polydegree, bias_bootstrap, label=f"bias, $\lambda$ = {lmbda}")
    # plt.xlabel("Model complexity")
    # plt.ylabel("Bias")
    # plt.legend()
    #
    # plt.subplot(212)
    # plt.plot(polydegree, variance_bootstrap, label=f"var, $\lambda$ = {lmbda}")
    # plt.xlabel("Model complexity")
    # plt.ylabel("Variance")
    # plt.legend()

plt.legend()
plt.show()
