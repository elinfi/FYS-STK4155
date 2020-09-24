import numpy as np
import matplotlib.pyplot as plt
from functions import Ridge
from bootstrap import bootstrap
from cross_validation import cross_validation

# initial data
n = 50            # number of data points
maxdegree = 15
noise = 0.8
n_folds = 5             # number of folds
n_bootstrap = 5
method = Ridge
lmbda_arr = np.array([1, 5, 10])
seed = 130

for lmbda in lmbda_arr:
    polydegree, MSE_mean, MSE_best, R2Score_skl, R2Score_mean = cross_validation(n, maxdegree, noise, n_folds, method, lmbda, seed)
    polydegree, MSE_bootstrap, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, lmbda, seed)
    plt.plot(polydegree, MSE_mean, label=f"CV, $\lambda$ = {lmbda}")
    # plt.plot(polydegree, MSE_bootstrap, label=f"bootstrap, $\lambda$ = {lmbda}")
plt.legend()
plt.title("Cross validation")
plt.show()

for lmbda in lmbda_arr:
    polydegree, MSE_bootstrap, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, lmbda, seed)
    # plt.subplot(311)
    plt.plot(polydegree, MSE_bootstrap, label=f"MSE, $\lambda$ = {lmbda}")
    plt.xlabel("Model complexity")
    plt.ylabel("MSE")
    plt.title("Bootstrap")
    plt.legend()

    plt.subplot(211)
    plt.plot(polydegree, bias_bootstrap, label=f"bias, $\lambda$ = {lmbda}")
    plt.xlabel("Model complexity")
    plt.ylabel("Bias")
    plt.legend()

    plt.subplot(212)
    plt.plot(polydegree, variance_bootstrap, label=f"var, $\lambda$ = {lmbda}")
    plt.xlabel("Model complexity")
    plt.ylabel("Variance")
    plt.legend()

plt.legend()
plt.show()
