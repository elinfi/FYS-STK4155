import numpy as np
import matplotlib.pyplot as plt
from bootstrap import bootstrap
from cross_validation import cross_validation
from functions import OLS

# initial data
n = 50                    # number of data points
maxdegree = 30
noise = 0.1
n_folds = 5               # number of folds
n_bootstrap = 30
method = OLS
lmbda = 0
seed = 133

polydegree_cv, MSE_mean, MSE_best, R2Score_skl, R2Score_mean = cross_validation(n, maxdegree, noise, n_folds, method, lmbda, seed)
polydegree_b, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, lmbda, seed)

plt.plot(polydegree_cv, MSE_mean, label='cross validation')
plt.plot(polydegree_b, MSE_bootstrap_test, label='bootstrap')
plt.legend()
plt.show()
