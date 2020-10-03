import numpy as np
import matplotlib.pyplot as plt
from bootstrap import bootstrap
from functions import OLS, Ridge
from cross_validation import cross_validation

# initial data
n = 30                    # number of data points
maxdegree = 14
noise = 0.1
n_folds = 5               # number of folds
n_bootstrap = 100
lmbda = 0
seed = 7053

# Bootstrap OLS
polydegree_ols, MSE_bootstrap_test_ols, MSE_bootstrap_train_ols, \
bias_bootstrap_ols, variance_bootstrap_ols = \
            bootstrap(n, maxdegree, n_bootstrap, noise, method=OLS, seed=seed)

# Bootstrap Ridge
polydegree_b, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, \
variance_bootstrap = \
            bootstrap(n, maxdegree, n_bootstrap, noise, method=Ridge, seed=seed)

# plt.plot(polydegree_ols, bias_bootstrap_ols, label='OLS bias')
plt.style.use('ggplot')
plt.plot(polydegree_b, MSE_bootstrap_test, label='Ridge MSE')
plt.plot(polydegree_b, bias_bootstrap, label='Ridge bias')
plt.plot(polydegree_b, variance_bootstrap, label='Ridge var')
plt.xlabel('Model complexity', size=12)
plt.ylabel('Error', size=12)
plt.title('Bias-variance trade-off', size=18)
plt.legend()
plt.show()

plt.plot(polydegree_ols, MSE_bootstrap_test_ols, label='OLS MSE')
plt.plot(polydegree_ols, bias_bootstrap_ols, label='OLS bias')
plt.plot(polydegree_ols, variance_bootstrap_ols, label='OLS var')
plt.xlabel('Model complexity', size=12)
plt.ylabel('Error', size=12)
plt.title('Bias-variance trade-off', size=18)
plt.legend()
plt.show()


plt.plot(polydegree_ols, variance_bootstrap_ols, label='OLS var')
plt.plot(polydegree_b, variance_bootstrap, label='Ridge var')
plt.xlabel('Model complexity', size=12)
plt.ylabel('variance', size=12)
plt.title('Bias-variance trade-off', size=18)
plt.legend()
plt.show()
