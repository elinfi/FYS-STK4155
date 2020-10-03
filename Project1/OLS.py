import numpy as np
import matplotlib.pyplot as plt
from functions import OLS
from p1_a import no_resampling

n = 30
maxdegree = 23
noise = 0.1
method = f.OLS
seed = 7053
n_folds = 5
n_bootstrap = 100

polydegree, MSE_train, MSE_test, MSE_train_scaled, MSE_test_scaled, R2Score_scaled = no_resampling(n, maxdegree, noise, method=method, seed=seed)
polydegree_b, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed)

plt.style.use("ggplot")

plt.plot(polydegree, MSE_test_scaled, label='test data')
plt.plot(polydegree, MSE_train_scaled, label='train data')
plt.xlabel("Model Complexity", size=12)
plt.ylabel("MSE", size=12)
plt.title('OLS without resampling', size=18)
plt.legend()
plt.show()

plt.plot(polydegree_b, MSE_bootstrap_test, label='test data')
plt.plot(polydegree_b, MSE_bootstrap_train, label='train data')
plt.xlabel("Model Complexity", size=12)
plt.ylabel("MSE", size=12)
plt.title('OLS without resampling', size=18)
plt.legend()
plt.show()
