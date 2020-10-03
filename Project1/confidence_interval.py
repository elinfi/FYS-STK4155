import numpy as np
import functions as f
import matplotlib.pyplot as plt
from p1_a import no_resampling

n = 100
maxdegree = 5
noise = 0.1
seed = 4018

# Make data
np.random.seed(seed)
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)

# Franke Function
z = np.ravel(f.FrankeFunction(x, y) + noise*np.random.randn(n, n))

for degree in range(maxdegree):
    #Create design matrix
    X = f.design_matrix(x, y, degree)

    X_train, X_test, z_train, z_test = train_test_split(X, z)

    # Ordinary least squares
    beta_OLS = f.OLS(X, z)
    ztilde = X @ beta_OLS

    # Confidence interval as function of beta
    var_beta = f.variance_beta(X, noise)
    err_beta = 1.96*np.sqrt(var_beta)           # 95% confidence interval
    beta_idx = np.linspace(0, X.shape[1] - 1, X.shape[1])


    # Plot of confidence interval for OLS
    plt.style.use('ggplot')
    plt.errorbar(beta_idx, beta_OLS, err_beta, fmt='.')
    plt.xlabel("n", size=12)
    plt.ylabel('Confidence interval', size=12)
    plt.title("95 % confidence interval as function of $\\beta$", size=16)
    plt.show()
