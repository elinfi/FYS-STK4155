import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import functions as f
from bootstrap import bootstrap
from cross_validation import cross_validation

# initial data
n = 30            # number of data points
maxdegree = 23
noise = 0.1
n_folds = 5             # number of folds
n_bootstrap = 50
method = f.Ridge
lmbda = np.logspace(-4, 1, 6)
seed = 7053

ridge_heatmap_cv = np.zeros((maxdegree, len(lmbda)))
ridge_heatmap_bootstrap = np.zeros((maxdegree, len(lmbda)))
ridge_heatmap_cv_sklearn = np.zeros((maxdegree, len(lmbda)))
for i in range(len(lmbda)):
    polydegree, MSE_mean, MSE_best, R2Score_skl, R2Score_mean, beta_best, best_degree, MSE_mean_sklearn, best_beta_sklearn = cross_validation(n, maxdegree, noise, n_folds, method, lmbda[i], seed)
    # polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed, lmbda=lmbda[i])
    plt.plot(polydegree, MSE_mean, label=f"CV, $\lambda$ = {lmbda[i]}")
    plt.plot(polydegree, MSE_mean_sklearn, label=f"Sklearn, $\lambda$ = {lmbda[i]}")
    # plt.plot(polydegree, MSE_bootstrap_test, label=f"bootstrap, $\lambda$ = {lmbda[i]}")

    ridge_heatmap_cv[:, i] = MSE_mean
    ridge_heatmap_cv_sklearn[:, i] = MSE_mean_sklearn
    # ridge_heatmap_bootstrap[:, i] = MSE_bootstrap_test
plt.legend()
plt.title("Cross validation")
plt.xlabel('MSE')
plt.show()


heatmap = sb.heatmap(ridge_heatmap_cv, annot=True, cmap='viridis_r', \
                     xticklabels=lmbda, \
                     cbar_kws={'label': 'MSE'})
heatmap.set_xlabel('$\lambda$', size=12)
heatmap.set_ylabel('Model complexity', size=12)
heatmap.invert_xaxis()
heatmap.set_title('Heatmap made for ridge using cross validation', size=18)
plt.show()

heatmap = sb.heatmap(ridge_heatmap_cv_sklearn, annot=True, cmap='viridis_r', \
                     xticklabels=lmbda, \
                     cbar_kws={'label': 'MSE'})
heatmap.set_xlabel('$\lambda$', size=12)
heatmap.set_ylabel('Model complexity', size=12)
heatmap.invert_xaxis()
heatmap.set_title('Heatmap made for ridge using Scikit Learn CV', size=18)
plt.show()

heatmap = sb.heatmap(ridge_heatmap_bootstrap, annot=True, cmap='viridis_r', \
                     xticklabels=lmbda, \
                     cbar_kws={'label': 'MSE'})
heatmap.set_xlabel('$\lambda$', size=12)
heatmap.set_ylabel('Model complexity', size=12)
heatmap.invert_xaxis()
heatmap.set_title('Heatmap made for ridge using bootstrap', size=18)
plt.show()
