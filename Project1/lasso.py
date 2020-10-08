import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from functions import Ridge
from bootstrap import bootstrap
from cross_validation import cross_validation

# initial data
n = 30            # number of data points
maxdegree = 23
noise = 0.1
n_folds = 5             # number of folds
n_bootstrap = 50
method = 'Lasso'
lmbda = np.logspace(-4, 1, 6)
seed = 7053

lasso_heatmap_cv = np.zeros((maxdegree, len(lmbda)))
lasso_heatmap_cv_sklearn = np.zeros((maxdegree, len(lmbda)))
lasso_heatmap_bootstrap = np.zeros((maxdegree, len(lmbda)))
for i in range(len(lmbda)):
    polydegree, MSE_mean, MSE_best, R2Score_skl, R2Score_mean, beta_best, best_degree, MSE_mean_sklearn, best_degree_sklearn, beta_best_sklearn = cross_validation(n, maxdegree, noise, n_folds, method, lmbda[i], seed)
    # polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, seed, lmbda=lmbda[i])
    plt.plot(polydegree, MSE_mean, label=f"CV, $\lambda$ = {lmbda[i]}")
    # plt.plot(polydegree, MSE_bootstrap_test, label=f"bootstrap, $\lambda$ = {lmbda[i]}")

    lasso_heatmap_cv[:, i] = MSE_mean
    lasso_heatmap_cv_sklearn[:, i] = MSE_mean_sklearn
    # lasso_heatmap_bootstrap[:, i] = MSE_bootstrap_test
plt.legend()
plt.title("Cross validation")
plt.xlabel('MSE')
plt.show()


heatmap = sb.heatmap(lasso_heatmap_cv, annot=True, cmap='viridis_r', \
                     xticklabels=lmbda, \
                     cbar_kws={'label': 'MSE'})
heatmap.set_xlabel('$\lambda$', size=12)
heatmap.set_ylabel('Model complexity', size=12)
heatmap.invert_xaxis()
heatmap.set_title('Heatmap made for lasso using cross validation', size=18)
plt.show()

heatmap = sb.heatmap(lasso_heatmap_cv_sklearn, annot=True, cmap='viridis_r', \
                     xticklabels=lmbda, \
                     cbar_kws={'label': 'MSE'})
heatmap.set_xlabel('$\lambda$', size=12)
heatmap.set_ylabel('Model complexity', size=12)
heatmap.invert_xaxis()
heatmap.set_title('Heatmap made for lasso using Scikit Learn CV', size=18)
plt.show()

heatmap = sb.heatmap(lasso_heatmap_bootstrap, annot=True, cmap='viridis_r', \
                     xticklabels=lmbda, \
                     cbar_kws={'label': 'MSE'})
heatmap.set_xlabel('$\lambda$', size=12)
heatmap.set_ylabel('Model complexity', size=12)
heatmap.invert_xaxis()
heatmap.set_title('Heatmap made for lasso using bootstrap', size=18)
plt.show()
