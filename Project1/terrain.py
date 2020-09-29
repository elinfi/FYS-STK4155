import numpy as np
import functions as f
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import  cm
from bootstrap import bootstrap
from cross_validation import cross_validation
from mpl_toolkits.mplot3d import Axes3D

# Load the terrain
filename = 'SRTM_data_Minneapolis.tif'
terrain = imread(filename)

# Show the terrain
plt.figure()
plt.title('Terrain over Minneapolis')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.close()


# Initial data
n = 50
maxdegree = 10
n_bootstrap = 5
n_folds = 5
noise = 0.1
method = f.OLS
lmbda = 0.1

polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, \
variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, \
                               datatype='Terrain', filename=filename)
polydegree_cv, MSE_mean, MSE_best, R2Score_skl, R2Score_mean \
    = cross_validation(n, maxdegree, noise, n_folds, method, lmbda=lmbda, \
                       datatype='Terrain', filename=filename)

plt.plot(polydegree, MSE_bootstrap_test, label=f"MSE bootstrap")
plt.plot(polydegree, MSE_mean, label=f"MSE cross validation")
plt.xlabel('Model complexity')
plt.ylabel('MSE')
plt.legend()
plt.show()
