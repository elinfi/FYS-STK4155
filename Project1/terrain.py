import numpy as np
import functions as f
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import  cm
from bootstrap import bootstrap
from mpl_toolkits.mplot3d import Axes3D
from cross_validation import cross_validation
from sklearn.preprocessing import StandardScaler

# Load the terrain
filename = 'SRTM_data_Kamloops.tif'
terrain = imread(filename)

# Show the terrain
plt.figure()
plt.title('Terrain over Minneapolis')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Initial data
n = 100
maxdegree = 23
n_bootstrap = 100
n_folds = 5
noise = 0.1
method = f.OLS
lmbda = 0.1

# polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, \
# variance_bootstrap = bootstrap(n, maxdegree, n_bootstrap, noise, method, \
#                                datatype='Terrain', filename=filename)
polydegree_cv, MSE_mean, MSE_best, R2Score_skl, R2Score_mean, beta_best, best_degree \
    = cross_validation(n, maxdegree, noise, n_folds, method, lmbda=lmbda, \
                       datatype='Terrain', filename=filename)

# plt.plot(polydegree, MSE_bootstrap_test, label=f"MSE bootstrap")
# plt.plot(polydegree, MSE_mean, label=f"MSE cross validation")
# plt.xlabel('Model complexity')
# plt.ylabel('MSE')
# plt.legend()
# plt.close()



# Normalize data
scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
scaler.fit(terrain)                         # compute the mean and std to be used for later scaling
terrain_scaled = scaler.transform(terrain)  # perform standardization by centering and scaling

# Fixing a set of points
terrain_scaled = terrain_scaled[:n, :n]

# Create mesh of image pixel
x = np.sort(np.linspace(0, 1, terrain_scaled.shape[0]))
y = np.sort(np.linspace(0, 1, terrain_scaled.shape[1]))
x, y = np.meshgrid(x, y)
X = f.design_matrix(x, y, best_degree)
z_tilde = X @ beta_best
z_tilde = z_tilde.reshape(x.shape[0], x.shape[1])
print(f.MSE(terrain_scaled, z_tilde))


plt.subplot(121)
plt.imshow(terrain_scaled, cmap='gist_rainbow')

plt.subplot(122)
plt.imshow(z_tilde, cmap='gist_rainbow')
plt.show()

fig, ax = plt.subplots(1, 2)
cp1 = ax[0].contour(x, y, terrain_scaled)
fig.colorbar(cp1)

cp2 = ax[1].contour(x, y, z_tilde)
fig.colorbar(cp2)
plt.show()
