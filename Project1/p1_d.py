import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

def design_matrix(x, y, p):
    # make sure x and y  are 1D
    if (len(x.shape)) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((p + 1)*(p + 2)/2)      # number of columns/terms
    X = np.ones((N, l))

    for i in range(1, p + 1):
        q = int(i*(i + 1)/2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)

    return X

def ridge(lmbda, X, y, p):
    # Ridge regression
    l = int((p + 1)*(p + 2)/2)
    beta_ridge = np.linalg.inv(X.T @ X + np.identity(10)*lmbda) @ (X.T @ y)
    return beta_ridge

# Make data
n = 1000
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))

#Create design matrix
p = 3
X = design_matrix(x, y, p)


# Franke Function
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)


beta = ridge(1e-4, X, z, p)
ztilde = X @ beta






# Start figure
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface
surf = ax.plot_surface(x, y, ztilde, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a colar bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
