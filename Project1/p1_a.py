import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

@jit
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

def OLS(X, z):
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ z)
    return beta

def ridge(lmbda, X, z, p):
    # Ridge regression
    l = int((p + 1)*(p + 2)/2)
    beta_ridge = np.linalg.pinv(X.T @ X + np.identity(10)*lmbda) @ (X.T @ z)
    return beta_ridge

def variance_beta(beta, X, noise):
    var_beta = np.diag(np.var(noise) * np.linalg.pinv(X.T @ X))
    return var_beta

def MSE(z, ztilde):
    MSE = np.sum((z - ztilde)**2)/len(z)
    return MSE

def R2Score(z, ztilde):
    R2Score = 1 - np.sum((z - ztilde)**2)/np.sum((z - np.mean(z))**2)
    return R2Score


# Make data
n = 100
print(f"n = {n}")
np.random.seed(101)
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)

#Create design matrix
p = 5
print(f"p = {p}")
X = design_matrix(x, y, p)

# Franke Function
noise = 0.8*np.random.randn(n, n)
z = np.ravel(FrankeFunction(x, y) + noise)

# Ordinary least squares
beta_OLS = OLS(X, z)
ztilde = X @ beta_OLS

# Confidence interval as function of beta
var_beta = variance_beta(beta_OLS, X, noise)
err_beta = 1.96*np.sqrt(var_beta)     # 95% confidence interval

# Plot of confidence interval
plt.errorbar(np.linspace(1, len(beta_OLS), len(beta_OLS)), beta_OLS, err_beta, fmt='.')
plt.title("95 % confidence interval as function of $\\beta$")
# plt.show()

# Mean squared error
MSE_OLS = MSE(z, ztilde)
print(f"MSE OLS: {MSE_OLS:.3}")

# R^2 score
R2Score_OLS = R2Score(z, ztilde)
print(f"R^2 Score OLS: {R2Score_OLS:.3}")

# Split in training and test data
X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size = 0.3)

# X_train = X_train[:, 1:]
# X_test = X_test[:, 1:]
# Scale data by subtracting the mean
scaler = StandardScaler()   # removes the mean and scales each feature/variable to unit variance
scaler.fit(X_train)     # compute the mean and std to be used for later scaling
X_train_scaled = scaler.transform(X_train)  # perform standardization by centering and scaling
X_test_scaled = scaler.transform(X_test)    # fit to data, then transform it
# X_train_scaled = np.column_stack((np.ones(np.shape(X_train_scaled)[0]), X_train_scaled))
# X_test_scaled = np.column_stack((np.ones(np.shape(X_test_scaled)[0]), X_test_scaled))
# Set the first column to 1 since StandardScaler sets it to 0
X_train_scaled[:, 0] = 1
X_test_scaled[:, 0] = 1
# print(X_train_scaled.mean(axis = 0))    # the mean of each feature (column) is 0
# print(X_train_scaled.std(axis = 0))     # the std of each feature (column) is 1
# print(X_test_scaled.mean(axis = 0))
# print(X_test_scaled.std(axis = 0))


beta_OLS_train_scaled = OLS(X_train_scaled, z_train)
ztilde_scaled = X_test_scaled @ beta_OLS_train_scaled

# print(f"MSE train: {MSE(z_train_scaled, ztilde_train_scaled)}")
print(f"MSE test: {MSE(z_test, ztilde_scaled):.3}")
print(f"R2Score: {R2Score(z_test, ztilde_scaled):.3}")

# Start figure
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface
ztilde_plot = np.reshape(ztilde, (n, n))
surf = ax.plot_surface(x, y, ztilde_plot, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a colar bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()