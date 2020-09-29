import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from functions import MSE, R2Score
from functions import OLS, Ridge, variance_beta
from functions import FrankeFunction, design_matrix

# initial data
n = 20
maxdegree = 14
noise = 0.1

# arrays for plotting of error
polydegree = np.zeros(maxdegree)
MSE_OLS = np.zeros(maxdegree)
R2Score_OLS = np.zeros(maxdegree)
MSE_OLS_test = np.zeros(maxdegree)
MSE_OLS_train = np.zeros(maxdegree)
MSE_OLS_train_scaled = np.zeros(maxdegree)
MSE_OLS_test_scaled = np.zeros(maxdegree)
R2Score_OLS_scaled = np.zeros(maxdegree)

# Make data
print(f"n = {n}")
np.random.seed(101)
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)

# Franke Function
z = np.ravel(FrankeFunction(x, y) + noise*np.random.randn(n, n))


for degree in range(0, maxdegree):
    print(f"degree = {degree}")
    polydegree[degree] = degree

    #Create design matrix
    X = design_matrix(x, y, degree)

    # Ordinary least squares
    beta_OLS = OLS(X, z)
    ztilde = X @ beta_OLS

    # Mean squared error
    MSE_OLS[degree] = MSE(z, ztilde)
    # print(f"MSE OLS: {MSE_OLS[degree]:.3}")

    # R^2 score
    R2Score_OLS[degree] = R2Score(z, ztilde)
    # print(f"R2Score OLS: {R2Score_OLS[degree]:.3}")

    # Split in training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size = 0.3)

    # Scale data by subtracting the mean
    scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
    scaler.fit(X_train)                         # compute the mean and std to be used for later scaling
    X_train_scaled = scaler.transform(X_train)  # perform standardization by centering and scaling
    X_test_scaled = scaler.transform(X_test)    # fit to data, then transform it

    # Set the first column to 1 since StandardScaler sets it to 0
    X_train_scaled[:, 0] = 1
    X_test_scaled[:, 0] = 1

    # Ordinary linear square estimate train/test
    beta_OLS_train = OLS(X_train, z_train)
    ztilde_test = X_test @ beta_OLS_train
    ztilde_train = X_train @ beta_OLS_train

    # Confidence interval as function of beta
    var_beta = variance_beta(X_test, noise)
    err_beta = 1.96*np.sqrt(var_beta)           # 95% confidence interval

    # Plot of confidence interval
    # plt.errorbar(np.linspace(1, len(beta_OLS), len(beta_OLS)), beta_OLS, err_beta, fmt='.')
    # plt.title("95 % confidence interval as function of $\\beta$")
    # plt.show()

    MSE_OLS_train[degree] = MSE(z_train, ztilde_train)
    MSE_OLS_test[degree] = MSE(z_test, ztilde_test)

    # Ordinary linear squared estimate with scaling
    print(np.shape(X_train_scaled))
    print(np.shape(z_train))
    beta_OLS_train_scaled = OLS(X_train_scaled, z_train)
    ztilde_test_scaled = X_test_scaled @ beta_OLS_train_scaled
    ztilde_train_scaled = X_train_scaled @ beta_OLS_train_scaled

    MSE_OLS_train_scaled[degree] = MSE(z_train, ztilde_train_scaled)
    MSE_OLS_test_scaled[degree] = MSE(z_test, ztilde_test_scaled)
    R2Score_OLS_scaled[degree] = R2Score(z_test, ztilde_test_scaled)

    # print(f"MSE train: {MSE_OLS_train_scaled[degree]:.3}")
    # print(f"MSE test: {MSE_OLS_test_scaled[degree]:.3}")
    # print(f"R2Score test: {R2Score_OLS_scaled[degree]:.3}")
    # plt.plot(p, MSE(z_train, ztilde_train_scaled), '.', label="MSE train")
    # plt.plot(p, MSE(z_test, ztilde_test_scaled), '.', label="MSE test")
    # plt.plot(p, R2Score(z_test, ztilde_test_scaled), '.', label="R2Score test")


plt.plot(polydegree, MSE_OLS_test_scaled, label='MSE test scaled')
plt.plot(polydegree, R2Score_OLS_scaled, label='R2Score scaled')
plt.xlabel("Model complexity")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.plot(polydegree, MSE_OLS_test, label='test')
plt.plot(polydegree, MSE_OLS_train, label='train')
plt.xlabel("Model COmplexity")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.plot(polydegree, MSE_OLS_test_scaled, label='test scaled')
plt.plot(polydegree, MSE_OLS_train_scaled, label='train scaled')
plt.xlabel("Model Complexity")
plt.ylabel("MSE")
plt.legend()
plt.show()

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
