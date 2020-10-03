import numpy as np
import functions as f
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def no_resampling(n, maxdegree, noise, method=f.OLS, lmbda=0, seed=7053):
    # arrays for plotting of error
    polydegree = np.zeros(maxdegree)
    MSE_OLS = np.zeros(maxdegree)
    R2Score_OLS = np.zeros(maxdegree)
    MSE_test = np.zeros(maxdegree)
    MSE_train = np.zeros(maxdegree)
    MSE_train_scaled = np.zeros(maxdegree)
    MSE_test_scaled = np.zeros(maxdegree)
    R2Score_scaled = np.zeros(maxdegree)

    # Make data
    np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x, y = np.meshgrid(x, y)

    # Franke Function
    z = np.ravel(f.FrankeFunction(x, y) + noise*np.random.randn(n, n))

    for degree in range(0, maxdegree):
        polydegree[degree] = degree

        #Create design matrix
        X = f.design_matrix(x, y, degree)

        # Split in training and test data
        X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size = 0.3)

        # OLS estimate train/test without scaled
        beta_OLS_train = f.OLS(X_train, z_train)
        ztilde_test = X_test @ beta_OLS_train
        ztilde_train = X_train @ beta_OLS_train

        MSE_train[degree] = f.MSE(z_train, ztilde_train)
        MSE_test[degree] = f.MSE(z_test, ztilde_test)

        # Scale data
        scaler = StandardScaler()                   # removes the mean and scales each feature/variable to unit variance
        scaler.fit(X_train)                         # compute the mean and std to be used for later scaling
        X_train_scaled = scaler.transform(X_train)  # perform standardization by centering and scaling
        X_test_scaled = scaler.transform(X_test)    # fit to data, then transform it
        scaler.fit(z_train)
        # z_train_scaled = scaler.transform(z_train)
        # z_test_scaled = scaler.transform(z_test)
        z_train_scaled = z_train
        z_test_scaled = z_test

        # Set the first column to 1 since StandardScaler sets it to 0
        X_train_scaled[:, 0] = 1
        X_test_scaled[:, 0] = 1

        if method == f.OLS:
            beta_train_scaled = method(X_train_scaled, z_train_scaled)
            z_tilde_test_scaled = X_test_scaled @ beta_train_scaled
            z_tilde_train_scaled = X_train_scaled @ beta_train_scaled

        elif method == f.Ridge:
            beta_train_scaled = method(X_train_scaled, z_train_scaled, lmbda, degree)
            z_tilde_test_scaled = X_test_scaled @ beta_train_scaled
            z_tilde_train_scaled = X_train_scaled @ beta_train_scaled

        elif method == 'Lasso':
            clf_lasso = skl.Lasso(alpha = lmbda, fit_intercept=False).fit(X_train_scaled, z_train_scaled)
            z_tilde_test_scaled = clf_lasso.predict(X_test_scaled)
            z_tilde_train_scaled = clf_lasso.predict(X_train_scaled)


        MSE_train_scaled[degree] = f.MSE(z_train_scaled, z_tilde_train_scaled)
        MSE_test_scaled[degree] = f.MSE(z_test_scaled, z_tilde_test_scaled)
        R2Score_scaled[degree] = f.R2Score(z_test_scaled, z_tilde_test_scaled)


    return polydegree, MSE_train, MSE_test, MSE_train_scaled, MSE_test_scaled, R2Score_scaled





    # Start figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface
    # ztilde_plot = np.reshape(ztilde, (n, n))
    # surf = ax.plot_surface(x, y, ztilde_plot, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a colar bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
