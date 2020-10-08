import numpy as np
import functions as f
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score


def cross_validation(n, maxdegree, noise, n_folds, method=f.OLS, seed=130, lmbda=0, datatype='Franke', filename='SRTM_data_Minneapolis'):
    """
    cross_validation

    Input:
        n           -   number of datapoints before meshgrid
        maxdegree   -   max degree to iterate over
        noise       -   amount of noise
        n_folds     -   number of folds in cross validation
        method      -   regression method (OLS, Ridge, Lasso)
        seed        -   seed to random number generator
        lmbda       -   lambda value to use in Ridge and Lasso
        datatype    -   datatype to fit (Franke, Terrain)
        filename    -   file with terrain data

    Output:
        polydegree      -   array with model complexity
        MSE_mean        -   array with mean MSE from each cross validation
        MSE_best        -   MSE for the best fit
        R2Score_skl     -   array with R2Score for Scikit Learn cross validation
        R2Score_mean    -   array with mean R2Score from each cross validation
    """
    if n%n_folds != 0:
        raise Exception("Can't divide data set in n_folds equally sized folds")

    polydegree = np.zeros(maxdegree)
    MSE_mean = np.zeros(maxdegree)
    MSE_mean_sklearn = np.zeros(maxdegree)
    R2Score_mean = np.zeros(maxdegree)
    R2Score_skl = np.zeros(maxdegree)

    # Make data
    np.random.seed(int(seed))

    if datatype == 'Franke':
        x_train, x_test, y_train, y_test, z_train, z_test = f.FrankeData(n, noise, test_size=0.3)

    elif datatype =='Terrain':
        x_train, x_test, y_train, y_test, z_train, z_test = f.TerrainData(n, filename)



    for degree in range(0, maxdegree):
        polydegree[degree] = degree

        # Create design matrix
        X_train = f.design_matrix(x_train, y_train, degree)

        # Shuffle data to get random folds
        index = np.arange(0, np.shape(X_train)[0], 1)
        np.random.seed(int(seed))
        np.random.shuffle(index)
        X_train_random = X_train[index,:]
        z_train_random = z_train[index]

        # Split data in n_folds folds
        X_folds = np.array(np.array_split(X_train_random, n_folds))
        z_folds = np.array(np.array_split(z_train_random, n_folds))

        if method == f.OLS:
            clf = skl.LinearRegression()
            scores = cross_val_score(clf, X_train, z_train, cv=n_folds, scoring='neg_mean_squared_error')
            MSE_mean_sklearn[degree] = np.abs(np.mean(scores))
            best_degree_sklearn = np.argmin(MSE_mean_sklearn)

            # Make fit to holy test data
            X_train_best = f.design_matrix(x_train, y_train, best_degree_sklearn)
            scaler = StandardScaler()
            scaler.fit(X_train_best)
            X_train_best_scaled = scaler.transform(X_train_best)
            X_test_best = f.design_matrix(x_test, y_test, best_degree_sklearn)
            X_test_best_scaled = scaler.transform(X_test_best)

            X_train_best_scaled[:, 0] = 1
            X_test_best_scaled[:, 0] = 1

            scaler.fit(z_train.reshape(-1, 1))
            z_train_scaled = scaler.transform(z_train.reshape(-1, 1))
            z_test_scaled = scaler.transform(z_test.reshape(-1, 1))

            beta_best_sklearn = f.OLS(X_train_best_scaled, z_train_scaled)

        elif method == f.Ridge:
            clf = skl.Ridge()
            scores = cross_val_score(clf, X_train, z_train, cv=n_folds, scoring='neg_mean_squared_error')
            MSE_mean_sklearn[degree] = np.abs(np.mean(scores))
            best_degree_sklearn = np.argmin(MSE_mean_sklearn)

            # Make fit to holy test data
            X_train_best = f.design_matrix(x_train, y_train, best_degree_sklearn)
            scaler = StandardScaler()
            scaler.fit(X_train_best)
            X_train_best_scaled = scaler.transform(X_train_best)
            X_test_best = f.design_matrix(x_test, y_test, best_degree_sklearn)
            X_test_best_scaled = scaler.transform(X_test_best)

            X_train_best_scaled[:, 0] = 1
            X_test_best_scaled[:, 0] = 1

            scaler.fit(z_train.reshape(-1, 1))
            z_train_scaled = scaler.transform(z_train.reshape(-1, 1))
            z_test_scaled = scaler.transform(z_test.reshape(-1, 1))

            beta_best_sklearn = f.OLS(X_train_best_scaled, z_train_scaled)


        elif method == 'Lasso':
            clf_lasso = skl.Lasso(alpha = lmbda, fit_intercept=False)
            scores = cross_val_score(clf_lasso, X_train, z_train, cv=n_folds, scoring='neg_mean_squared_error')
            MSE_mean_sklearn[degree] = np.abs(np.mean(scores))
            best_degree_sklearn = np.argmin(MSE_mean_sklearn)

            # Make fit to holy test data
            X_train_best = f.design_matrix(x_train, y_train, best_degree_sklearn)
            scaler = StandardScaler()
            scaler.fit(X_train_best)
            X_train_best_scaled = scaler.transform(X_train_best)
            X_test_best = f.design_matrix(x_test, y_test, best_degree_sklearn)
            X_test_best_scaled = scaler.transform(X_test_best)

            X_train_best_scaled[:, 0] = 1
            X_test_best_scaled[:, 0] = 1

            scaler.fit(z_train.reshape(-1, 1))
            z_train_scaled = scaler.transform(z_train.reshape(-1, 1))
            z_test_scaled = scaler.transform(z_test.reshape(-1, 1))

            beta_best_sklearn = f.OLS(X_train_best_scaled, z_train_scaled)

        # cross validation
        for k in range(n_folds):
            # Validation data
            X_val = X_folds[k]
            z_val = np.reshape(z_folds[k], (-1, 1))

            # Training data
            idx = np.ones(n_folds, dtype=bool)
            idx[k] = False
            X_train_fold = X_folds[idx]

            # Combine folds
            X_train_fold = np.reshape(X_train_fold, (X_train_fold.shape[0]*X_train_fold.shape[1], X_train_fold.shape[2]))
            z_train_fold = np.reshape(np.ravel(z_folds[idx]), (-1, 1))

            # Scaling data
            scaler = StandardScaler()                               # removes the mean and scales each feature/variable to unit variance
            scaler.fit(X_train_fold)                                # compute the mean and std to be used for later scaling
            X_train_fold_scaled = scaler.transform(X_train_fold)    # perform standardization by centering and scaling
            X_val_scaled = scaler.transform(X_val)
            # Set first column to one as StandardScaler sets it to zero
            X_train_fold_scaled[:, 0] = 1
            X_val_scaled[:, 0] = 1

            # scaler.fit(z_train_fold)
            # z_train_fold_scaled = scaler.transform(z_train_fold)
            # z_val_scaled = scaler.transform(z_val)
            z_train_fold_scaled = z_train_fold
            z_val_scaled = z_val

            # Choose method for calculating coefficients beta
            if method == f.OLS:
                beta_fold = method(X_train_fold_scaled, z_train_fold_scaled)
                z_tilde_fold = X_val_scaled @ beta_fold
                # z_tilde_fold_train = X_train_
            elif method == f.Ridge:
                beta_fold = method(X_train_fold_scaled, z_train_fold_scaled, lmbda, degree)
                z_tilde_fold = X_val_scaled @ beta_fold

            elif method == 'Lasso':
                clf_lasso = skl.Lasso(alpha = lmbda, fit_intercept=False).fit(X_train_fold_scaled, z_train_fold_scaled)
                z_tilde_fold = clf_lasso.predict(X_val_scaled)

            MSE_mean[degree] += f.MSE(z_val_scaled, z_tilde_fold)
            R2Score_mean[degree] += f.R2Score(z_val_scaled, z_tilde_fold)

        MSE_mean[degree] /= n_folds
        R2Score_mean[degree] /= n_folds

        # # Cross-validation using Scikit-Learn
        # clf = skl.LinearRegression()
        # R2Score_skl[degree] = np.mean(cross_val_score(clf, X_train, z_train, scoring='r2', cv=n_folds))

    # Find the degree with smallest MSE
    best_degree = np.argmin(MSE_mean)
    print(best_degree)

    # Make fit to holy test data
    X_train_best = f.design_matrix(x_train, y_train, best_degree)
    scaler.fit(X_train_best)
    X_train_best_scaled = scaler.transform(X_train_best)
    X_test_best = f.design_matrix(x_test, y_test, best_degree)
    X_test_best_scaled = scaler.transform(X_test_best)

    X_train_best_scaled[:, 0] = 1
    X_test_best_scaled[:, 0] = 1

    scaler.fit(z_train.reshape(-1, 1))
    z_train_scaled = scaler.transform(z_train.reshape(-1, 1))
    z_test_scaled = scaler.transform(z_test.reshape(-1, 1))

    beta_best = f.OLS(X_train_best_scaled, z_train_scaled)
    z_tilde_best = X_test_best_scaled @ beta_best
    MSE_best = f.MSE(z_test_scaled, z_tilde_best)
    print(MSE_best)



    return polydegree, MSE_mean, MSE_best, R2Score_skl, R2Score_mean, beta_best, best_degree, MSE_mean_sklearn, beta_best_sklearn

if __name__ == '__main__':
    # initial data
    n = 50            # number of data points
    maxdegree = 15
    noise = 0.1
    n_folds = 5             # number of folds
    method = f.OLS
    lmbda = 0
    seed = 130

    polydegree, MSE_mean, MSE_best, R2Score_skl, R2Score_mean \
        = cross_validation(n, maxdegree, noise, n_folds, method, seed, lmbda)
    plt.plot(polydegree, MSE_mean)
    plt.show()

    print(R2Score_skl.shape)

    plt.subplot(211)
    plt.plot(polydegree, R2Score_skl, label='sklearn')
    # plt.legend()

    plt.subplot(212)
    plt.plot(polydegree, R2Score_mean, label='own method')
    plt.legend()
    plt.show()
