import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPrep:
    def design_matrix(self, x, y, degree):
        # make sure x and y  are 1D
        if (len(x.shape)) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((degree + 1)*(degree + 2)/2)      # number of columns/terms
        X = np.ones((N, l))

        for i in range(1, degree + 1):
            q = int(i*(i + 1)/2)
            for k in range(i + 1):
                X[:, q + k] = (x**(i - k)) * (y**k)
        return X

    def create_one_hot(self, n_inputs, labels):
        one_hot = np.zeros((n_inputs, 10))
        for i in range(n_inputs):
            one_hot[i, labels[i]] = 1
        return one_hot

    def train_test_scale(self, X, z):
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.3)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaler.fit(z_train)
        z_train_scaled = scaler.transform(z_train)
        z_test_scaled = scaler.transform(z_test)

        # Set the first column to 1 since StandardScaler sets it to 0
        X_train_scaled[:, 0] = 1
        X_test_scaled[:, 0] = 1

        return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled

    def train_test_split(self, X, z):
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.3)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Set the first column to 1 since StandardScaler sets it to 0
        X_train_scaled[:, 0] = 1
        X_test_scaled[:, 0] = 1

        return X_train_scaled, X_test_scaled, z_train, z_test
