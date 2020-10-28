import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPrep:
    def __init__(self, x, y, z, degree):
        self.x = x
        self.y = y
        self.z = z
        self.degree = degree

    def design_matrix(self):
        # make sure x and y  are 1D
        if (len(self.x.shape)) > 1:
            self.x = np.ravel(self.x)
            self.y = np.ravel(self.y)

        N = len(self.x)
        l = int((self.degree + 1)*(self.degree + 2)/2)      # number of columns/terms
        X = np.ones((N, l))

        for i in range(1, self.degree + 1):
            q = int(i*(i + 1)/2)
            for k in range(i + 1):
                X[:, q + k] = (self.x**(i - k)) * (self.y**k)
        return X

    def __call__(self):
        X = self.design_matrix()
        X_train, X_test, z_train, z_test = train_test_split(X, self.z, test_size=0.3)

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
