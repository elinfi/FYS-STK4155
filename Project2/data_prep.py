import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPrep:
    def design_matrix(self, x, y, degree):
        """Create design matrix.

        Keyword arguments:
        x -- x coordinate input
        y -- y coordinate input
        degree -- model complexity

        Return value:
        X -- design matrix
        """
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
        """Transform the labels to onehot of size 10.

        Keyword arguments:
        n_inputs -- numbers of datapoints
        labels -- the true label to each datapoint

        Return value:
        one_hot -- 2D matrix containing one hot vector of all labels
        """
        one_hot = np.zeros((n_inputs, 10))
        for i in range(n_inputs):
            one_hot[i, labels[i]] = 1
        return one_hot

    def train_test_scale(self, X, z):
        """Split and scale both input and output data in train and test data.

        Keyword arguments:
        X -- input data
        z -- output data

        Return values:
        X_train_scaled -- scaled input train data
        X_test_scaled -- scaled input test data
        z_train_scaled -- scaled output train data
        z_test_scaled -- scaled ooutput test data
        """
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
        """Split data in test and train, but only scale input data.

        Keyword arguments:
        X -- input data
        z -- output data

        Return value:
        X_train_scaled -- scaled input train data
        X_test_scaled -- scaled input test data
        z_train -- output train data
        z_test -- input test data
        """
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.3)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Set the first column to 1 since StandardScaler sets it to 0
        X_train_scaled[:, 0] = 1
        X_test_scaled[:, 0] = 1

        return X_train_scaled, X_test_scaled, z_train, z_test
