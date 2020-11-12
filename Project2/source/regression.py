import numpy as np
import functions as f

class Regression:
    def __init__(self, X_train, X_test, z_train, z_test):
        """Something about regression.

        Keyword arguments:
        X_train -- input train data
        X_test -- input test data
        z_train -- output train data
        z_test -- output test data
        """
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

    def learning_rate(self, t, t0 = 5, t1 = 100):
        """Decreasing learning rate.

        Keyword arguments:
        t -- timestep
        t0 -- numerator (default = 5)
        t1 -- addition to t in enumerator (default = 100)

        Return value:
        t0/(t + t1) -- updated learning rate
        """
        return t0/(t + t1)

    def grad_cost_function(self, X_batch, z_batch, beta, batch_size, lmbda=0):
        """Computes the gradient of OLS (lmbda = 0) and Ridge cost functions.

        Keyword arguments:
        X_batch -- input minibatch
        z_batch -- output minibatch
        beta -- weights/coefficients
        batch-size -- size of minibatch
        lmbda -- regularization parameter (lmbda = 0)

        Return value:
        gradient -- gradient of OLS (lmbda = 0) and Ridge cost function
        """
        gradient = 2/batch_size * X_batch.T @ ((X_batch @ beta) - z_batch)
                   + 2*lmbda*beta
        return gradient

    def SGD(self, n_epochs, batch_size, gamma=0.9, lmbda=0):
        """Stochastic gradient descent.

        Keyword arguments:
        n_epochs -- number of epochs
        batch_size -- size of minibatch
        gamma -- momentum parameter (default = 0.9)
        lmbda -- regularization parameter (default = 0)

        Exception:
        Exception raised when batch size does not result in an equal division of
        training data.
        """
        n = self.X_train.shape[0]
        if n % batch_size:
            raise Exception("Batch number and dataset not compatible")
        n_batches = int(n/batch_size)

        beta = np.random.randn(self.X_train.shape[1], 1)    # initialize beta

        v = 0
        self.mse_epochs = np.zeros(n_epochs)
        index_array = np.arange(n)
        for epoch in range(n_epochs):
            np.random.shuffle(index_array)
            X_minibatches = np.split(self.X_train[index_array], n_batches)
            z_minibatches = np.split(self.z_train[index_array], n_batches)

            i = 0
            for X_batch, z_batch in zip(X_minibatches, z_minibatches):
                # Calculate mean gradient of minibatch
                gradient = self.grad_cost_function(X_batch, z_batch, beta,
                                                   batch_size, lmbda)

                # Update beta
                eta = self.learning_rate(epoch*n + i)
                v = gamma*v + eta*gradient
                beta = beta - v
                i += 1

            z_tilde = self.X_test @ beta
            self.mse_epochs[epoch] = f.MSE(self.z_test, z_tilde)

    def OLS(self):
        """Calculates the ordinary least squares and its mean squared error."""
        beta = f.OLS(self.X_train, self.z_train)
        z_tilde = self.X_test @ beta
        self.MSE = f.MSE(self.z_test, z_tilde)
