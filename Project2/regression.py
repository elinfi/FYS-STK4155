import numpy as np
import functions as f

class Regression:
    def __init__(self, X_train, X_test, z_train, z_test):
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

    def learning_rate(self, t, t0 = 5, t1 = 100):
        return t0/(t + t1)

    def grad_cost_function(self, X_batch, z_batch, beta, batch_size, lmbda=0):
        gradient = 2/batch_size * X_batch.T @ ((X_batch @ beta) - z_batch) + 2*lmbda*beta
        return gradient

    def SGD(self, n_epochs, batch_size, gamma=0.9, lmbda=0):
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
                gradient = self.grad_cost_function(X_batch, z_batch, beta, batch_size, lmbda)

                # Update beta
                eta = self.learning_rate(epoch*n + i)
                v = gamma*v + eta*gradient
                beta = beta - v
                i += 1

            z_tilde = self.X_test @ beta
            self.mse_epochs[epoch] = f.MSE(self.z_test, z_tilde)

    def OLS(self):
        beta = f.OLS(self.X_train, self.z_train)
        z_tilde = self.X_test @ beta
        self.MSE = f.MSE(self.z_test, z_tilde)
