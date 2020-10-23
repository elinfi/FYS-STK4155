import numpy as np
import functions as f
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def learning_rate(t, t0 = 5, t1 = 50):
    return t0/(t + t1)

def grad_cost_function(X_batch, z_batch, beta, batch_size, lmbda=0):
    gradient = 2/batch_size * X_batch.T @ ((X_batch @ beta) - z_batch) + 2*lmbda*beta
    return gradient

def SGD(X_train, X_test, z_train, z_test, n_epochs, batch_size, gamma=0.9, lmbda=0):
    n = X_train.shape[0]
    if n % batch_size:
        raise Exception("Batch number and dataset not compatible")
    n_batches = int(n/batch_size)

    beta = np.random.randn(X_train.shape[1], 1)    # initialize beta

    v = 0
    mse_epochs = np.zeros(n_epochs)
    index_array = np.arange(n)
    for epoch in range(n_epochs):
        np.random.shuffle(index_array)
        X_minibatches = np.split(X_train[index_array], n_batches)
        z_minibatches = np.split(z_train[index_array], n_batches)

        i = 0
        for X_batch, z_batch in zip(X_minibatches, z_minibatches):
            # Calculate mean gradient of minibatch
            gradient = grad_cost_function(X_batch, z_batch, beta, batch_size, lmbda)

            # Update beta
            eta = learning_rate(epoch*n + i)
            v = gamma*v + eta*gradient
            beta = beta - v
            i += 1

        z_tilde = X_test @ beta
        mse_epochs[epoch] = f.MSE(z_test, z_tilde)
    return beta, mse_epochs


n = 100
noise = 0.1
degree = 10
n_epochs = 500
batch_size = 10

X_train, X_test, z_train, z_test = f.FrankeData(n, noise, degree)
X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = f.FrankeData(n, noise, degree, scaling=True)

sgdr = skl.SGDRegressor(loss='squared_loss').fit(X_train, np.ravel(z_train))
z_tilde_sklearn = sgdr.predict(X_test)
mse_sklearn = f.MSE(z_test, z_tilde_sklearn)
print(mse_sklearn)

beta, mse_epochs = SGD(X_train, X_test, z_train, z_test, n_epochs, batch_size)
beta_scaled, mse_epochs_scaled = SGD(X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled, n_epochs, batch_size)

plt.plot(np.arange(n_epochs), mse_epochs)
plt.legend()
plt.show()

plt.plot(np.arange(n_epochs), mse_epochs_scaled, label='scaled')
plt.legend()
plt.show()
