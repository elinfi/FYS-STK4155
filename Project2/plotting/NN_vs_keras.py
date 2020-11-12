import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("../src")
import functions as f
import activation as act
import cost_functions as cost

from data_prep import DataPrep
from NeuralNetwork import DenseLayer, NeuralNetwork
from NN_keras import Keras

n = 100
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = np.ravel(f.FrankeFunction(x, y))
z = z.reshape(-1, 1)

data = DataPrep()
X = data.design_matrix(x, y, degree=1)
X_train, X_test, z_train, z_test = data.train_test_split(X, z)
X_train, X_test,= X_train[:, 1:], X_test[:, 1:]


n_inputs = X_train.shape[1]
neurons = [10, 10, 30]
n_outputs = 1

network = NeuralNetwork(n_inputs, neurons, n_outputs, cost.MSE())
network.create_layers(act.Sigmoid(), act.Identity())
network.feedforward(X_test)
print(f"MSE before {f.MSE(z_test, network.layers[-1].a)}")


n_epochs = 100
print(len(X_train))
n_batches = 700
batch_size = len(X_train)//700
eta = 0.5
lmbda = 0
mse_epochs = np.zeros(n_epochs)
for i in range(n_epochs):
    network.backprop(X_train, z_train, eta, lmbda)
    network.feedforward(X_test)
    mse_epochs[i] = f.MSE(z_test, network.layers[-1].a)

plt.plot(np.arange(n_epochs), mse_epochs)
plt.show()

# print(f"MSE train {f.MSE(z_train, network.layers[-1].a)}")
network.feedforward(X_test)
print(f"MSE test {f.MSE(z_test, network.layers[-1].a)}")

index_array = np.arange(len(X_train))
mse_epochs = np.zeros(n_epochs)
for i in range(n_epochs):
    np.random.shuffle(index_array)
    X_minibatches = np.split(X_train[index_array], n_batches)
    z_minibatches = np.split(z_train[index_array], n_batches)

    for j in range(n_batches):
        network.backprop(X_minibatches[j], z_minibatches[j], eta)
    network.feedforward(X_test)
    mse_epochs[i] = f.MSE(z_test, network.layers[-1].a)

plt.plot(np.arange(n_epochs), mse_epochs)
plt.show()

network.feedforward(X_test)
print(f"MSE test SGD {f.MSE(z_test, network.layers[-1].a)}")
#
NN_keras = Keras(neurons, n_outputs, eta, lmbda, loss='mean_squared_error',
                 metrics=[], input_act='sigmoid', output_act=tf.identity)
model = NN_keras.create_neural_network_keras()
model.fit(X_train, z_train, epochs=n_epochs, batch_size=X_train.shape[0], verbose=0)
scores = model.evaluate(X_train, z_train)
scores = model.evaluate(X_test, z_test)
#
#
# print("Learning rate = ", eta)
# print("Test accuracy: %.3f" % scores)
# print()
