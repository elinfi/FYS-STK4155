import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("../source")
import functions as f
import activation as act
import cost_functions as cost

from data_prep import DataPrep
from NeuralNetwork import DenseLayer, NeuralNetwork
from NN_keras import Keras

# set the parameters
n = 100
n_epochs = 300
n_batches = 100
eta = 0.5
lmbda = 0
neurons = [10, 10]
n_outputs = 1
hidden_act = act.Sigmoid()
output_act = act.Identity()

# create data using franke function
seed = 2034
np.random.seed(seed)
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = np.ravel(f.FrankeFunction(x, y) + 0.1*np.random.randn(x.shape[0], x.shape[1]))
z = z.reshape(-1, 1)

# set up the design matrix
data = DataPrep()
X = data.design_matrix(x, y, degree=1)[:, 1:]

# split data in train and test and scale it
X_train, X_test, z_train, z_test = data.train_test_scale(X, z)

# set up the neural network
network = NeuralNetwork(X_train.shape[1], neurons, n_outputs, cost.MSE())
network.create_layers(hidden_act, output_act, seed)

# train the network
batch_size = len(X_train)//n_batches
index_array = np.arange(len(X_train))
for k in range(n_epochs):
    np.random.shuffle(index_array)
    X_minibatches = np.split(X_train[index_array], n_batches)
    z_minibatches = np.split(z_train[index_array], n_batches)

    for l in range(n_batches):
        network.backprop(X_minibatches[l], z_minibatches[l], eta, lmbda)

network.feedforward(X_test)
print(f"MSE test NN {f.MSE(z_test, network.layers[-1].a)}")

# Set up keras network
NN_keras = Keras(neurons, n_outputs, eta, lmbda, loss='mean_squared_error',
                 metrics=[], hidden_act='sigmoid', output_act=tf.identity)
model = NN_keras.create_neural_network_keras()
model.fit(X_train, z_train, epochs=n_epochs, batch_size=batch_size, verbose=0)
scores = model.evaluate(X_test, z_test)
print(f"MSE test keras {scores}")
