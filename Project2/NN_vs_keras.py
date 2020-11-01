import numpy as np
import functions as f
import activation as act
import cost_functions as cost
import matplotlib.pyplot as plt
from data_prep import DataPrep
from NeuralNetwork import DenseLayer, NeuralNetwork
from NN_keras import create_neural_network_keras

n = 100
n_epochs = 100
eta = 0.5

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = np.ravel(f.FrankeFunction(x, y))
z = z.reshape(-1, 1)

data = DataPrep()
X = data.design_matrix(x, y, degree=1)
X_train, X_test, z_train, z_test = data.train_test_scale(X, z)
X_train, X_test,= X_train[:, 1:], X_test[:, 1:]

n_inputs = X_train.shape[1]
n_neurons_layer1 = 10
n_neurons_layer2 = 6
n_outputs = 1

layer1 = DenseLayer(n_inputs, n_neurons_layer1, act.Sigmoid())
layer2 = DenseLayer(n_neurons_layer1, n_neurons_layer2, act.Sigmoid())
layer3 = DenseLayer(n_neurons_layer2, n_outputs, act.Softmax())

layers = [layer1, layer2, layer3]
network = NeuralNetwork(layers, cost.MSE())
network.feedforward(X_train)
print(f.MSE(z_train, layer3.a))

for i in range(n_epochs):
    network.backprop(X_train, z_train, eta)

print(f.MSE(z_train, layer3.a))

network.feedforward(X_test)
print(f"her {f.MSE(z_test, layer3.a)}")

DNN = create_neural_network_keras(n_neurons_layer1, n_neurons_layer2,\
                                  n_outputs, eta=eta)
DNN.fit(X_train, z_train, epochs=n_epochs, batch_size=X_train.shape[0], verbose=0)
scores = DNN.evaluate(X_test, z_test)


print("Learning rate = ", eta)
print("Test accuracy: %.3f" % scores)
print()
