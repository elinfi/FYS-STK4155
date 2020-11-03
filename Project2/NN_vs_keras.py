import numpy as np
import functions as f
import activation as act
import cost_functions as cost
import matplotlib.pyplot as plt
from data_prep import DataPrep
from NeuralNetwork import DenseLayer, NeuralNetwork
from NN_keras import Keras

n = 20
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = np.ravel(f.FrankeFunction(x, y))
z = z.reshape(-1, 1)

data = DataPrep()
X = data.design_matrix(x, y, degree=5)
X_train, X_test, z_train, z_test = data.train_test_split(X, z)
X_train, X_test,= X_train[:, 1:], X_test[:, 1:]


n_inputs = X_train.shape[1]
neurons = [50, 20, 100, 50, 50, 20]
n_outputs = 1

network = NeuralNetwork(n_inputs, neurons, n_outputs, cost.MSE())
network.create_layers(act.Sigmoid(), act.Softmax())
network.feedforward(X_train)
print(f.MSE(z_train, network.layers[-1].a))


n_epochs = 500
eta = 0.5
for i in range(n_epochs):
    network.backprop(X_train, z_train, eta)

print(f"MSE train {f.MSE(z_train, network.layers[-1].a)}")
network.feedforward(X_test)
print(f"MSE test {f.MSE(z_test, network.layers[-1].a)}")


NN_keras = Keras(neurons, n_outputs, eta, loss='mean_squared_error', \
                 metrics=[], input_act='sigmoid', output_act='softmax')
model = NN_keras.create_neural_network_keras()
model.fit(X_train, z_train, epochs=n_epochs, batch_size=X_train.shape[0], verbose=0)
scores = model.evaluate(X_test, z_test)
scores = model.evaluate(X_train, z_train)


print("Learning rate = ", eta)
print("Test accuracy: %.3f" % scores)
print()
