import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

import activation as act
import cost_functions as cost
from data_prep import DataPrep
from NeuralNetwork import NeuralNetwork
from NN_keras import Keras

# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target.reshape(-1, 1)

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))


data = DataPrep()
one_hot = data.create_one_hot(n_inputs, labels)
X_train, X_test, z_train, z_test = data.train_test_split(inputs, one_hot)

n_inputs = X_train.shape[1]
neurons = [50, 20, 100, 50, 20]
n_outputs = 10

network = NeuralNetwork(n_inputs, neurons, n_outputs, cost.MSE())
network.create_layers(act.Sigmoid(), act.Softmax())

n_epochs = 500
eta = 0.1
for i in range(n_epochs):
    network.backprop(X_train, z_train, eta)

network.feedforward(X_test)

accuracy = cost.Accuracy()
# z_tilde = np.argmax(network.layers[-1].a, axis=1)
z_tilde = network.layers[-1].a
print(z_tilde)
print(np.argmax(z_test, axis=1))
print(accuracy(z_tilde, np.argmax(z_test, axis=1)))


# NN_keras = Keras(neurons, n_outputs, eta, loss='categorical_crossentropy',
#                  metrics=['accuracy'], input_act='relu',
#                  output_act='softmax')
# model = NN_keras.create_neural_network_keras()
# model.fit(X_train, z_train, epochs=n_epochs, batch_size=X_train.shape[0])
# scores = model.evaluate(X_test, z_test)
# scores = model.evaluate(X_train, z_train)



# print(np.argmax(model.predict(X_test), axis=1))
# print(np.argmax(z_test, axis=1))
# z_tilde = np.argmax(model.predict(X_test), axis=1)
# print(z_tilde.shape)
# # print(z_test - z_tilde)
# print(z_train.shape)
# print(accuracy(z_tilde, np.argmax(z_test, axis=1)))
