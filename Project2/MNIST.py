import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

import activation as act
import cost_functions as cost
from data_prep import DataPrep
from NeuralNetwork import NeuralNetwork, DenseLayer
from NN_keras import Keras

# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
dataset = digits.images
labels = digits.target.reshape(-1, 1)

# flatten the image
N = len(dataset)
dataset = dataset.reshape(N, -1)

# Transform labels to onehot vectors and split in train and test
data = DataPrep()
one_hot = data.create_one_hot(N, labels)
X_train, X_test, z_train, z_test = data.train_test_split(dataset, one_hot)

n_inputs = X_train.shape[1]
neurons = []
n_outputs = 10

network = NeuralNetwork(n_inputs, neurons, n_outputs, cost.MSE())
network.create_layers(act.Sigmoid(), act.Sigmoid())


n_epochs = 500
eta = 0.001
lmbda = 0
for i in range(n_epochs):
    network.backprop_two_layers(X_train, z_train, eta, lmbda)

network.feedforward(X_test)

accuracy = cost.Accuracy()
# z_tilde = np.argmax(network.layers[-1].a, axis=1)
z_tilde = np.argmax(network.layers[-1].a, axis=1)
# print(np.argmax(z_test, axis=1))
print(z_tilde)
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
