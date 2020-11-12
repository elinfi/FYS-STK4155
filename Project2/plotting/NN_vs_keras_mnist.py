import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import datasets

sys.path.append("../source")
import functions as f
import activation as act
import cost_functions as cost

from data_prep import DataPrep
from NeuralNetwork import DenseLayer, NeuralNetwork
from NN_keras import Keras

# set the parameters
n_epochs = 300
n_batches = 100
eta = 0.1
lmbda = 0.001
neurons = [50, 50]
n_outputs = 10
hidden_act = act.Sigmoid()
output_act = act.Softmax()
cost_func = cost.CrossEntropy()
seed = 2034

# download MNIST dataset
digits = datasets.load_digits()

# define input data and labels
dataset = digits.images
labels = digits.target.reshape(-1, 1)

# flatten the image
N = len(dataset)
dataset = dataset.reshape(N, -1)

# Transform labels to onehot vectors and split in train and test
data = DataPrep()
accuracy = cost.Accuracy()
one_hot = data.create_one_hot(N, labels)
X_train, X_test, z_train, z_test = data.train_test_split(dataset, one_hot)


# set up the neural network
network = NeuralNetwork(X_train.shape[1], neurons, n_outputs, cost_func)
network.create_layers(hidden_act, output_act, seed)

# train the network
batch_size = len(X_train)//n_batches
index_array = np.arange(len(X_train))
for k in range(n_epochs):
    np.random.shuffle(index_array)
    X_train = X_train[index_array]
    z_train = z_train[index_array]
    for l in range(0, len(X_train), batch_size):
        X_minibatch = X_train[l:l + batch_size]
        z_minibatch = z_train[l:l + batch_size]
        network.backprop(X_minibatch, z_minibatch, eta, lmbda)

network.feedforward(X_test)
z_tilde = np.argmax(network.layers[-1].a, axis=1)
print(f"Accuracy test NN {accuracy(z_tilde, np.argmax(z_test, axis=1))}")

# Set up keras network
NN_keras = Keras(neurons, n_outputs, eta, lmbda, loss='categorical_crossentropy',
                 metrics=['categorical_accuracy'], hidden_act='sigmoid', output_act='softmax')
model = NN_keras.create_neural_network_keras()
model.fit(X_train, z_train, epochs=n_epochs, batch_size=batch_size, verbose=0)
scores = model.evaluate(X_test, z_test)
print(f"MSE test keras {scores}")
