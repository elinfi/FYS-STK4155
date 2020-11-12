import sys
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn import datasets

sys.path.append('../src')
import functions as f
import activation as act
import cost_functions as cost

from data_prep import DataPrep
from NeuralNetwork import NeuralNetwork

# set the parameters
n = 100
n_epochs = 300
n_batches = 100
neurons = [50, 50]
n_outputs = 10
hidden_act = act.Sigmoid()
output_act = act.Softmax()
cost_func = cost.CrossEntropy()
no_hidden = False
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
z_test = np.argmax(z_test, axis=1)
batch_size = len(X_train)//n_batches

# set up the neural network
network = NeuralNetwork(X_train.shape[1], neurons, n_outputs, cost_func)

array_lambda = [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.9]
array_eta = [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1]

accuracy_heatmap = np.zeros((len(array_lambda), len(array_eta)))
index_array = np.arange(len(X_train))

for i, lmbda in enumerate(array_lambda):
    for j, eta in enumerate(array_eta):
        network.create_layers(hidden_act, output_act, seed, no_hidden)
        for k in range(n_epochs):
            np.random.shuffle(index_array)
            X_train = X_train[index_array]
            z_train = z_train[index_array]
            for l in range(0, len(X_train), batch_size):
                X_minibatch = X_train[l:l + batch_size]
                z_minibatch = z_train[l:l + batch_size]
                # eta = network.learning_rate(epoch*N + j, 2, 20)
                network.backprop(X_minibatch, z_minibatch, eta, lmbda)

        network.feedforward(X_test)
        z_tilde = np.argmax(network.layers[-1].a, axis=1)
        accuracy_heatmap[i, j] = accuracy(z_tilde, z_test)

heatmap = sb.heatmap(accuracy_heatmap , annot=True, cmap='YlGnBu_r',
                     xticklabels=array_eta, yticklabels=array_lambda,
                     cbar_kws={'label': 'Accuracy'})
heatmap.set_xlabel('$\eta$', size=12)
heatmap.set_ylabel('$\lambda$', size=12)
heatmap.invert_xaxis()
heatmap.set_title('Sigmoid + Softmax (MNIST)', size=16)
plt.show()
