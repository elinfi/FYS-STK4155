import sys
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

sys.path.append('../source')
import functions as f
import activation as act
import cost_functions as cost

from data_prep import DataPrep
from NeuralNetwork import NeuralNetwork

# set the parameters
n = 100
n_epochs = 300
n_batches = 50
neurons = [10, 10]
n_outputs = 1
hidden_act = act.RELU()
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

array_lambda = [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.9]
array_eta = [0.01, 0.1, 0.2, 0.5, 0.7, 1]

mse_heatmap = np.zeros((len(array_lambda), len(array_eta)))
index_array = np.arange(len(X_train))
for i, lmbda in enumerate(array_lambda):
    for j, eta in enumerate(array_eta):
        network.create_layers(hidden_act, output_act, seed)
        for k in range(n_epochs):
            np.random.shuffle(index_array)
            X_minibatches = np.split(X_train[index_array], n_batches)
            z_minibatches = np.split(z_train[index_array], n_batches)

            for l in range(n_batches):
                # eta = network.learning_rate(epoch*N + j, 2, 20)
                network.backprop(X_minibatches[l], z_minibatches[l], eta, lmbda)

        network.feedforward(X_test)
        mse_heatmap[i, j] = np.log10(f.MSE(z_test, network.layers[-1].a))

heatmap = sb.heatmap(mse_heatmap, annot=mse_heatmap, cmap='YlGnBu',
                     xticklabels=array_eta, yticklabels=array_lambda,
                     cbar_kws={'label': 'MSE'})
heatmap.set_xlabel('$\eta$', size=12)
heatmap.set_ylabel('$\lambda$', size=12)
heatmap.invert_xaxis()
heatmap.set_title('RELU + Identity', size=16)
plt.show()
