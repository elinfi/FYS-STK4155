import sys
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import confusion_matrix

sys.path.append('../source')
import functions as f
import activation as act
import cost_functions as cost

from data_prep import DataPrep
from NeuralNetwork import NeuralNetwork

# set the parameters
n = 100
n_epochs = 200
n_batches = 500
eta = 0.01
lmbda = 0.001
neurons = []
n_outputs = 10
hidden_act = act.Sigmoid()
output_act = act.Softmax()
cost_func = cost.CrossEntropy()
no_hidden = True
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

# set up the neural network
network = NeuralNetwork(X_train.shape[1], neurons, n_outputs, cost_func)
network.create_layers(hidden_act, output_act, seed, no_hidden)

index_array = np.arange(len(X_train))
batch_size = len(X_train)//n_batches
for k in range(n_epochs):
    np.random.shuffle(index_array)
    X_train = X_train[index_array]
    z_train = z_train[index_array]
    for l in range(0, len(X_train), batch_size):
        X_minibatch = X_train[l:l + batch_size]
        z_minibatch = z_train[l:l + batch_size]
        # eta = network.learning_rate(epoch*N + j, 2, 20)
        network.backprop_two_layers(X_minibatch, z_minibatch, eta, lmbda)

network.feedforward(X_test)
z_tilde = np.argmax(network.layers[-1].a, axis=1)

numbers = np.arange(0,10)
confusion_matrix = confusion_matrix(z_tilde, z_test, normalize='true')
heatmap = sb.heatmap(confusion_matrix,cmap='YlGnBu_r',
                              xticklabels=["%d" %i for i in numbers],
                              yticklabels=["%d" %i for i in numbers],
                              cbar_kws={'label': 'Accuracy'},
                              fmt = ".2",
                              edgecolor="none",
                              annot = True)
heatmap.set_xlabel('Prediction', size=12)
heatmap.set_ylabel('Expected value', size=12)
heatmap.set_title('Logistic regression prediction accuracy', size=16)
fig = heatmap.get_figure()
plt.yticks(rotation=0)
plt.show()
