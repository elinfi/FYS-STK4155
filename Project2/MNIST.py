import numpy as np
import matplotlib.pyplot as plt
import activation as act
import cost_functions as cost
from sklearn import datasets
from data_prep import DataPrep
from NeuralNetwork import DenseLayer, NeuralNetwork

# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target.reshape(-1, 1)

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))

data = DataPrep()
X_train, X_test, z_train, z_test = data.train_test_scale(inputs, labels)

n_inputs = X_train.shape[1]
n_neurons_layer1 = 20
n_neurons_layer2 = 50
n_outputs = 1

layer1 = DenseLayer(n_inputs, n_neurons_layer1, act.Sigmoid())
layer2 = DenseLayer(n_neurons_layer1, n_neurons_layer2, act.Sigmoid())
layer3 = DenseLayer(n_neurons_layer2, n_outputs, act.Softmax())

layers = [layer1, layer2, layer3]
network = NeuralNetwork(layers, cost.Accuracy())
network.feedforward(X_train)
print(layer3.a)

# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
plt.show()
