import numpy as np
import functions as f
import activation as act
import cost_functions as cost


class DenseLayer:
    def __init__(self, n_inputs, n_outputs, activation):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.w = np.random.rand(self.n_inputs, self.n_outputs)
        self.b = np.random.randn(1, self.n_outputs)

    def __call__(self, X):
        self.z = X @ self.w + self.b
        self.a = self.activation(self.z)
        self.a_deriv = self.activation.deriv(self.z)
        return self.a

class NeuralNetwork:
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def feedforward(self, a):
        for layer in self.layers:
            a = layer(a)

    def backprop(self, X, y, eta):
        layers = self.layers
        dCda = self.cost.deriv(layers[-1].a, y)
        delta_L = dCda * layers[-1].a_deriv

        for l in reversed(range(1, len(layers) - 1)):
            delta_L = (delta_L @ layers[l + 1].w.T) * layers[l].a_deriv
            layers[l].w =  layers[l].w - eta*(layers[l - 1].a.T @ delta_L)
            layers[l].b = layers[l].b - eta*delta_L

        delta_L = (delta_L @ layers[1].w.T) * layers[0].a_deriv
        layers[0].w = layers[0].w - eta*(X.T @ delta_L)
        layers[0].b = layers[0].b - eta*delta_L

        self.feedforward(X)

x = np.linspace(0,1,100).reshape(-1,1)
y = x**2 + 0.1*np.random.randn(100, 1)

layer1 = DenseLayer(1, 10, act.Sigmoid())
layer2 = DenseLayer(10, 10, act.Sigmoid())
layer3 = DenseLayer(10, 1, act.Identity())

layers = [layer1, layer2, layer3]
network = NeuralNetwork(layers, cost.MSE())
network.feedforward(x)
print(f.MSE(y, layer3.a))

for i in range(1000):
    network.backprop(x, y, 0.5)
print(f.MSE(y, layer3.a))
