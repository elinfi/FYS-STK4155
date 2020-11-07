import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_outputs, activation):
        """Create one layer in the neural network.

        Keyword arguments:
        n_inputs -- number of inputs to the layer
        n_outputs -- number of outputs to the layer
        activation -- activation function used on input to get output
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.w = np.random.randn(self.n_inputs, self.n_outputs)
        self.b = np.random.randn(1, self.n_outputs)
        # self.b = 0.01*np.ones((1, self.n_outputs))

    def __call__(self, X):
        """Returns the output of the layer.

        Keyword arguments:
        X -- input data to layer

        Return value:
        self.a -- activation function applied on weighted sum
        """
        self.z = X @ self.w + self.b
        self.a = self.activation(self.z)
        self.a_deriv = self.activation.deriv(self.z)
        return self.a

class NeuralNetwork:
    def __init__(self, n_inputs, neurons, n_outputs, cost):
        """Set up and update feed forward neural network.

        Keyword arguments:
        n_inputs -- number of nodes in input layer
        neurons -- list with number of nodes in each hidden layer
        n_outputs -- number of nodes in output layer
        cost -- cost function to use to update the weights and biases for each
                layer
        """
        self.n_inputs = n_inputs
        self.neurons = neurons
        self.n_outputs = n_outputs
        self.cost = cost

    def create_layers(self, activation, output_activation, no_hidden=False):
        """Set up the layers in the neural network.

        Keyword arguments:
        activation -- activation function used on the hidden layers
        output_activation -- activation function used on output layer
        hidden -- creates neural network without hidden layers when true
                  (defaul = False)
        """
        self.layers = []

        if no_hidden:
            # creates network with only input and output layers
            self.layers.append(DenseLayer(self.n_inputs, self.n_outputs,
                                          output_activation))
        else:
            # input layer to first hidden layer
            self.layers.append(DenseLayer(self.n_inputs, self.neurons[0],
                                          activation))
            # hidden layers
            for i in range(len(self.neurons) - 1):
                self.layers.append(DenseLayer(self.neurons[i], self.neurons[i + 1],
                                              activation))
            # last hidden layer to output layer
            self.layers.append(DenseLayer(self.neurons[-1], self.n_outputs,
                                          output_activation))

    def feedforward(self, a):
        """Feed forward.

        Keyword arguments:
        a -- input to the activation function of a layer
        """
        for layer in self.layers:
            a = layer(a)

    def backprop(self, X, y, eta, lmbda):
        """Backpropagation.

        Keyword arguments:
        X -- matrix with training data and its input features
        y -- the true classification
        eta -- learning rate
        lmbda - regularization parameter
        """
        self.feedforward(X)
        layers = self.layers

        # calculate delta_L for output layer
        dCda = self.cost.deriv(layers[-1].a, y)
        delta_L = dCda * layers[-1].a_deriv

        # last hidden layer to second first hidden layer
        for l in reversed(range(1, len(layers) - 1)):
            delta_L = (delta_L @ layers[l + 1].w.T) * layers[l].a_deriv
            # update weights
            layers[l].w =  layers[l].w - eta*(layers[l - 1].a.T @ delta_L) \
                           - 2*eta*lmbda*layers[l].w
            # update bias
            layers[l].b = layers[l].b - eta*delta_L[0, :]

        # calculate delta_L for first hidden layer
        delta_L = (delta_L @ layers[1].w.T) * layers[0].a_deriv
        # update weights
        layers[0].w = layers[0].w - eta*(X.T @ delta_L)
                      - 2*eta*lmbda*layers[0].w
        # update bias
        layers[0].b = layers[0].b - eta*delta_L[0, :]

    def backprop_two_layers(self, X, y, eta, lmbda):
        """ Backpropagation with no hidden layers.

        Keyword arguments:
        X -- matrix with training data and its input features
        y -- the true classification
        eta -- learning rate
        lmbda - regularization parameter
        """
        self.feedforward(X)
        layers = self.layers
        dCda = self.cost.deriv(layers[-1].a, y)
        # delta_L = dCda * layers[-1].a_deriv
        delta_L = layers[-1].a - y
        # update weights
        layers[0].w = layers[0].w - eta*(X.T @ delta_L) \
                      - 2*eta*lmbda*layers[0].w
        # update bias
        layers[0].b = layers[0].b - eta*delta_L[0, :]
