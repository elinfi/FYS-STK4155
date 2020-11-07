import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

class Keras:
    def __init__(self, neurons, n_outputs, eta, lmbda, loss, metrics, hidden_act,
                 output_act):
        """Set up neural network using keras.

        Keyword arguments:
        neurons -- list with number of neurons in each hidden layer
        n_outputs -- number of outputs from the network
        eta -- learning rate
        lmbda -- regularization parameter
        loss -- loss function
        metrics -- metrics
        hidden_act -- activation function used on hidden layers
        ooutput_act -- activation function used on output layer
        """
        self.neurons = neurons
        self.n_outputs = n_outputs
        self.eta = eta
        self.loss = loss
        self.metrics = metrics
        self.input_act = input_act
        self.output_act = output_act

    def create_neural_network_keras(self):
        """Create the neural network using keras with SGD as optimizer.

        Return value:
        model -- neural network using keras
        """
        model = Sequential()

        # create hidden layers
        for i in range(len(self.neurons)):
            model.add(Dense(self.neurons[i], activation=self.hidden_act,
                            kernel_regularizer=regularizers.l2(lmbda)))
        # create output layer
        model.add(Dense(self.n_outputs, activation=self.output_act,
                        kernel_regularizer=regularizers.l2(lmbda)))

        # set optimization method
        sgd = optimizers.SGD(lr=self.eta)
        # sgd = optimizers.Adam(lr=self.eta)

        # finish setup of neural network
        model.compile(loss=self.loss, optimizer=sgd, metrics=self.metrics)

        return model
