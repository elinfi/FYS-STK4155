import numpy as np
import functions as f
import tensorflow as tf
from data_prep import DataPrep
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

class Keras:
    def __init__(self, neurons, n_outputs, eta, loss, metrics, input_act, output_act):
        self.neurons = neurons
        self.n_outputs = n_outputs
        self.eta = eta
        self.loss = loss
        self.metrics = metrics
        self.input_act = input_act
        self.output_act = output_act

    def create_neural_network_keras(self):
        model = Sequential()

        for i in range(len(self.neurons)):
            model.add(Dense(self.neurons[i], activation=self.input_act))
        model.add(Dense(self.n_outputs, activation=self.output_act))

        sgd = optimizers.SGD(lr=self.eta)
        # sgd = optimizers.Adam(lr=self.eta)
        model.compile(loss=self.loss, optimizer=sgd, metrics=self.metrics)

        return model
