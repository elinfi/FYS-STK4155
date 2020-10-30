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


epochs = 100
batch_size = 100
n_neurons_layer1 = 100
n_neurons_layer2 = 50
n_neurons_layer3 = 1
n = 50
eta = 0.5

def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta):
    model = Sequential()
    model.add(Dense(n_neurons_layer1, activation='sigmoid'))
    model.add(Dense(n_neurons_layer2, activation='sigmoid'))
    # model.add(Dense(n_categories, activation=tf.identity))
    model.add(Dense(n_categories, activation='softmax'))

    sgd = optimizers.SGD(lr=eta)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

if __name__ == '__main__':

    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x, y = np.meshgrid(x, y)
    z = np.ravel(f.FrankeFunction(x, y))
    z = z.reshape(-1, 1)

    data = DataPrep(x, y, z, degree=5)
    X_train, X_test, z_train, z_test = data()

    DNN = create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, \
                                      n_outputs, eta=eta)
    DNN.fit(X_train, z_train, epochs=epochs, batch_size=X_train.shape[0], verbose=0)
    scores = DNN.evaluate(X_test, z_test)


    print("Learning rate = ", eta)
    print("Test accuracy: %.3f" % scores)
    print()
