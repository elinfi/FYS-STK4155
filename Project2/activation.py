import numpy as np

class Identity:
    def __call__(self, z):
        return z

    def deriv(self, z):
        return np.ones(z.shape)

class LeakyRELU:
    def __call__(self, z):
        if z < 0:
            return 0.01*z
        else:
            return z

    def deriv(self, z):
        if z < 0:
            return 0.01
        else:
            return 1

class RELU:
    def __call__(self, z):
        if z > 0:
            return z
        else:
            return 0

    def deriv(self, z):
        if z > 0:
            return 1
        else:
            return 0

class Sigmoid:
    def __call__(self, z):
        return 1/(1 + np.exp(-z))

    def deriv(self, z):
        return np.exp(-z)/(1 + np.exp(-z))**2

class Softmax:
    def __call__(self, z):
        return np.exp(z)/np.sum(np.exp(z), axis=1)[:, None]

    def deriv(self, z):
        return self.__call__(z) - (self.__call__(z))**2
        # z * np.ones(len(z)).T.dot(np.eye(len(z)) - np.ones(len(z))*z.T)
