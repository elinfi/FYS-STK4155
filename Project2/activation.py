import numpy as np

class Sigmoid:
    def __call__(self, z):
        return 1/(1 + np.exp(-z))

    def deriv(self, z):
        return np.exp(-z)/(1 + np.exp(-z))**2

class Identity:
    def __call__(self, z):
        return z

    def deriv(self, z):
        return z**0
