import numpy as np

class Identity:
    """Computes the identity activation function and its derivative."""
    def __call__(self, z):
        """Applies the identity function.

        Keyword arguments:
        z -- input data

        Return value:
        z -- output data from identity function
        """
        return z

    def deriv(self, z):
        """Computes the derivative of the identity function.

        Keyword arguments:
        z -- input data

        Return value:
        np.ones(z.shape) -- the derivative of the identity function
        """
        return np.ones(z.shape)

class LeakyRELU:
    """Computes the leaky ReLU activation function and its derivative."""
    def __call__(self, z):
        """Applies the leaky ReLU activation function.

        Keyword arguments:
        z -- input data

        Return value:
        z -- output data from leaky ReLU function
        """
        boolean = z < 0
        z[boolean] *= 0.01
        return z
        # if z < 0:
        #     return 0.01*z
        # else:
        #     return z

    def deriv(self, z):
        """Computes the derivative of the leaky ReLU function.

        Keyword arguments:
        z -- input data

        Return value:
        z -- the derivative of the leaky ReLU function
        """
        boolean = z < 0
        z[boolean] = 0.01
        z[~boolean] = 1
        return z
        # if z < 0:
        #     return 0.01
        # else:
        #     return 1

class RELU:
    """Computes the ReLU activation function and its derivative."""
    def __call__(self, z):
        """Applies the ReLU activation function.

        Keyword arguments:
        z -- input data

        Return value:
        z -- output data from ReLU function
        """
        boolean = z <= 0
        z[boolean] = 0
        return z
        # if z > 0:
        #     return z
        # else:
        #     return 0

    def deriv(self, z):
        """Computes the derivative of the ReLU function.

        Keyword arguments:
        z -- input data

        Return value:
        z -- the derivative of the ReLU function
        """
        boolean = z > 0
        z[boolean] = 1
        z[~boolean] = 0
        return z
        # if z > 0:
        #     return 1
        # else:
        #     return 0

class Sigmoid:
    """Computes the sigmoid activation function and its derivative."""
    def __call__(self, z):
        """Applies the sigmoid activation function.

        Keyword arguments:
        z -- input data

        Return value:
        sigmoid -- output data from sigmoid function
        """
        sigmoid = 1/(1 + np.exp(-z))
        return sigmoid
        # return np.exp(z)/(np.exp(z) + 1)

    def deriv(self, z):
        """Computes the derivative of the sigmoid function.

        Keyword arguments:
        z -- input data

        Return value:
        deriv -- the derivative of the sigmoid function
        """
        # return np.exp(-z)/(1 + np.exp(-z))**2
        deriv = self.__call__(z) - self.__call__(z)**2
        return deriv

class Softmax:
    """Computes the softmax activation function and its derivative."""
    def __call__(self, z):
        """Applies the softmax activation function.

        Keyword arguments:
        z -- input data

        Return value:
        softmax -- output data from sigmoid function
        """
        max_z = np.max(z, axis=1).reshape(-1, 1)
        softmax = np.exp(z - max_z)/np.sum(np.exp(z - max_z), axis=1)[:, None]
        return softmax

    def deriv(self, z):
        """Computes the derivative of the softmax function.

        Keyword arguments:
        z -- input data

        Return value:
        deriv -- the derivative of the softmax function
        """
        deriv = self.__call__(z) - self.__call__(z)**2
        return deriv
