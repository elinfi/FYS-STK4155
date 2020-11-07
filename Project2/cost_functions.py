import numpy as np

class MSE:
    """Calculates the mean squared error cost function and its derivative."""
    def __call__(self, z_tilde, z):
        """Calculates the mean squared error.

        Keyword arguments:
        z_tilde -- predicted data
        z -- true data

        Return value:
        MSE -- mean squared error of predicted and true data
        """
        MSE = np.mean((z_tilde - z)**2)
        return MSE

    def deriv(self, z_tilde, z):
        """Computes the derivative of mean squared error cost function.

        Keyword arguments:
        z_tilde -- predicted data
        z -- true data

        Return value:
        deriv -- the derivative of the mean squared error
        """
        deriv = 2*(z_tilde - z)/z.shape[0]
        return deriv

class CrossEntropy:
    """
    Calculates the categorical cross entropy cost function and its derivative.
    """
    def __call__(self, z_tilde, z):
        """Calculates the categorical cross entropy.

        Keyword arguments:
        z_tilde -- predicted data
        z -- true data

        Return value:
        cross_entropy -- categorical cross entropy of predicted and true data
        """
        cross_entropy = -np.log(np.prod(np.pow(z_tilde,z)))
        return cross_entropy

    def deriv(self, z_tilde, z):
        """Computes the derivative of categorical cross entropy cost function.

        Keyword arguments:
        z_tilde -- predicted data
        z -- true data

        Return value:
        deriv -- the derivative of the categorical cross entropy cost function
                 when using softmax as the output activation function
        """
        # return -np.sum(np.sum(z/z_tilde, axis=1), axis=0)
        # return -np.sum(z/z_tilde)
        return (z_tilde - z)

class Accuracy:
    """ Calculates the accuracy score metric."""
    def __call__(self, z_tilde, z):
        """Calculates the accuracy score.

        Keyword arguments:
        z_tilde -- predicted data
        z -- true data

        Return value:
        accuracy -- accuracy of the predicted value
        """
        accuracy = np.mean(z_tilde == z)
        return accuracy
