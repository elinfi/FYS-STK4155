import numpy as np

class MSE:
    def __call__(self, z_tilde, z):
        return np.mean((z_tilde - z)**2)

    def deriv(self, z_tilde, z):
        return 2*(z_tilde - z)/z.shape[0]

class Accuracy:
    def __call__(self, z_tilde, z):
        return np.mean(z_tilde == z)

class CrossEntropy:
    def __call__(self, z_tilde, z):
        return -np.log(np.prod(np.pow(z_tilde,z)))

    def deriv(self, z_tilde, z):
        # return -np.sum(np.sum(z/z_tilde, axis=1), axis=0)
        return z_tilde - z
