import numpy as np

class MSE:
    def __call__(self, z_tilde, z):
        return np.mean((z_tilde - z)**2)

    def deriv(self, z_tilde, z):
        return 2*(z_tilde - z)/z.shape[0]

class Accuracy:
    def __call__(self, z_tilde, z):
        return np.mean(z_tilde == z)
