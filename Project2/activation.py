import numpy as np

class Identity:
    def __call__(self, z):
        return z

    def deriv(self, z):
        return np.ones(z.shape)

class LeakyRELU:
    def __call__(self, z):
        boolean = z < 0
        z[boolean] *= 0.01
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
        boolean = z <= 0
        z[boolean] = 0
        return z
        # if z > 0:
        #     return z
        # else:
        #     return 0

    def deriv(self, z):
        boolean = z > 0
        z[boolean] = 1
        z[~boolean] = 0
        return z
        # if z > 0:
        #     return 1
        # else:
        #     return 0

class Sigmoid:
    def __call__(self, z):
        return 1/(1 + np.exp(-z))
        # return np.exp(z)/(np.exp(z) + 1)

    def deriv(self, z):
        # return np.exp(-z)/(1 + np.exp(-z))**2
        return self.__call__(z) - self.__call__(z)**2

class Softmax:
    def __call__(self, z):
        # print(z.shape)
        # print(np.exp(z).shape)
        # print(np.sum(np.exp(z), axis=1))
        # print(np.exp(z)/np.sum(np.exp(z), axis=1)[:, None])
        # print(z)
        max_z = np.max(z, axis=1).reshape(-1, 1)
        return np.exp(z - max_z)/np.sum(np.exp(z - max_z), axis=1)[:, None]

    def deriv(self, z):
        # print(self.__call__(z) - (self.__call__(z))**2)
        return self.__call__(z) - (self.__call__(z))**2

        # z * np.ones(len(z)).T.dot(np.eye(len(z)) - np.ones(len(z))*z.T)
