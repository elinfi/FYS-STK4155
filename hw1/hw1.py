import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from IPython.display import display

# R-squared (variance) score
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model)**2) / \
           np.sum((y_data - np.mean(y_data)) ** 2)

# Mean square error
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2)/n

# Relative error
def RelativeError(y_data, y_model):
    return abs((y_data - y_model)/y_data)

# Read data from cvs file
Data  = pd.read_csv("EoS.csv", header = None, sep = ',')
# Convert data to numpy array
EoS = Data.to_numpy()
x = EoS[:, 0]
y = EoS[:, 1]

# Create the design matrix for 3rd order polynomial
X = np.zeros((len(EoS), 4))
X[:, 0] = 1
X[:, 1] = x
X[:, 2] = x**2
X[:, 3] = x**3

# nice printing of design matrix
DesignMatrix = pd.DataFrame(X)
DesignMatrix.index = x
DesignMatrix.columns = ['1', 'x', 'x^2', 'x^3']
display(DesignMatrix)

# Matrix inversion to find beta
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# make the prediction
ytilde = X.dot(beta)

print("Variance score:  {}".format(R2(y, ytilde)))
print("Mean squared error: {}".format(MSE(y, ytilde)))
# print("Relative Error: {}".format(RelativeError(y, ytilde)))

# plot the polynomial fit and the original data
plt.plot(x, ytilde, 'b', label='polynomial fit')
plt.plot(x, y, 'r.', label='data points')
plt.legend()
plt.show()
