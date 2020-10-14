import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create data set
x = np.random.rand(100)
y = 2.0 + 5*x*x + 0.9*np.random.randn(100)

# Design matrix for second order polynomial
X = np.zeros((len(x), 3))
X[:, 0] = 1
X[:, 1] = x
X[:, 2] = x**2

# Parametrization of the data set fitting a second order polynomial
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
ytilde = X @ beta

# Use Scikit-learn's linear regression
clf = skl.LinearRegression().fit(X, y)
ytilde_skl = clf.predict(X)


print("PARAMETRIZATION")
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y, ytilde))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % r2_score(y, ytilde))

print("SCIKIT-LEARN")
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y, ytilde_skl))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % r2_score(y, ytilde_skl))

# Split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# matrix inversion to find beta
beta_skl = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(beta)
print(clf)

# make the prediction
# ytilde_skl = X_train @ beta_skl
ypredict = X_test @ beta_skl

plt.plot(x, y, 'b.')
plt.plot(x, ytilde, 'g.')
plt.plot(x, ytilde_skl, 'r.')
plt.show()
