# Project 2 FYS-STK4155

## Classification and regression, from linear and logistic regression to neural networks

In this project, we study the stochastic gradient descent algorithm and a feedforward neural network with backpropagation, applied on the Franke function and the MNIST dataset for hand-written digits.

The project is done in collaboration with Elias Roland Udnæs, Jacob Lie and Jonas Thoen Faber.

The folder source includes the python programs for  the stochastic gradient descent algorithm and the set up of the neural network with backpropagation. In the folder plotting, you can find scripts for reproducing the heatmaps presented in the report. They contain analysis of epochs vs. minibatches and learning rate vs. penalty on both the Franke function and the MNIST data. To run with logistic regression, set '''no_hidden = True''' and change the backpropagation function to '''backpropagation_two_layers'''. In addition, the folder contains a comparison of our neural network with keras for both Franke function and MNIST. The final folder Figures contains the figures used in the report.
