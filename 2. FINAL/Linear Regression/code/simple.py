## Simple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la


# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)
  return mean, std


# Standardize the features of the examples in X by subtracting their mean and 
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
  S = (X - mean) / std

  return S

# Read data matrix X and labels t from text file.
def read_data(file_name):
  data=np.loadtxt(file_name)
  X = data[:, 0:1]  
  t = data[:, 1] 
  return X, t


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, t, eta, epochs):
  
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
  #  Use 'compute_gradient' function below to find gradient of cost function and update w each epoch.
  #  Compute and append cost and epoch number to variables costs and ep every 10 epochs.
  for epoch in range(epochs):
    gradient = compute_gradient(X, t, w)
    w = w - eta * gradient

    if epoch % 10 == 0:
      cost = np.sum((X @ w - t) ** 2) / (2 * len(X))
      costs.append(cost)
      ep.append(epoch)

  return w,ep,costs

# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
  predictions = X @ w
  errors = predictions - t
  squared_errors = errors ** 2
  mean_squared_error = np.mean(squared_errors)
  rmse = np.sqrt(mean_squared_error)

  return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
  predictions = X @ w
  errors = predictions - t
  squared_errors = errors ** 2
  cost = np.mean(squared_errors) / 2

  return cost


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X, t, w):
  grad = np.zeros(w.shape)
  predictions = X @ w
  errors = predictions - t
  grad = X.T @ errors / len(X)

  return grad


# BONUS: Implement stochastic gradient descent algorithm to compute w = [w0, w1].
def train_SGD(X, t, eta, epochs):
#  YOUR CODE here:
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
  #  YOUR CODE here. Implement stochastic gradient descent to compute w for given epochs. 
  #  Compute and append cost and epoch number to variables costs and ep every 10 epochs.

  return w,ep,costs


##======================= Main program =======================##
parser = argparse.ArgumentParser('Simple Regression Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='D:\AIUB\Academics\Semester 10\ML\Final Assignment\Part 1\linear_regression\data\simple',
                    #default='../data/simple/',  #this line is not working
                    help='Directory for the simple houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

# Compute the mean and standard deviation of the training data.
mean, std = mean_std(Xtrain)

# Standardize the training and test features using the mean and std computed over training.
Xtrain = standardize(Xtrain, mean, std)
Xtest = standardize(Xtest, mean, std)

# Add the bias feature (a column of ones) as the first column of the training and test examples.
Xtrain = np.hstack((np.ones((Xtrain.shape[0], 1)), Xtrain))
Xtest = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))


# Computing parameters for each training method for eta=0.1 and 200 epochs
eta=0.1
epochs=200

w,eph,costs=train(Xtrain,ttrain,eta,epochs)
#wsgd,ephsgd,costssgd=train_SGD(Xtrain,ttrain,eta,epochs)


# Print model parameters.
print('Params GD: ', w)
#print('Params SGD: ', wsgd)

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % compute_rmse(Xtrain, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(Xtrain, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % compute_rmse(Xtest, ttest, w))
print('Test cost: %0.2f.' % compute_cost(Xtest, ttest, w))

# Plotting epochs vs. cost for gradient descent methods
plt.figure(figsize=(10, 6))
plt.xlabel('epochs')
plt.ylabel('cost')
plt.yscale('log')
plt.plot(eph, costs, 'bo-', label='train_jw_gd')
plt.legend()
plt.savefig('gd_cost_simple.png')
plt.show()
plt.close()

# Plotting linear approximation for each training method
plt.figure(figsize=(10, 6))
plt.xlabel('Floor sizes')
plt.ylabel('House prices')
plt.plot(Xtrain[:, 1], ttrain, 'bo', label='Training data')  # Xtrain[:, 1] to exclude the bias feature
plt.plot(Xtest[:, 1], ttest, 'g^', label='Test data')  # Xtest[:, 1] to exclude the bias feature
plt.plot(Xtrain[:, 1], w[0] + w[1]*Xtrain[:, 1], 'b', label='GD')  # Xtrain[:, 1] to exclude the bias feature
plt.legend()
plt.savefig('train-test-line.png')
plt.show()
plt.close()
