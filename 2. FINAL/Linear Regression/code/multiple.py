## Multiple Regression Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la


# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.
def mean_std(X):
  mean = np.zeros(X.shape[1])
  std = np.ones(X.shape[1])

  ## Your code here. Hint: You can use numpy to compute mean and std.
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)

  return mean, std


# Standardize the features of the examples in X by subtracting their mean and
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):
  S = np.zeros(X.shape)
  S = (X - mean) / std

  return S

# Read data matrix X and labels t from text file.
def read_data(file_name):

  data=np.loadtxt(file_name)
  X = data[:, :3]
  t = data[:, 3]

  return X, t


# Implement gradient descent algorithm to compute w = [w0, w1, ..].
def train(X, t, eta, epochs):
  costs=[]
  ep=[]
  w = np.zeros(X.shape[1])
  for epoch in range(epochs):

    grad = compute_gradient(X, t, w)
    w = w - eta * grad

    if epoch % 10 == 0:

      cost = compute_cost(X, t, w)
      costs.append(cost)
      ep.append(epoch)



  return w,ep,costs

# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):

  N = X.shape[0]
  rmse = np.sqrt(1/N * np.sum((t - X.dot(w))**2))

  return rmse


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):

  N = X.shape[0]
  cost = 1/N * np.sum((t - X.dot(w))**2)

  return cost


# Compute gradient of the objective function (cost) on dataset (X, t).
def compute_gradient(X, t, w):

  N = X.shape[0]
  grad = np.zeros(w.shape)

  grad = -2/N * X.T.dot(t - X.dot(w))

  return grad


##======================= Main program =======================##
parser = argparse.ArgumentParser('Multiple Regression Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='D:\AIUB\Academics\Semester 10\ML\Final Assignment\Part 1\linear_regression\data\multiple',
                    #default='../data/multiple',  #this line is not working
                    help='Directory for the multiple regression houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")


#  Standardize the training and test features using the mean and std computed over *training*.
mean, std = mean_std(Xtrain)
Xtrain = standardize(Xtrain, mean, std)
Xtest = standardize(Xtest, mean, std)
#  Make sure you add the bias feature to each training and test example.
#  The bias features should be a column of ones addede as the first columns of training and test examples
Xtrain = np.hstack((np.ones((Xtrain.shape[0], 1)), Xtrain))
Xtest = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))

# Computing parameters for each training method for eta=0.1 and 200 epochs
eta=0.1
epochs=200
w,eph,costs=train(Xtrain,ttrain,eta,epochs)



# Print model parameters.
print('Params GD: ', w)

# Print cost and RMSE on training data.
print('Training RMSE: %0.2f.' % compute_rmse(Xtrain, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(Xtrain, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSE: %0.2f.' % compute_rmse(Xtest, ttest, w))
print('Test cost: %0.2f.' % compute_cost(Xtest, ttest, w))

# Plotting Epochs vs. cost for Gradient descent methods
plt.figure(figsize=(10, 6))
plt.xlabel(' epochs')
plt.ylabel('cost')
plt.yscale('log')
plt.plot(eph, costs, 'bo-', label='train_Jw_gd')
plt.title('Epochs vs. Cost for Gradient Descent')
plt.legend()
plt.grid(True)
plt.savefig('gd_cost_multiple.png')
plt.show()
plt.close()