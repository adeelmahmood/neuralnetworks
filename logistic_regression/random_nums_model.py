import numpy as np
import matplotlib.pyplot as plt
import math

num_features = 100
num_samples = 50
iterations = 1000
learning_rate = 0.009
verbose = False

np.random.seed(39)

training_set = np.random.randn(num_features, num_samples)
test_set = np.random.randn(num_features, math.ceil((num_samples*10)/100))
training_labels = np.random.choice(2, num_samples)
test_labels = np.random.choice(2, math.ceil((num_samples*10)/100))

if verbose:
  print('training set')
  print(training_set)
  print('training labels')
  print(training_labels)
  print('--')
  print('test set')
  print(test_set)
  print('test labels')
  print(test_labels) 

def sigmoid(z):
  s = 1 / (1 + np.exp(-z))
  return s

def initialize_weights_and_biases(dim):
  w = np.zeros((dim, 1))
  b = 0
  return w,b

def forward(w, b, X, Y):
  m = X.shape[1]
  
  # compute activation matrix
  A = sigmoid(np.matmul(w.T, X) + b)

  # compute cost function
  cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

  # compute partial derivatives
  dw = (1/m) * (np.dot(X, (A-Y).T))
  db = (1/m) + np.sum(A-Y)

  cost = np.squeeze(cost)

  grads = {"dw": dw, "db": db}

  return grads, cost

def optimize(w, b, X, Y, iterations, learning_rate, verbose = False):
  costs = []
  for i in range(iterations):
    grads, cost = forward(w, b, X, Y)

    dw = grads["dw"]
    db = grads["db"]

    w = w - learning_rate * dw;
    b = b - learning_rate * db;

    if i % 100 == 0:
      print("cost after iteration " + str(i) + ": " + str(cost))
      costs.append(cost)

  params = {"w": w, "b": b}
  grads = {"dw": dw, "db": db}

  return params, grads, costs

def predict(w, b, X):
  m = X.shape[1]
  predictions = np.zeros((1, m))

  A = sigmoid(np.matmul(w.T, X) + b)

  for i in range(A.shape[1]):
    if(A[0,i] < 0.5):
      predictions[0,i] = 0
    else:
      predictions[0,i] = 1 

  return predictions

def model(training_set, training_labels, test_set, test_labels, iterations, learning_rate, verbose = False):
  w, b = initialize_weights_and_biases(training_set.shape[0])
  if verbose:
    print('weight = ' + str(w) + ', bias = ' + str(b))

  # run the forward and backward propagation to figure out the best weights
  params, grads, costs = optimize(w, b, training_set, training_labels, iterations, learning_rate, verbose)
  if verbose: 
   print('after optimization, weights and biases')
   print(params)
   print(grads)

  w = params["w"]
  b = params["b"]

  predictions_train = predict(w, b, training_set)
  predictions_test = predict(w, b, test_set)

  print("modal accuracy with training set: {} %".format(100 - np.mean(np.abs(predictions_train - training_labels)) * 100))
  print("modal accuracy with test set: {} %".format(100 - np.mean(np.abs(predictions_test - test_labels)) * 100))

  return {"costs": costs, "predictions_train": predictions_train, "predictions_test": predictions_test}

# run the model  
d = model(training_set, training_labels, test_set, test_labels, iterations, learning_rate, verbose)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

