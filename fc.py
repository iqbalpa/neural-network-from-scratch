import numpy as np
from base import Layer

class FullyConnectedLayer(Layer):
  def __init__(self, input_size, output_size):
    # initialize random weights, shape = [input_size, output_size]
    self.weight = np.random.rand(input_size, output_size) 
    # initialize random bias, shape = [1, output_size]
    self.bias = np.random.rand(1, output_size)
  
  def forward_prop(self, input_data):
    # input_data is a vector, shape = [1, input_size]
    self.input = np.array(input_data)
    # res is a vector, shape = [output_size, 1]
    self.output = np.dot(self.input, self.weight) + self.bias
    return self.output
  
  def backward_prop(self, output_error, learning_rate):
    # gradients
    # input error
    dE_dX = np.dot(output_error, self.weight.T)
    # weights error
    dE_dW = np.dot(self.input.T, output_error)
    # bias error
    dE_db = output_error
    # update weights and bias (perform gradient descent)
    self.weight -= learning_rate * dE_dW
    self.bias -= learning_rate * dE_db
    return dE_dX


if __name__ == '__main__':
  fc = FullyConnectedLayer(3, 2)
  x = [1, 2, 3]
  res = fc.forward_prop(x)
  print(res)
  print(fc.weight, fc.bias)

  output_error = np.array([[0.1, -0.2]])  
  learning_rate = 0.01
  res = fc.backward_prop(output_error, learning_rate)
  print(res)
  print(fc.weight, fc.bias)

