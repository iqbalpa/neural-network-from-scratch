from base import Layer
from loss import mse, mse_derivative
from fc import FullyConnectedLayer

class NeuralNet:
  def __init__(self):
    self.loss = mse
    self.loss_derivative = mse_derivative
    self.layers = []

  # add layer to nn
  def add(self, layer):
    self.layers.append(layer)

  # training 
  def fit(self, x_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
      # iterate over data sample
      err = 0
      for i in range(len(x_train)):
        # perform forward prop from first to last layer
        out = x_train[i]
        for layer in self.layers:
          out = layer.forward_prop(out)
        # compute loss
        loss = self.loss(y_train[i], out)
        err += loss
        # perform backward prop from last to first layer
        error = self.loss_derivative(y_train[i], out)
        for layer in reversed(self.layers):
          error = layer.backward_prop(error, learning_rate)
      # mean error
      avg_err = err / len(x_train)
      print(f"Epoch {epoch+1}/{epochs}  Error: {avg_err}")

  # prediction 
  def predict(self, input_data):
    res = []
    for i in range(len(input_data)):
      out = input_data[i]
      for layer in self.layers:
        out = layer.forward_prop(out)
      res.append(out)
    return res
