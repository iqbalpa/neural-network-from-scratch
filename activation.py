import numpy as np
from base import Layer

class ActivationLayer(Layer):
  def __init__(self, activation, activation_derivative):
    # forward prop
    self.activation = activation
    # backward prop
    self.activation_derivative = activation_derivative
  def forward_prop(self, input_data):
    self.input = input_data
    self.output = self.activation(self.input)
    return self.output
  def backward_prop(self, output_error, learning_rate=None):
    return output_error * self.activation_derivative(self.input)


# Activation function and its derivative
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

if __name__ == '__main__':
    # Example usage
    act_layer = ActivationLayer(activation=relu, activation_derivative=relu_derivative)
    x = np.array([[1, -2, 3], [-1, 2, -3]])  # Example input data
    print("Forward Propagation Result:")
    print(act_layer.forward_prop(x))
    
    output_error = np.array([[1, 1, 1], [1, 1, 1]])  # Example output error
    print("Backward Propagation Result:")
    print(act_layer.backward_prop(output_error, learning_rate=0.01))