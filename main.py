import numpy as np
from nn import NeuralNet
from fc import FullyConnectedLayer
from activation import ActivationLayer, relu, relu_derivative

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])  # input_size=[1,2]
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])          # output_size=[1,1]

# network
net = NeuralNet()
net.add(FullyConnectedLayer(2, 3))
net.add(ActivationLayer(relu, relu_derivative))
net.add(FullyConnectedLayer(3, 1))
net.add(ActivationLayer(relu, relu_derivative))

# train
net.fit(x_train, y_train, epochs=5000, learning_rate=0.01)

# test
out = net.predict(x_train)
print(out)