import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]
# biases = [2, 3, 0.5]
#
# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
#
# output = np.dot(inputs, np.array(weights).T) + biases

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # To skip transpose
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def foward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


lay1 = Layer_Dense(4, 5)
lay2 = Layer_Dense(5, 2)

lay1.foward(inputs)

lay2.foward(lay1.output)

print(lay2.output)
