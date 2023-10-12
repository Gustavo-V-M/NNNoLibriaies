import numpy as np 
import Activation 
import Loss


class Layer:
    def __init__(self, num_neurons, num_inputs, learning_rate):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.bias = np.zeros((1, num_neurons))

    def update(self, delta_weights, delta_bias):
        self.weights = self.weights - (self.learning_rate * delta_weights)
        self.bias = self.bias - (self.learning_rate * delta_bias)


class Hidden(Layer):
    
    def forward(self, input):
        relu = Activation.ReLU()

        self.pre_activation = np.dot(input, self.weights) + self.bias
        self.output = relu.forward(self.pre_activation)

    def backwards(self, delta, num_inputs):
        relu = Activation.ReLU()

        self.deltaWeights = np.dot(self.output.T, delta) / num_inputs
        self.deltaBias = np.sum(delta.T, axis=0) / num_inputs

        return np.dot(delta, self.weights) * relu.derivative(self.pre_activation) 

class Output(Layer):
    def forward(self, input):
        softmax = Activation.Softmax()

        self.pre_activation = np.dot(input, self.weights) + self.bias
        self.output = softmax.forward(self.pre_activation)

    def backwards(self, y_expect, input_size):

        squaredError = Loss.SquaredError()
        
        delta = squaredError.derivative(y_expect, self.output)

        self.deltaWeights = np.dot(self.output.T, delta) / input_size
        self.deltaBias = np.sum(delta, axis=0) / input_size 

        return delta 




