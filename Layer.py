import numpy as np 
import Activation 
import Loss
class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.bias = np.zeros((1, num_neurons))




class Hidden(Layer):
    def forward(self, input):
        relu = Activation.ReLU
        
        self.pre_activation = np.dot(input, self.weights) + self.bias
        self.output = relu().forward(self.pre_activation)

class Output(Layer):
    def forward(self, input):
        softmax = Activation.Softmax

        self.pre_activation = np.dot(input, self.weights) + self.bias
        self.output = softmax().forward(self.pre_activation)

    def backwards(self, y_expect):

        squaredError = Loss.SquaredError
        
        self.delta = squaredError.derivative(y_expect=y_expect, y_pred=self.output)





