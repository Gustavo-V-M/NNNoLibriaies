import numpy as np 

class Dense:
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.bias = np.zeros((1, num_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias

    def backwards_output(self, dLoss, activated_outputs):
        self.detaBias = [0]
        self.delta = dLoss

        self.deltaWeights = np.dot(activated_outputs.T, dLoss) / self.num_inputs 


