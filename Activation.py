import numpy as np 

class ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)



class Softmax:
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        


