import numpy as np 

class ReLU:
    def forward(self, input):
        return np.maximum(np.zeros(input.shape), input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)



class Softmax:
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        


