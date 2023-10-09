import numpy as np

class SquaredError():
    def forward(self, y_expect, y_pred):
        self.output = (1/2) * (np.sum((y_expect - y_pred)**2))

    def derivative(self, y_expect, y_pred):
        return (y_expect - y_pred)





