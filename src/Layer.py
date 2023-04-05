import numpy as np
from ActivationFunction import *


class Layer:
    def __init__(self, weight, bias, activation):
        self.b = bias
        self.w = np.transpose(weight)
        if activation == "linear":
            self.activation_function = LinearActivation()
        elif activation == "relu":
            self.activation_function = ReluActivation()
        elif activation == "sigmoid":
            self.activation_function = SigmoidActivation()
        else:
            self.activation_function = SoftmaxActivation()
        self.num_of_neurons = len(weight)
        self.num_of_input = len(weight[0])

    def calculate(self, input) -> np.ndarray:
        # calculate the matrix output to the next layer
        # input matriks kolom
        wtx = np.matmul(input, self.w)
        bias = self.b
        net = wtx + bias
        return self.activation_function.calculate(net)
