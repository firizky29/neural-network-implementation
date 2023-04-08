import numpy as np
from ActivationFunction import *


class Layer:
    def __init__(self, weight, bias, activation, is_output=True):
        self.b = bias
        self.w = np.transpose(weight)
        self.activation_name = activation
        self.is_output = is_output

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

    def layer_delta(self, input: np.ndarray, *, error_diff: np.ndarray = None, expected: np.ndarray = None) -> np.ndarray:
        wtx = np.matmul(input, self.w)
        bias = self.b
        net = wtx + bias

        if self.is_output:
            if not isinstance(expected, np.ndarray):
                raise Exception(
                    "expected is required to get output layer delta and must be np.ndarray")

            res = self.activation_function.calculate(net)
            o_diff = self.activation_function.error_differential(
                expected, res)
            return self.activation_function.delta(net, o_diff)
        elif isinstance(error_diff, np.ndarray):
            return self.activation_function.delta(net, error_diff)
        else:
            raise Exception(
                "target_differential is required to get hidden layer delta")

    def weight_diff(self, input: np.ndarray, delta: np.ndarray):
        row_weigths = []

        # Do face-splitting product
        # Ref: https://en.wikipedia.org/wiki/Khatri-Rao_product#Face-splitting_product
        for i in range(len(input)):
            row_weigths.append(np.kron(delta[i], input[i]))

        weight_delta = np.array(np.sum(row_weigths, axis=0))
        bias_delta = delta.sum(axis=0)

        weight_dw = weight_delta.reshape(self.w.shape[1], self.w.shape[0])
        return weight_dw, bias_delta

    def error_diff(self, delta: np.ndarray):
        weight = np.transpose(self.w)
        return np.matmul(delta, weight)
