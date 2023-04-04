import numpy as np
import math
from ActivationFunction import *

class Layer:
    neurons = [];

    def __init__(self, weight, bias, activation):
        self.b = bias
        self.w = weight
        if activation == "linear":
            self.activation_function = ActivationFunction()
        elif activation == "relu":
            self.activation_function = ActivationFunction()
        elif activation == "sigmoid":
            self.activation_function = ActivationFunction()
        else:
            self.activation_function = ActivationFunction()
        self.num_of_neurons = len(weight)
        self.num_of_input = len(weight[0])
        
        
    def calculate(self,input) -> np.ndarray:
        # calculate the matrix output to the next layer
        # input matriks kolom
        # bias = np.transpose(self.b)
        
        wtx = np.matmul(self.w, input)
        bias = self.b
        bias = np.resize(bias, (len(bias), len(input[0])))
        net = wtx + bias
        return self.activation_function.calculate(net)
        

#class SoftMaxLayer(Layer):
#    def __init__(self, weight, bias, activation):
#        super().__init__(weight, bias, activation)
#    
#    def calculate(self, input) -> np.ndarray:
#        linear_output = super.calculate(self, input)
#        self.exp_sum = 0
#        self.exp_output = []
#        for row in linear_output:
#            self.exp_sum += row[0]
#            self.exp_output.append(row[0])
#
#        for i in range(self.num_of_neurons):
#            linear_output[i] = np.insert(output, i, [self.exp_output[i]/self.exp_sum], axis=0)
#        # calculate e^x for all x in output
#        # calculate the sum of all e^x
#        # calculate the ratio of e^x to the sum
#        return output    