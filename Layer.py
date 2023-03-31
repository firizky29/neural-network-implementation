import numpy as np
import math
import Neuron

class Layer:
    neurons = [];
    
    def __init__(self, num_of_neurons, num_of_input, next_layer):
        
        for i in range(num_of_neurons):
            neuron = Neuron()
            self.neurons.append(neuron)
        self.num_of_input = num_of_input
        self.next_layer = next_layer
        self.w = np.zeros((num_of_neurons, num_of_input))
        self.b = np.zeros(num_of_neurons)
    
    def getNeurons(self) -> list:
        return self.neurons
    
    def setWeight(self) :
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            for j in range(len(neuron.weight)):
                self.w[i][j] = neuron.weight[j]
    
    def setBias(self, bias):
        self.b = bias
        
    def calculate(self,input) -> np.ndarray:
        # calculate the matrix output to the next layer
        input = np.transpose(input)
        bias = np.transpose(self.b)
        wtx = np.matmul(self.w, input)
        return wtx + bias
    
class SoftMaxLayer(Layer):
    def __init__(self, num_of_neurons, next_layer):
        super().__init__(num_of_neurons, next_layer)
    
    def calculate(self, input) -> np.ndarray:
        linear_output = super.calculate(self, input)
        self.exp_sum = 0
        self.exp_output = []
        for row in linear_output:
            self.exp_sum += row[0]
            self.exp_output.append(row[0])
        
        output = np.ndarray(shape=(0,1))
        for i in range(self.num_of_neurons):
            output = np.insert(output, i, [self.exp_output[i]/self.exp_sum], axis=0)
        # calculate e^x for all x in output
        # calculate the sum of all e^x
        # calculate the ratio of e^x to the sum
        return output    