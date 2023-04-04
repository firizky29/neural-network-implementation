import numpy as np
import src.Neuron as Neuron

class Layer:
    neurons = []
    
    def __init__(self, num_of_neurons, next_layer):
        
        for i in range(num_of_neurons):
            neuron = Neuron()
            self.neurons.append(neuron)
        self.next_layer = next_layer
        self.w = np.zeros( num_of_neurons+1 )
        self.w[0] = 1
    
    def getNeurons(self) -> list:
        return self.neurons
    
    def calculate(input) -> np.ndarray:
        # calculate the matrix output to the next layer
        pass
    
class SoftMaxLayer(Layer):
    def __init__(self, num_of_neurons, next_layer):
        super().__init__(num_of_neurons, next_layer)
    
    def calculate(input) -> np.ndarray:
        linear_output = super.calculate(input)
        # calculate e^x for all x in output
        # calculate the sum of all e^x
        # calculate the ratio of e^x to the sum
        return np.ndarray()
        