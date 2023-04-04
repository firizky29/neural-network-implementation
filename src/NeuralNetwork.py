import Layer
import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer: Layer):
        self.layers.append(layer)

    def calculate(self, input: np.ndarray) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.calculate(output)
        return output

    def draw(self):
        pass

    def fit(self, input: np.ndarray, target: np.ndarray):
        pass
