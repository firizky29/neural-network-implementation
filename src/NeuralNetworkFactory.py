from Layer import Layer
from NeuralNetwork import NeuralNetwork
import json

class NeuralNetworkFactory:
    def fromJson(self, filename: str) -> NeuralNetwork:
        try:
            with open(filename, 'r') as f:
                model = json.load(f)
        except FileNotFoundError:
            print("File not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format.")
        else:
            nn = NeuralNetwork()
            for layer in model.get('layers'):
                nn.addLayer(Layer(layer['weight'], layer['bias'], layer['activation']))
            return nn

nn = NeuralNetworkFactory().fromJson('src/ffnn.json')
nn.draw()