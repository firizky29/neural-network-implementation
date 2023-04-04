from NeuralNetwork import NeuralNetwork
import json

class NeuralNetworkFactory:
    def __init__(self):
        pass

    def fromJson(self, filename: str) -> NeuralNetwork:
        try:
            with open(filename, 'r') as f:
                model = json.load(f)
        except FileNotFoundError:
            print("File not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format.")
        else:
            layers = model["neural_network"]["layers"]
            print(layers)

NeuralNetworkFactory().fromJson('src/ffnn.json')