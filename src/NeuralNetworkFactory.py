from Layer import Layer
from NeuralNetwork import NeuralNetwork
import json
import numpy as np


class NeuralNetworkFactory:
    def fromJson(self, filename: str) -> NeuralNetwork:
        try:
            with open(filename, 'r') as f:
                model = json.load(f)
        except FileNotFoundError:
            raise Exception("File not found.")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON format.")
        else:
            return self.fromModel(model)

    def fromModel(self, model) -> NeuralNetwork:
        nn = NeuralNetwork()
        for layer in model.get('layers'):
            nn.addLayer(
                Layer(layer['weight'], layer['bias'], layer['activation']))
        return nn

    def assistantJson(self, filename: str) -> NeuralNetwork:
        try:
            with open(filename, 'r') as f:
                model = json.load(f)
        except FileNotFoundError:
            raise Exception("File not found.")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON format.")
        else:
            layers = model["case"]["weights"]
            new_format = dict()
            new_format["layers"] = list()

            for i in range(len(layers)):
                layer_info = model["case"]["model"]["layers"][i]
                layer_weight = np.array(layers[i])

                b = layer_weight[0]
                w = np.transpose(layer_weight[1:])

                new_format["layers"].append({
                    "activation": layer_info["activation_function"],
                    "weight": w,
                    "bias": b
                })

            return self.fromModel(new_format)
