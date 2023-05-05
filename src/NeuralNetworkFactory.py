from Layer import Layer
from NeuralNetwork import NeuralNetwork
import json
import numpy as np


class NeuralNetworkFactory:
    def from_json(self, filename: str) -> NeuralNetwork:
        try:
            with open(filename, 'r') as f:
                model = json.load(f)
        except FileNotFoundError:
            raise Exception("File not found.")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON format.")
        else:
            return self.from_model(model)

    def from_model(self, model) -> NeuralNetwork:
        nn = NeuralNetwork()
        for layer in model.get('layers'):
            nn.add_layer(
                Layer(layer['weight'], layer['bias'], layer['activation']))
        return nn

    def assistant_json(self, filename: str) -> NeuralNetwork:
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

            return self.from_model(new_format)
        
    def assistant_backward_json(self, filename: str) -> NeuralNetwork:
        try:
            with open(filename, 'r') as f:
                model = json.load(f)
        except FileNotFoundError:
            raise Exception("File not found.")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON format.")
        else:
            layers = model["case"]["initial_weights"]
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
            learning_parameters = model["case"]["learning_parameters"]
            
            learning_rate = learning_parameters["learning_rate"]
            batch_size = learning_parameters["batch_size"]
            max_iteration = learning_parameters["max_iteration"]
            error_threshold = learning_parameters["error_threshold"]



            target = np.array(model["case"]["target"])
            input = np.array(model["case"]["input"])

            return self.from_model(new_format) \
                        .set_learning_properties(
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            max_iter=max_iteration,
                            error_threshold=error_threshold
                        ).fit(input, target)
        
# nn = NeuralNetworkFactory().assistant_backward_json("sample/backward/linear.json")
# nn.fit()

# for layer in nn.layers:
#     print(layer.get_w())
#     print(layer.get_b())
    # print(layer.activation_function)