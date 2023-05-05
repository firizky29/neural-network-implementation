import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from Layer import Layer

class NeuralNetwork:
    def __init__(self):
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer):
        if len(self.layers) > 0:
            self.layers[-1].is_output = False

        self.layers.append(layer)

    def calculate(self, input: np.ndarray) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.calculate(output)
        return output

    def draw(self):
        def get_id(i, j):
            layer_id = ""
            for k in range(1, i+2):
                if (i >= 26**k):
                    i -= 26**k
                else:
                    for l in range(k):
                        layer_id += chr(ord('A') + i % 26)
                        i = i // 26
                    break
            layer_id = layer_id[::-1]
            return layer_id + str(j)

        def get_color_random():
            return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        labels = {}
        colors = []
        G = nx.Graph()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_color = get_color_random()
            if (i == 0):
                layer0_color = get_color_random()
                colors.append(layer0_color)
                G.add_node(get_id(i, 0))
                G.nodes[get_id(i, 0)]["subset"] = 0
                for j in range(self.layers[i].num_of_input):
                    colors.append(layer0_color)
                    G.add_node(get_id(i, j+1))
                    G.nodes[get_id(i, j+1)]["subset"] = 0
            if (i < len(self.layers) - 1):
                colors.append(layer_color)
                G.add_node(get_id(i+1, 0))
                G.nodes[get_id(i+1, 0)]["subset"] = i+1
                labels[get_id(i+1, 0)] = layer.activation_name
            for j in range(self.layers[i].num_of_neurons):
                colors.append(layer_color)
                # bias
                G.add_edge(get_id(i, 0), get_id(i+1, j+1), weight=layer.b[j])
                # input
                for k in range(self.layers[i].num_of_input):
                    G.add_edge(get_id(i, k+1), get_id(i+1, j+1),
                               weight=layer.w[k][j])
                G.nodes[get_id(i+1, j+1)]["subset"] = i+1
            if (i == len(self.layers) - 1):
                labels[get_id(i+1, 1)] = layer.activation_name

        pos = nx.multipartite_layout(G)
        nx.draw(G, pos, with_labels=True,
                font_weight='bold', node_color=colors, node_size=100, font_size=10)
        nx.draw_networkx_labels(
            G, pos, labels=labels, verticalalignment='bottom', horizontalalignment="left", font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(
            G, 'weight'), label_pos=0.8, font_size=8)
        plt.show()

    def fit(self, input: np.ndarray, target: np.ndarray, learning_rate: float = 0.1, batch_size=1, max_iter=1000, error_threshold:float =0.01) -> None:
        neural_network_learning = NeuralNetworkLearning(self, learning_rate=learning_rate)
        for i in range(max_iter):
            neural_network_learning.run_epoch(input, target, batch_size=batch_size, shuffle=False)
            error = neural_network_learning.calculate_error(input, target)
            if(error <= error_threshold):
                # print(f"i: {i}")
                self.stopped_by = "error_threshold"
                return
        self.stopped_by = "max_iteration"        

class NeuralNetworkLearning:
    def __init__(self, neural_network: NeuralNetwork, *, learning_rate: float = 0.1, update_bias: bool = True) -> None:
        self.__neural_network = neural_network
        self.__learning_rate = learning_rate
        self.__update_bias = update_bias

    def fit_batch(self, input: np.ndarray, output: np.ndarray) -> None:
        """Menjalankan dan update berdasarkan data"""

        result_array = []

        # Forward Propagation
        output_layer = input
        for layer in self.__neural_network.layers:
            result_array.append(output_layer)

            if not layer.is_output:
                output_layer = layer.calculate(output_layer)

        # Backward propagation
        error_diff: np.ndarray = None
        for layer in reversed(self.__neural_network.layers):
            layer_input = result_array.pop()

            # Calculate dE/dNet
            if layer.is_output:
                delta = layer.layer_delta(layer_input, expected=output)
            else:
                delta = layer.layer_delta(layer_input, error_diff=error_diff)

            # Calculate dE/dw
            weight_diff, bias_diff = layer.weight_diff(layer_input, delta)

            error_diff = layer.error_diff(delta)

            # Set w
            new_w = layer.get_w() - self.__learning_rate * weight_diff
            layer.set_w(new_w)

            if self.__update_bias:
                new_b = layer.get_b() - self.__learning_rate * bias_diff
                layer.set_b(new_b)

    def run_epoch(self, input: np.ndarray, output: np.ndarray, *, batch_size=1, shuffle=True) -> None:
        """Menjalankan hanya 1 epoch"""
        epoch_data = input
        epoch_target = output
        data_size = input.shape[0]

        if shuffle:
            idx_list = np.array(range(data_size))
            np.random.shuffle(idx_list)

            epoch_data = input[idx_list]
            epoch_target = output[idx_list]

        batch_iteration = (data_size + batch_size - 1) // batch_size
        for i in range(batch_iteration):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            self.fit_batch(epoch_data[start_idx:end_idx],
                           epoch_target[start_idx:end_idx])

    def calculate_error(self, input: np.ndarray, output: np.ndarray) -> float:
        actual = self.__neural_network.calculate(input)
        last_layer = self.__neural_network.layers[-1]

        loss = last_layer.activation_function.loss_function(output, actual)

        return np.average(loss)
