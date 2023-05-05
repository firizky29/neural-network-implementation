import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from Layer import Layer
from NeuralNetworkLearning import NeuralNetworkLearning


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

    def set_input_target(self, input: np.ndarray, target: np.ndarray):
        self.input = input
        self.target = target
        return self

    def set_learning_properties(self, learning_rate: float = 0.1, batch_size=1, max_iter=1000, error_threshold:float =0.01):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.error_threshold = error_threshold
        return self

    def fit(self) -> None:
        neural_network_learning = NeuralNetworkLearning(self, learning_rate=self.learning_rate)
        for i in range(self.max_iter):
            neural_network_learning.run_epoch(self.input, self.target, batch_size=self.batch_size, shuffle=False)
            error = neural_network_learning.calculate_error(self.input, self.target)
            if(error <= self.error_threshold):
                # print(f"i: {i}")
                self.stopped_by = "error_threshold"
                return
        self.stopped_by = "max_iteration"        
