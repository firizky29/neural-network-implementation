import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from Layer import Layer


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
                labels[get_id(i+1,0)] = layer.activation_name
            for j in range(self.layers[i].num_of_neurons):
                colors.append(layer_color)
                # bias
                G.add_edge(get_id(i, 0), get_id(i+1, j+1), weight=layer.b[j])
                # input
                for k in range(self.layers[i].num_of_input):
                    G.add_edge(get_id(i, k+1), get_id(i+1, j+1),
                               weight=layer.w[j][k])
                G.nodes[get_id(i+1, j+1)]["subset"] = i+1
            if(i == len(self.layers) - 1):
                labels[get_id(i+1,1)] = layer.activation_name

        pos = nx.multipartite_layout(G)
        nx.draw(G, pos, with_labels=True,
                font_weight='bold', node_color=colors, node_size=100, font_size=10)
        nx.draw_networkx_labels(G, pos, labels=labels, verticalalignment='bottom', horizontalalignment="left", font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(
            G, 'weight'), label_pos=0.8, font_size=8)
        plt.show()

    def fit(self, input: np.ndarray, target: np.ndarray):
        pass


if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.addLayer(Layer(
        [[0.05, 0.09], [0.10, 0.12], [0.08, 0.15]],
        [1, 2, 3],
        "linear"
    ))
    nn.addLayer(Layer(
        [[0.11, 0.15, 0.69], [0.13, 0.35, 0.21]],
        [1, 2],
        "relu"
    ))
    nn.addLayer(Layer(
        [[0.15, 0.69], [0.35, 0.21]],
        [1, 2],
        "softmax"
    ))
    nn.draw()
