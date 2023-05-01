import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork


class NeuralNetworkLearning:
    def __init__(self, neural_network: NeuralNetwork, *, learning_rate: float = 0.1) -> None:
        self.__neural_network = neural_network
        self.__learning_rate = learning_rate

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
        last_delta: np.ndarray = None
        last_layer: Layer = None
        for layer in reversed(self.__neural_network.layers):
            layer_input = result_array.pop()

            # Calculate dE/dNet
            if layer.is_output:
                delta = layer.layer_delta(layer_input, expected=output)
            else:
                error_diff = last_layer.error_diff(last_delta)
                delta = layer.layer_delta(layer_input, error_diff=error_diff)

            # Calculate dE/dw
            weight_diff, bias_diff = layer.weight_diff(layer_input, delta)

            new_w = layer.get_w() - self.__learning_rate * weight_diff
            new_b = layer.get_b() - self.__learning_rate * bias_diff

            layer.set_w(new_w)
            layer.set_b(new_b)

            last_delta = delta
            last_layer = layer

    def run_epoch(self, input: np.ndarray, output: np.ndarray, *, batch_size=1, shuffle=True) -> None:
        """Menjalankan hanya 1 epoch"""
        epoch_data = input
        epoch_target = output
        data_size = input.shape[0]

        if shuffle:
            idx_list = np.array(range(data_size))
            idx_list = np.random.shuffle(idx_list)

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

        return loss.sum()
