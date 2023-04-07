from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def calculate(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def delta(self, wtx: np.ndarray, target_differential: np.ndarray) -> np.ndarray:
        pass

    def loss_function(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 0.5 * np.square(actual - expected).sum(axis=1)

    def output_delta(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return (-1 * (actual - expected)).sum(axis=1)


class ActivationFunDiffentiable(ActivationFunction):
    @abstractmethod
    def differential(self, input: np.ndarray) -> np.ndarray:
        pass

    def delta(self, wtx: np.ndarray, target_differential: np.ndarray) -> np.ndarray:
        # TODO: Menghitung delta dari wtx, target_differential adalah dE/dNext
        diff_data = self.differential(wtx)
        return target_differential * diff_data


class LinearActivation(ActivationFunDiffentiable):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        return input

    def differential(self, input: np.ndarray) -> np.ndarray:
        return np.ones(input.shape)


class ReluActivation(ActivationFunDiffentiable):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(np.zeros(input.shape), input)

    def differential(self, input: np.ndarray) -> np.ndarray:
        return (input >= 0).astype('float64')


class SigmoidActivation(ActivationFunDiffentiable):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        return 1 / (np.exp(-1 * input) + 1)

    def differential(self, input: np.ndarray) -> np.ndarray:
        calculated_input = self.calculate(input)
        return np.multiply(calculated_input, 1 - calculated_input)


class SoftmaxActivation(ActivationFunction):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        exp_input = np.exp(input)
        sum_layer = np.sum(exp_input, axis=1)

        return exp_input/sum_layer

    def delta(self, wtx: np.ndarray, target_class: np.ndarray) -> np.ndarray:
        """Calculate delta of softmax. Assumption only in output layer"""
        result = self.calculate(wtx)

        for i in range(len(wtx)):
            result[i][target_class[i]] -= 1

        return result

    def loss_function(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        result = np.zeros(len(actual))
        argmax = np.argmax(actual)

        for i in range(len(actual)):
            result[i] = -np.log(actual[argmax[i]])

        return -1 * np.log(expected)

    def output_delta(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.argmax(expected, axis=1)
