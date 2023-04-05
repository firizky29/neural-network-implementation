from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def calculate(self, input: np.ndarray) -> np.ndarray:
        pass


class LinearActivation(ActivationFunction):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        return input


class ReluActivation(ActivationFunction):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(np.zeros(input.shape), input)


class SigmoidActivation(ActivationFunction):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        return 1 / (np.exp(-1 * input) + 1)


class SoftmaxActivation(ActivationFunction):
    def calculate(self, input: np.ndarray) -> np.ndarray:
        exp_input = np.exp(input)
        sum_layer = np.sum(exp_input, axis=1)

        return exp_input/sum_layer
