from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def calculate(input: np.ndarray) -> np.ndarray:
        pass