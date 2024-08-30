import numpy as np
from source.activations.base_activation import Activation


class ActReLU(Activation):
    """ReLU Activation Function"""

    def function(self, Z: np.array) -> np.array:
        return np.maximum(Z, 0)

    def prime(self, Z: np.array) -> np.array:
        return 1 * (Z > 0)
