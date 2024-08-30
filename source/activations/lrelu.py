import numpy as np
from source.activations.base_activation import Activation


class ActLReLU(Activation):
    """Leaky ReLU Activation Function"""

    def __init__(self, alpha=0):
        """
        :param alpha: factor to scale the numbers <0
        """
        self.alpha = alpha

    def function(self, Z: np.array) -> np.array:
        return np.where(Z > 0, Z, Z * self.alpha)

    def prime(self, Z: np.array) -> np.array:
        return np.where(Z > 0, 1, self.alpha)
