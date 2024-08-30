import numpy as np
from source.activations.base_activation import Activation


class ActTanh(Activation):
    """Hyperbolic Tan Activation Function"""

    def function(self, Z: np.array) -> np.array:
        return np.tanh(Z)

    def prime(self, Z: np.array) -> np.array:
        return 1 - ((np.tanh(Z)) ** 2)
