import numpy as np
from source.activations.base_activation import Activation


class ActSoftmax(Activation):
    """Softmax Activation Function"""

    def function(self, Z: np.array) -> np.array:
        ex = np.exp(Z)
        return ex / np.sum(ex, axis=0)

    def prime(self, Z: np.array) -> np.array:
        ex = np.exp(Z)
        return ex / np.sum(ex, axis=0)
