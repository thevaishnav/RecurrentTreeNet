import numpy as np
from source.activations.base_activation import Activation


class ActSoftmax(Activation):
    """Softmax Activation Function"""

    def function(self, Z: np.array) -> np.array:
        s = np.max(Z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(Z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div

    def prime(self, Z: np.array) -> np.array:
        return self.function(Z)
