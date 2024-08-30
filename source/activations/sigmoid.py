import numpy as np
from source.activations.base_activation import Activation


class ActSigmoid(Activation):
    """Sigmoid Activation Function"""

    def function(self, Z: np.array) -> np.array:
        Z = np.clip(Z, -100, 100)
        return 1.0 / (1.0 + np.exp(-Z))

    def prime(self, Z: np.array) -> np.array:
        sig = self.function(Z)
        return sig * (1 - sig)
