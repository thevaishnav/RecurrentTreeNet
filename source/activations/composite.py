import numpy as np
from source.activations.base_activation import Activation


class ActCompose(Activation):
    """
    Compose multiple activation functions as once.
    The order in which activation functions are passed will be considered as reversed order of operation.
    In order to compose, Act1, Act2, and Act3 as
        output = Act1(Act2(Act3(input)))
    create instance of this class as
        instance = ActCompose(Act1, Act2, Act3)"""

    def __init__(self, *args: Activation):
        self.acts = list(reversed(args))

    def function(self, Z: np.array) -> np.array:
        temp_Z = np.zeros(Z.shape)
        for af in self.acts:
            temp_Z = af.function(temp_Z)
        return temp_Z

    def prime(self, Z: np.array) -> np.array:
        derivatives = []
        temp_Z = Z
        for af in self.acts[:-1]:
            temp_Z = af.function(temp_Z)
            derivatives.append(temp_Z)

        output = self.acts[0].prime(Z)
        for f, h in zip(self.acts[1:], derivatives):
            output = np.multiply(output, f.prime(h))
        return output
