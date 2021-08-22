import numpy as np


class Activation:
    """
    Activation Class. No need to create instance of this class.
    For Example, use Activation.Sigmoid to refer to Sigmoid Activation Function.
    """

    def __call__(self, Z, prime=True):
        if prime: return self.prime(Z)
        return self.function(Z)

    def function(self, Z: np.array) -> np.array:
        """Calculate activation non-linearity during forward pass"""
        raise NotImplementedError("This Function is not implemented")

    def prime(self, Z: np.array) -> np.array:
        """calculate derivative of activation non-linearity, called during backward pass"""
        raise NotImplementedError("This Function is not implemented")


class ActSigmoid(Activation):
    """Sigmoid Activation Function"""

    def function(self, Z: np.array) -> np.array:
        Z = np.clip(Z, -100, 100)
        return 1.0 / (1.0 + np.exp(-Z))

    def prime(self, Z: np.array) -> np.array:
        sig = self.function(Z)
        return sig * (1 - sig)


class ActReLU(Activation):
    """ReLU Activation Function"""

    def function(self, Z: np.array) -> np.array:
        return np.maximum(Z, 0)

    def prime(self, Z: np.array) -> np.array:
        return 1 * (Z > 0)


class ActTanh(Activation):
    """Hyperbolic Tan Activation Function"""

    def function(self, Z: np.array) -> np.array:
        return np.tanh(Z)

    def prime(self, Z: np.array) -> np.array:
        return 1 - ((np.tanh(Z)) ** 2)


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


class ActSoftmax(Activation):
    """Softmax Activation Function"""

    def function(self, Z: np.array) -> np.array:
        ex = np.exp(Z)
        return ex / np.sum(ex, axis=0)

    def prime(self, Z: np.array) -> np.array:
        ex = np.exp(Z)
        return ex / np.sum(ex, axis=0)


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
