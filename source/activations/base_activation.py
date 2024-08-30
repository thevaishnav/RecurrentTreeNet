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
