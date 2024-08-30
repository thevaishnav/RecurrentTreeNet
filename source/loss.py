import numpy as np


class Loss:
    @staticmethod
    def RMS(output: np.array, expected: np.array) -> np.array:
        """Root Mean Squared Loss"""
        delta = output - expected
        return np.mean((np.mean(delta ** 2, axis=1)) ** 0.5)

    @staticmethod
    def cross_entropy(output: np.array, expected: np.array) -> np.array:
        """Cross Entropy Loss"""
        return -(output * np.log(expected) + (1 - output) * np.log(1 - expected))

    @staticmethod
    def RMS_delta(delta):
        return np.mean((np.mean(delta ** 2, axis=1)) ** 0.5)
