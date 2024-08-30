import numpy as np


class InputScaler:
    def __init__(self):
        self._sig = None
        self._mu = None
        self._X = None
        self._Y = None
        self._validX = None
        self._validY = None

    @property
    def is_active(self):
        return self._sig is not None

    def set_params(self, X: np.array):
        self._mu = np.average(X, axis=0)
        X_new = X - self._mu
        self._sig = np.average(np.square(X), axis=0) + 0.01  # 0.01 is added so that division don`t return inf.
        X_new /= self._sig
        return X_new

    def scale(self, data: np.array):
        return (data - self._mu) / self._sig