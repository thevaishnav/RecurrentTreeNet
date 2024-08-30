from source.optimizers.base_optimizer import Optimizer
import numpy as np


class OptimRMPProp(Optimizer):
    """Optimizer Root Mean Squared Propagation (RMS Prop)"""

    def init_param(self) -> None:
        self._Sd = np.zeros(self._master.delta_shape)
        self._beta = 0.9
        self._epsilon = 0.00000001

    def get_delta(self, delta: np.array) -> np.array:
        self._Sd = self._beta * self._Sd + (1 - self._beta) * np.square(delta)
        return np.divide(delta, np.sqrt(self._Sd) + self._epsilon)

    def new_instance(self, master):
        return OptimRMPProp(master)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if type(value) is float and -1 < value < 1:
            self._beta = value
        raise ValueError("Hyper-parameter beta should be between -1 to 1, both endpoints excluded.")

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if type(value) is not float: raise ValueError("Hyper-parameter epsilon must be of type float.")
        self._epsilon = value