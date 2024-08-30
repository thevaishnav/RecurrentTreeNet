from source.optimizers.base_optimizer import Optimizer
import numpy as np


class OptimMomentum(Optimizer):
    """Optimizer Stochastic Gradient Descent with Momentum"""

    def init_param(self) -> None:
        self._Vd = np.zeros(self._master.delta_shape)
        self._beta = 0.9

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if type(value) is float and -1 < value < 1:
            self._beta = value
        raise ValueError("Hyper-parameter alpha should be between -1 to 1, both endpoints excluded.")

    def get_delta(self, delta: np.array) -> np.array:
        self._Vd = self._beta * self._Vd + (1 - self._beta) * delta
        return self._Vd

    def new_instance(self, master): return OptimMomentum(master)
