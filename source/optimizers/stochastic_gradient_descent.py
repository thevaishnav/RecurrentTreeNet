from source.optimizers.base_optimizer import Optimizer
import numpy as np


class OptimSGD(Optimizer):
    """Optimizer Stochastic Gradient Descent"""

    def init_param(self) -> None: pass

    def get_delta(self, delta: np.array) -> np.array: return delta

    def new_instance(self, master): return OptimSGD(master)
