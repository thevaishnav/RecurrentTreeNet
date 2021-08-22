import numpy as np


class Optimizer:
    """
    Parent class to all other Optimizer Classes.
    Every layer or edge can have their own optimizer.
        Which optimizer to use can be passed in while
        instantiating Layer or Edge
    Or Entire network can have only one Optimizer.
        Which optimizer to use can be passed in while
        instantiating Network class.
    Can`t change optimizer after instance is created.
    In order to create a New Type of Optimizer,
    Create a Child Class to this class and implement
    following two functions-
        init_param: Initialize the hyper-parameters
        get_delta: return the amount by which Weights or
            Biases should be updated in this iteration

    """

    def __init__(self, master=None):
        self._master = master

    def new_instance(self, master):
        raise NotImplementedError()

    @property
    def master(self):
        return self._master

    def init_param(self) -> None:
        """
        Initialize the hyper-parameters
        :return: None
        """
        raise NotImplementedError()

    def get_delta(self, delta: np.array) -> np.array:
        """
        :param delta: deltaW or deltaB depending on whether the parent is Edge or Layer.
        :return: corrected deltaW or deltaB.
            return amount will be subtracted from weights (or biases),
            after multiplying by learning rate.
        """
        raise NotImplementedError()

    def set_parent(self, parent):
        if (self._master) and (parent != self._master): raise BrokenPipeError(
            "Cannot change parent of Optimizer instance")
        self._master = parent


class OptimSGD(Optimizer):
    """Optimizer Stochastic Gradient Descent"""

    def init_param(self) -> None: pass

    def get_delta(self, delta: np.array) -> np.array: return delta

    def new_instance(self, master): return OptimSGD(master)


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


class OptimAdam(Optimizer):
    """Optimizer Adam"""

    def init_param(self) -> None:
        self._Vd = np.zeros(self._master.delta_shape)
        self._Sd = np.zeros(self._master.delta_shape)
        self._beta = 0.9
        self._epsilon = 0.00000001

    def new_instance(self, master):
        return OptimAdam(master)

    def get_delta(self, delta: np.array) -> np.array:
        self._Vd = self._beta * self._Vd + (1 - self._beta) * delta
        self._Sd = self._beta * self._Sd + (1 - self._beta) * np.square(delta)
        return np.divide(self._Vd, np.sqrt(self._Sd) + self._epsilon)

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

