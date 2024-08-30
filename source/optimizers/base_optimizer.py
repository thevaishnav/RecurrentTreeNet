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
