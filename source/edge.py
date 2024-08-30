from .exceptions import *
from source.layers.base_layer import Layer
from source.optimizers.base_optimizer import Optimizer
import numpy as np


class Edge:
    """
    Edge connect two Layers together.
    """

    def __init__(self,
                 from_layer: Layer,
                 to_layer: Layer,
                 optimizer: Optimizer = None,
                 delay_iterations: int = 0):
        """
        Output of "from_layer" in nth iteration (batch) will be passed
        to input of "to_layer" at n + "delay_iterations" iteration (batch)
        """
        if from_layer.network == to_layer.network:
            self.network = from_layer.network
        else:
            raise LayerParentError(
                f"The given layers {from_layer.title} and {to_layer.title} are from different network.")
        if (type(delay_iterations) is not int) or (delay_iterations < 0):
            raise ValueError("Delay Iterations (delay_iterations) must be an positive integer or zero.")
        if optimizer and (not issubclass(type(optimizer), Optimizer)):
            raise ValueError(f"Invalid type for parameter optimizer, expected Optimizer, got {type(optimizer)}")

        self._start = from_layer
        self._end = to_layer
        self._delay_iterations = delay_iterations
        self._start.add_output(self)
        self._end.add_input(self)
        self._savedW = []
        self._optimizer = optimizer or self.network.optimizer(self)
        self._optimizer.set_parent(self)
        self._optimizer.init_param()
        self.init_parameters()

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def delay_iterations(self):
        return self._delay_iterations

    def init_parameters(self) -> None:
        """Initialize the hyper-parameters. Called when the edge is instantiated."""
        self._weights = np.random.randn(self._start.nerve_count, self._end.nerve_count)

    @property
    def delta_shape(self) -> tuple:
        """
        :return: tuple (shape) of weight matrix.
        """
        return self._start.delta_shape[1], self._end.delta_shape[1]

    @property
    def weights(self) -> np.array:
        """
        :return: Current Weights if delay_iterations == 1 else weights
                "delay_iterations" iterations before this one.
                If current iteration < delay_iterations, then random weights.
        """
        itt = self.network.current_batch_no
        if (self._delay_iterations == 0) or (itt == "test"):
            return self._weights
        elif len(self._savedW) < itt:
            return np.random.randn(self._start.nerve_count, self._end.nerve_count)
        return self._savedW[0]

    def get_other_layer(self, layer: Layer) -> Layer:
        return self._start if layer == self._end else self._end

    def update_weights(self, lr: float) -> None:
        """
        :param lr: learning rate
        update weights and save them.
        """
        deltaW = np.dot(self._start._nuerv_acts.T, self._end._delta)
        self._weights -= lr * self._optimizer.get_delta(deltaW)
        if self._delay_iterations and len(self._savedW) == self._delay_iterations:
            self._savedW.pop(0)
        self._savedW.append(self._weights)
