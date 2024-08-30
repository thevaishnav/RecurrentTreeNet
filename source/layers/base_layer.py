from source.activations.base_activation import Activation
from source.optimizers.base_optimizer import Optimizer
import numpy as np


class Layer:
    """
    Parent class to Layer Class.
    """

    def __init__(self,
                 network,
                 neurons: int,
                 _act_fun: Activation = Activation(),
                 optimizer: Optimizer = None,
                 title: str = None):
        if optimizer and (not issubclass(type(optimizer), Optimizer)):
            raise ValueError(f"Invalid type for parameter optimizer, expected Optimizer, got {type(optimizer)}")

        self._id = id(self)
        self._network = network
        self._nerve_count = neurons
        self._input_socket = set()
        self._output_socket = set()

        self.title = f"layer_{network.layer_count}" if not network else title
        self.__act_fun = _act_fun
        self._optimizer = optimizer or self.network.optimizer(self)
        self._optimizer.set_parent(self)
        self._optimizer.init_param()

        self._biases = None
        self._delta = None
        self._nuerv_acts = None
        network.add_layer(self)

    def __str__(self):
        return self.title

    def __repr__(self):
        return self.__str__()

    @property
    def id(self):
        return self._id

    @property
    def delta_shape(self):
        return 1, self._nerve_count

    @property
    def network(self):
        return self._network

    @property
    def nerve_count(self):
        return self._nerve_count

    @property
    def delta(self):
        return self._delta

    def add_input(self, layer):
        self._input_socket.add(layer)

    def add_output(self, layer):
        self._output_socket.add(layer)

    def feed_forward(self) -> None:
        self.Z = 0
        for edge in self._input_socket:
            prev_layer = edge.get_other_layer(self)
            self.Z = self.Z + np.dot(prev_layer._nuerv_acts, edge.weights) + self._biases
        self._nuerv_acts = self.__act_fun.function(self.Z)

    def calc_delta(self, lr: float) -> None:
        self._delta = 0
        for edge in self._output_socket:
            layer_after = edge.get_other_layer(self)
            self._delta += \
                np.dot(layer_after._delta, edge.weights.T) * layer_after.__act_fun.prime(self.Z)

    def update_weights(self, lr: float):
        deltaB = np.mean(self._delta, axis=0)
        self._biases -= lr * self._optimizer.get_delta(deltaB)
        for edge in self._input_socket:
            edge.update_weights(lr)

    def test(self) -> None:
        raise NotImplementedError()
