from source.activations.base_activation import Activation
from source.activations.sigmoid import ActSigmoid
from source.exceptions import IsolatedLayerError
from source.layers.base_layer import Layer
from source.optimizers.base_optimizer import Optimizer
import numpy as np


class HiddenLayer(Layer):
    def __init__(self,
                 network,
                 neurons: int,
                 _act_fun: Activation = ActSigmoid(),
                 optimizer: Optimizer = None,
                 title: str = "HL"):
        super(HiddenLayer, self).__init__(network, neurons, _act_fun, optimizer, title)
        if self.network.optimizer: self.network.optimizer(self)
        self.init_parameters()

    def init_parameters(self) -> None:
        self._delta = np.zeros(self.delta_shape)
        self._biases = np.random.randn(1, self._nerve_count)
        self.MB = np.zeros((1, self._nerve_count))
        self.SB = np.zeros((1, self._nerve_count))

    def test(self) -> None:
        if len(self._input_socket) == 0: raise IsolatedLayerError(
            f"Layer \"{self.title}\" has no connections at Input.")
        if len(self._output_socket) == 0: raise IsolatedLayerError(
            f"Layer \"{self.title}\" has no connections at Output.")
