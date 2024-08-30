from source.exceptions import IsolatedLayerError
from source.layers.base_layer import Layer
from source.optimizers.base_optimizer import Optimizer
from source.activations.base_activation import Activation
from source.activations.sigmoid import ActSigmoid
import numpy as np


class OutputLayer(Layer):
    def __init__(self,
                 network,
                 neurons: int,
                 _act_fun: Activation = ActSigmoid(),
                 optimizer: Optimizer = None,
                 title="OL"):
        super(OutputLayer, self).__init__(network, neurons, _act_fun, optimizer, title)
        if self.network.optimizer: self.network.optimizer(self)
        self.__delattr__("_output_socket")
        self.init_parameters()

    def init_parameters(self) -> None:
        self._biases = np.random.randn(1, self._nerve_count)
        self.MB = np.zeros((1, self._nerve_count))
        self.SB = np.zeros((1, self._nerve_count))

    def feed_forward(self) -> None:
        super(OutputLayer, self).feed_forward()

    def calc_delta(self, lr: float, Y: np.array) -> None:
        self._delta = self._nuerv_acts - Y

    def update_weights(self, lr: float) -> None:
        super(OutputLayer, self).update_weights(lr)

    def test(self) -> None:
        if len(self._input_socket) == 0: raise IsolatedLayerError(
            f"Layer \"{self.title}\" has no connections at Input.")