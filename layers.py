from activations import *
from optimizers import *
from errors import *


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


class InputLayer(Layer):
    def __init__s(self, network, neurons: int, title: str = None):
        super(InputLayer, self).__init__(network, neurons, title=title)
        self.__delattr__("_input_socket")
        self.__delattr__("_biases")
        self.__delattr__("_delta")
        self.__delattr__("__act_fun")
        self.__delattr__("_optimizer")

    def feed_forward(self) -> None: pass

    def calc_delta(self, lr: float) -> None: pass

    def test(self) -> None:
        if len(self._output_socket) == 0:
            raise IsolatedLayerError(f"Layer \"{self.title}\" has no connections at Output.")


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
