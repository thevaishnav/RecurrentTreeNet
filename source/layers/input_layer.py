from source.exceptions import IsolatedLayerError
from source.layers.base_layer import Layer


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
