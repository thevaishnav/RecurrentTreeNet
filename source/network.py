from .edge import Edge
from .input_scaler import InputScaler
from .loss import Loss
import numpy as np

from source.layers.base_layer import Layer
from source.layers.input_layer import InputLayer
from source.layers.output_layer import OutputLayer
from source.optimizers.base_optimizer import Optimizer
from source.optimizers.stochastic_gradient_descent import OptimSGD
from source.compilation import loop_detection, get_execution_order
from source.exceptions import LoopError

class Network:
    def __init__(self, optimizer: Optimizer = None):
        """
        :param optimizer: SGD is used as default optimizer. This optimizer will be used by Layer and Edge.
        """
        if optimizer and (not issubclass(type(optimizer), Optimizer)):
            raise ValueError(f"Invalid type for parameter optimizer, expected Optimizer, got {type(optimizer)}")
        self._optimizer = optimizer or OptimSGD()
        self._input_layer = None
        self._output_layer = None
        self._hidden_layers = set()
        self._current_batch_no = None
        self._input_scalar = InputScaler()
        self._is_stopped_during_training = False
        self._eo_forward: list[Layer] = None        # Execution Order for Forward Pass, None if the model is not compiled
        self._eo_backward: list[Layer] = None       # Execution Order for Backward Pass, None if the model is not compiled

    @property
    def current_batch_no(self):
        return self._current_batch_no

    def optimizer(self, master):
        """Internal Method: Creates and returns new instance of Default optimizer class"""
        return self._optimizer.new_instance(master)

    @property
    def layer_count(self) -> int:
        return len(self._hidden_layers) + 1

    def halt_training(self):
        self._is_stopped_during_training = True
        print("halt_training is called, Model will not be trained any further.")

    def add_layer(self, layer: Layer) -> None:
        """Internal method: Used by any instance of Layer class to declare its presence for network"""
        if type(layer) is InputLayer:
            self._input_layer = layer
        elif type(layer) is OutputLayer:
            self._output_layer = layer
        else:
            self._hidden_layers.add(layer)

    def _test_validity(self, first_input, first_output):
        """Internal Function. Check if the network is valid or not"""
        ip, op = first_input.shape[0], first_output.shape[0]
        if self._input_layer is None: raise ValueError("Input Layer is not specified")
        if self._output_layer is None: raise ValueError("Output Layer is not specified")
        if self._input_layer.nerve_count != ip: raise ValueError(
            f"The given data contains invalid number of input parameters ({ip}), Expected: {self._input_layer.nerve_count}")
        if self._output_layer.nerve_count != op: raise ValueError(
            f"The given data contains invalid number of output parameters ({op}), Expected: {self._output_layer.nerve_count}")

    def _fit_batch(self, X: np.array, Y: np.array, lr):
        """Internal function: Backprop for single batch"""
        # Forward pass
        self._input_layer._nuerv_acts = X
        for layer in self._eo_forward:
            layer.feed_forward()
        self._output_layer.feed_forward()

        # Backward Pass
        self._output_layer.calc_delta(lr=lr, Y=Y)
        for layer in self._eo_backward:
            layer.calc_delta(lr)

        self._output_layer.update_weights(lr=lr)
        for layer in self._eo_backward:
            layer.update_weights(lr)

    def compile(self):
        """
        Compiles this models, i.e. decides the execution order for the forward and backward pass.
        The prediction and training will be done based on compiled model,
        therefore the model must be compiled everytime connections between layers, or neuron counts in any layer changes.
        You can however, change activation function and optimizers without having to re-compile.
        """

        used_titles = set()
        # Check for Isolated Layers
        for layer in self._hidden_layers:
            lower = layer.title.lower()
            if lower in used_titles:
                raise ValueError(f"Multiple layers with same title: \"{layer.title}\" (Case Insensitive)"
                                 f"The layer titles will be used to uniquely identify each layer during serialization.")

            used_titles.add(lower)
            layer.test()

        # Check for infinite loops
        loop_entry = loop_detection(self._hidden_layers)
        if loop_entry is not None:
            raise LoopError(f"Found infinite loop in the connections. Entry Point: {loop_entry.title}")

        self._eo_forward = get_execution_order(self._hidden_layers, False)
        self._eo_backward = get_execution_order(self._hidden_layers, True)

    def predict(self, inputs: np.array) -> np.array:
        """
        MUST COMPILE THE NETWORK BEFORE RUNNING PREDICT FUNCTION
        Predicts the output for given input
        :return: prediction.
        """
        if self._eo_forward is None:
            raise BrokenPipeError("Must compile the model (.compile function) before predict.")

        if self._input_scalar.is_active:
            inputs = self._input_scalar.scale(inputs)

        self._input_layer._nuerv_acts = inputs
        for layer in self._eo_forward:
            layer.feed_forward()
        self._output_layer.feed_forward()
        return self._output_layer._nuerv_acts

    def serialize(self) -> dict[str, dict[str, object]]:
        if self._eo_forward is None:
            raise BrokenPipeError("Must compile the model (.compile function) before saving.")
        d = {}
        for layer in self._eo_forward:
            d[layer.title.lower()] = layer.serialize()
        d[self._output_layer.title.lower()] = self._output_layer.serialize()
        return d

    def deserialize(self, data: dict[str, dict[str, object]]):
        def load_layer(lyr: Layer):
            lower = lyr.title.lower()
            if lower in data: lyr.deserialize(data[lower])
            else: print(f"Data for layer \"{lyr.title}\" not found while loading. Check if the layers are titled correctly. Titles are case insensitive")

        if self._eo_forward is None:
            raise BrokenPipeError("Must compile the model (.compile function) before loading.")

        for layer in self._eo_forward:
            load_layer(layer)
        load_layer(self._output_layer)

    def fit(self,
            trainX: np.array,
            trainY: np.array,
            mini_batch_size: int,
            epoch: int = 5,
            lr: float = 0.05,
            shuffle: bool = True,
            epoch_complete_call=None,
            batch_complete_call=None,
            validation_ratio: float = 0.2
            ):
        """
        MUST COMPILE THE NETWORK BEFORE RUNNING FIT FUNCTION
        Train network on given training data trinX :param trainX: and :param trainY:
        :param mini_batch_size: Training will be done in small batches
        :param epoch: Number of epochs
        :param lr: Learning rate
        :param shuffle: Whether to shuffle data after every epoch
        :param epoch_complete_call: What function to call after completion of every epoch.
        :param batch_complete_call: What functions to call after completion of every mini-batch
        :param validation_ratio: fraction of training data which will be used for validation. (between 0 & 1, endpoints excluded)
        :return: None
        """
        if self._eo_forward is None or self._eo_backward is None:
            raise BrokenPipeError("Must compile the model (.compile function) before predict.")

        trainX = self._input_scalar.set_params(trainX)
        training_data = list(zip(trainX, trainY))
        self._batch_size = mini_batch_size
        self._test_validity(trainX[0], trainY[0])
        data_size = len(training_data)

        len_validation = int(len(training_data) * validation_ratio)
        validation_data = training_data[:len_validation]
        training_data = training_data[len_validation:]
        for epoch in range(epoch):
            if self._is_stopped_during_training:
                break
            if shuffle:
                np.random.shuffle(training_data)

            trainX, trainY = zip(*training_data)
            mini_batches = [[trainX[k - mini_batch_size:k], trainY[k - mini_batch_size:k]]
                            for k in range(mini_batch_size, data_size, mini_batch_size)]
            self._current_batch_no = 0
            for miniX, miniY in mini_batches:
                if (not miniX) or (not miniY): continue
                miniX, miniY = np.array(miniX), np.array(miniY)
                self._fit_batch(miniX, miniY, lr)
                if batch_complete_call:
                    batch_complete_call(self._current_batch_no, self.get_validation_error(validation_data))
                self._current_batch_no += 1
            if epoch_complete_call is not None:
                epoch_complete_call(epoch, self.get_validation_error(validation_data))
        self._current_batch_no = -1
        self._batch_size = -1

    def connect(self,
                layer1: Layer,
                layer2: Layer,
                optimizer: Optimizer = None,
                delay_iterations: int = 0):
        """
        Connect :param layer1: to :param layer2:
        :param optimizer: Optimizer to be used for this edge.
        :param delay_iterations: parameter delay_iterations for edge
        :return:
        """
        return Edge(layer1, layer2, optimizer, delay_iterations)

    def linear_connect(self, *layers):
        """
        Connect every layer in :param layers: to next layer.
        All arguments for Edge are defaulted.
        To change default arguments use self.connect, to connect two layers.
        :return: None
        """
        if len(layers) < 2: return
        for l1, l2 in zip(layers[:-1], layers[1:]):
            Edge(l1, l2)

    def get_validation_error(self, validation_data):
        """
        Internal Function.
        :return: Root Mean Squared loss
        """
        validX, validY = zip(*validation_data)
        validX, validY = np.array(validX), np.array(validY)
        cbi = int(self._current_batch_no)
        self._current_batch_no = -2

        # Feed Forward
        self._input_layer._nuerv_acts = validX
        for layer in self._eo_forward:
            layer.feed_forward()
        self._output_layer.feed_forward()
        delta = self._output_layer._nuerv_acts - validY

        self._current_batch_no = cbi
        return Loss.RMS_delta(delta)
