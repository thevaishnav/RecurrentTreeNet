from .edge import Edge
from .loss import Loss
import numpy as np

from source.exceptions import LoopError
from source.layers.base_layer import Layer
from source.layers.hidden_layer import HiddenLayer
from source.layers.input_layer import InputLayer
from source.layers.output_layer import OutputLayer
from source.optimizers.base_optimizer import Optimizer
from source.optimizers.stochastic_gradient_descent import OptimSGD


class InputScaler:
    def __init__(self):
        self._sig = None
        self._mu = None
        self._X = None
        self._Y = None
        self._validX = None
        self._validY = None

    @property
    def is_active(self):
        return self._sig is not None

    def set_params(self, X: np.array):
        self._mu = np.average(X, axis=0)
        X_new = X - self._mu
        self._sig = np.average(np.square(X), axis=0) + 0.01  # 0.01 is added so that division don`t return inf.
        X_new /= self._sig
        return X_new

    def scale(self, data: np.array):
        return (data - self._mu) / self._sig


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

    @property
    def current_batch_no(self):
        return self._current_batch_no

    def optimizer(self, master):
        """Internal Method: Creates and returns new instance of Default optimizer class"""
        return self._optimizer.new_instance(master)

    @property
    def layer_count(self) -> int:
        return len(self._hidden_layers) + 1

    def add_layer(self, layer: Layer) -> None:
        """Internal method: Used by any instance of Layer class to declare its presence for network"""
        if type(layer) is InputLayer:
            self._input_layer = layer
        elif type(layer) is OutputLayer:
            self._output_layer = layer
        else:
            self._hidden_layers.add(layer)

    def _get_order_of_operations(self, reversed=True):
        """
        Internal function:
        Implementation of Depth First Search Algorithm with layers as node.
        Returns order in which layers should be called.
        If Reversed = True, then returns order for back-propagation.
        Else returns order for forward propagation.
        """
        ooo = []

        reqs = {}
        for lyr in self._hidden_layers:
            lyr.test()
            rs = set()
            sockets = lyr._output_socket if reversed else lyr._input_socket
            for edge in sockets:
                if edge.delay_iterations >= 1: continue
                ol = edge.get_other_layer(lyr)
                if type(ol) is HiddenLayer: rs.add(ol)
            reqs[lyr.id] = rs

        layers = list(self._hidden_layers)
        count = 0
        while layers:
            for lyr in layers:
                for req_lyr in reqs[lyr.id]:
                    if req_lyr not in ooo: break
                else:  # Else to a for loop is called when break is called inside for loop
                    ooo.append(lyr)
                    layers.remove(lyr)
            if count == len(self._hidden_layers):
                raise LoopError("Given network has infinite data loop.")
            count += 1
        return ooo

    def _test_validity(self, first_input, first_output):
        """Internal Function. Check if the network is valid or not"""
        ip, op = first_input.shape[0], first_output.shape[0]
        if self._input_layer is None: raise ValueError("Input Layer is not specified")
        if self._output_layer is None: raise ValueError("Output Layer is not specified")
        if self._input_layer.nerve_count != ip: raise ValueError(
            f"The given data contains invalid number of input parameters ({ip}), Expected: {self._input_layer.nerve_count}")
        if self._output_layer.nerve_count != op: raise ValueError(
            f"The given data contains invalid number of output parameters ({op}), Expected: {self._output_layer.nerve_count}")

    def _fit_batch(self, X: np.array, Y: np.array, lr, forw_ooo: list, back_ooo: list):
        """Internal function: Backprop for single batch"""
        # Forward pass
        self._input_layer._nuerv_acts = X
        for layer in forw_ooo:
            layer.feed_forward()
        self._output_layer.feed_forward()

        # Backward Pass
        self._output_layer.calc_delta(lr=lr, Y=Y)
        for layer in back_ooo:
            layer.calc_delta(lr)

        self._output_layer.update_weights(lr=lr)
        for layer in back_ooo:
            layer.update_weights(lr)

    def predict(self, inputs: np.array) -> np.array:
        """
        Predicts the output for given input
        :return: prediction.
        """
        if self._input_scalar.is_active: inputs = self._input_scalar.scale(inputs)
        self._input_layer._nuerv_acts = inputs
        for layer in self._get_order_of_operations(False):
            layer.feed_forward()
        self._output_layer.feed_forward()
        return self._output_layer._nuerv_acts

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
        Train network on given training data trinX :param trainX: and :param trainY:
        :param mini_batch_size: Training will be done in small batches
        :param epoch: Number of epochs
        :param lr: Learning rate
        :param shuffle: Whether to shuffle data after every epoch
        :param epoch_complete_call: What function to call after completion of every epoch.
        :param batch_complete_call: What functions to call after completion of every mini-batch
        :param validation_ratio: fraction of training data which will be used for validation. (between 0 to 1, endpoints excluded)
        :return: None
        """
        trainX = self._input_scalar.set_params(trainX)
        training_data = list(zip(trainX, trainY))
        self._batch_size = mini_batch_size
        self._test_validity(trainX[0], trainY[0])
        forw_ooo = self._get_order_of_operations(False)
        back_ooo = self._get_order_of_operations(True)
        data_size = len(training_data)

        len_validation = int(len(training_data) * validation_ratio)
        validation_data = training_data[:len_validation]
        training_data = training_data[len_validation:]
        for epoch in range(epoch):
            if shuffle: np.random.shuffle(training_data)
            trainX, trainY = zip(*training_data)
            mini_batches = [[trainX[k - mini_batch_size:k], trainY[k - mini_batch_size:k]]
                            for k in range(mini_batch_size, data_size, mini_batch_size)]
            self._current_batch_no = 0
            for miniX, miniY in mini_batches:
                if (not miniX) or (not miniY): continue
                miniX, miniY = np.array(miniX), np.array(miniY)
                self._fit_batch(miniX, miniY, lr, forw_ooo, back_ooo)
                if batch_complete_call:
                    batch_complete_call(self._current_batch_no, self.get_validation_error(validation_data, forw_ooo))
                self._current_batch_no += 1
            if epoch_complete_call is not None:
                epoch_complete_call(epoch, self.get_validation_error(validation_data, forw_ooo))
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

    def get_validation_error(self, validation_data, forw_ooo):
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
        for layer in forw_ooo:
            layer.feed_forward()
        self._output_layer.feed_forward()
        delta = self._output_layer._nuerv_acts - validY

        self._current_batch_no = cbi
        return Loss.RMS_delta(delta)
