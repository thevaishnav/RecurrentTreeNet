from source import Layer, HiddenLayer
from source.exceptions import LoopError


# class Layer:
#     def __init__(self, name):
#         self._input_socket = set()  # Set of layers on which this layer depends
#         self.name = name
#
#     def add(self, other):
#         self._input_socket.add(other)

def loop_detection(layers: set[Layer]) -> Layer:
    """
    Uses Tortoise & Hair algorithm to detect infinite loops in the dependencies of Layer.
    :param layers: List of layers
    :return: Layer, entry point of the loop if loop exists, None otherwise
    """

    def get_next(l: Layer) -> Layer:
        """
        :return: Next layer in the dependency chain if it exists, otherwise None.
        """
        for edge in l._input_socket:
            if edge.delay_iterations >= 1: continue
            return edge.get_other_layer(l)
        return None

    for layer in layers:
        tortoise = layer
        hare = layer

        while hare is not None:
            # Move tortoise one step
            tortoise = get_next(tortoise)

            # Move hare two steps
            hare = get_next(hare)
            if hare is not None:
                hare = get_next(hare)

            # Check if they meet
            if tortoise == hare:
                if tortoise is None:
                    break

                # If a loop is detected, find the entry point
                entry = layer
                while entry != tortoise:
                    entry = get_next(entry)
                    tortoise = get_next(tortoise)
                return entry
    return None  # No loop detected


def get_execution_order(layers: set[Layer], is_backward: bool) -> list[Layer]:
    """
    Computes the order in the provided list of layers should be traversed,
    considering that some layers will be dependent on other another.
    :param layers: list of layers to be traversed
    :param is_backward: if True, returns order for backward pass, else returns order for forward pass.
    :return: (order for forward pass, order for backward pass)
    """
    def is_satisfied(lyr: Layer):
        sockets = lyr._output_socket if is_backward else lyr._input_socket
        for edge in sockets:
            if edge.delay_iterations <= 0:
                other = edge.get_other_layer(lyr)
                if type(other) is HiddenLayer and other not in ooo:
                    return False
        return True

    ooo = []
    layers_list = list(layers)
    while len(layers_list) > 0:
        for layer in filter(is_satisfied, layers_list):
            ooo.append(layer)
            layers_list.remove(layer)
    return ooo

