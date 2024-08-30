class LoopError(Exception):
    """
    Primary cause of this error is if the network has an infinite loop.
    For Example, consider 3 instances of HiddenLayer class, named as
    layer_1, layer_2, layer_3.
    Then following set of Edges will cause a loop error.
        Edge(from_layer=layer_1, to_layer=layer_2, delay_iterations=0)
        Edge(from_layer=layer_2, to_layer=layer_3, delay_iterations=0)
        Edge(from_layer=layer_3, to_layer=layer_1, delay_iterations=0)
    because the network is being asked to
        take "output of layer_1 in current iteration" as "input to layer_2 in current iteration"
        take "output of layer_2 in current iteration" as "input to layer_3 in current iteration"
        take "output of layer_3 in current iteration" as "input to layer_1 in current iteration"
    which causes an infinite loop.
    Note that "current iteration" in code means "delay_iterations=0".

    In contrast, following set of Edges will NOT cause a loop error
        Edge(from_layer=layer_1, to_layer=layer_2, delay_iterations=1)
        Edge(from_layer=layer_2, to_layer=layer_3, delay_iterations=0)
        Edge(from_layer=layer_3, to_layer=layer_1, delay_iterations=0)
    because the network is being asked to
        take "output of layer_1 in PREVIOUS ITERATION" as "input to layer_2 in CURRENT ITERATION"
        take "output of layer_2 in current iteration" as "input to layer_3 in current iteration"
        take "output of layer_3 in current iteration" as "input to layer_1 in current iteration"
    Which is possible, and crucial for Recurrent Neural Networks.
    Note that "previous iteration" in code means "delay_iterations=1".
    """


class LayerParentError(Exception):
    """
    The primary cause of this error, is if an Edge receives two layers which belong
    to two different instances of Network class. If you are attaining to create
    Functional Neural Network (Combining multiple Neural Networks as one), then
    you don`t actually have to instantiate the Network class twice.
    Separate two networks virtually.
    """


class IsolatedLayerError(Exception):
    """
    The primary cause of this error, is if a Layer has no
    edge connected to its input or output. The layer is
    Isolated from the network.
    """