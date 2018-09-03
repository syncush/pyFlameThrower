import torch
from torch.nn import Module
from torch.nn import Linear, BatchNorm1d, Dropout

class DnnFactory(Module):
    def __init__(self, fc_layers_sizes=[], regularization_layers_index=[]):
        super(DnnFactory, self).__init__()
        self.layers = []
        if len(fc_layers_sizes) >= 2:
            first = fc_layers_sizes[0]
            second = fc_layers_sizes[1]
            self.layers.append(Linear(first, second))
            for i in range(2, len(fc_layers_sizes)):
                first = second
                second = fc_layers_sizes[i]
                self.layers.append(Linear(first, second))
        else:
            raise ValueError("Input and Outsize are Mandatory")

