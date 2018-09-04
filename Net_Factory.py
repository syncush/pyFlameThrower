from torch.nn import Module
from torch.nn import Linear, BatchNorm1d, Dropout
from torch.nn import ReLU, PReLU, LeakyReLU, Tanh, Sigmoid, LogSigmoid, Softmax, LogSoftmax


relu_names = ['relu', 'Relu', 'ReLU']
softmax_names = ['Softmax', 'softmax', 'SoftMax', 'sm']
tanh_names = ['tanh', 'Tanh', 'TanH', 'tanH']
sigmoid_names = ['sigmoid', 'Sigmoid']

leaky_relu_names = ['leaky_' + x for x in relu_names]
p_relu_names = ['p' + x for x in relu_names]
log_sigmoid_name = ['log_' + x for x in sigmoid_names]
log_softmax_names = ['log_' + x for x in softmax_names]


def parse_activation(name):
    if name in relu_names:
        return ReLU()
    if name in leaky_relu_names:
        return LeakyReLU()
    if name in sigmoid_names:
        return Sigmoid()
    if name in log_sigmoid_name:
        return LogSigmoid()
    if name in p_relu_names:
        return PReLU()
    if name in tanh_names:
        return Tanh()
    if name in softmax_names:
        return Softmax(dim=1)
    if name in log_softmax_names:
        return LogSoftmax(dim=1)


class DnnFactory(Module):
    def __init__(self, fc_layers_sizes=[], regularization_layers_index=[], activations=[]):
        super(DnnFactory, self).__init__()
        self.layers = []
        bn_dict = {}
        do_dict = {}
        activations.insert(0, None)
        for regularization_name, layer_index in regularization_layers_index:
            if regularization_name == 'BatchNorm1d':
                bn_dict[layer_index] = 1
            else:
                layer_index, drp_rate = layer_index
                do_dict[layer_index] = drp_rate
        if len(fc_layers_sizes) >= 2:
            first = fc_layers_sizes[0]
            second = fc_layers_sizes[1]
            self.layers.append(Linear(first, second))
            if 1 in bn_dict:
                self.layers.append(BatchNorm1d(second))
            self.layers.append(parse_activation(activations[1]))
            for i in range(2, len(fc_layers_sizes)):
                first = second
                second = fc_layers_sizes[i]
                self.layers.append(Linear(first, second))
                if i in bn_dict:
                    self.layers.append(BatchNorm1d(second))
                self.layers.append(parse_activation(activations[i]))
                if i in do_dict:
                    self.layers.append(Dropout(p=do_dict[i]))
        else:
            raise ValueError("Input and Outsize are Mandatory")

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
