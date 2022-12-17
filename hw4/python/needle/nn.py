"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                in_features,
                out_features,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )
        self.require_bias = bias
        if bias:
            self.bias = Parameter(
                ops.transpose(
                    init.kaiming_uniform(
                        out_features, 1, dtype=dtype, device=device, requires_grad=True
                    )
                )
            )
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.require_bias:
            return ops.matmul(X, self.weight) + ops.broadcast_to(
                self.bias, (X.shape[0], self.out_features)
            )
        else:
            return ops.matmul(X, self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        dimension = 1
        for i in X.shape[1:]:
            dimension = dimension * i
        return ops.reshape(X, (batch_size, dimension))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar((ops.exp(-x) + 1), -1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m = logits.shape[0]  # num of samples
        k = logits.shape[1]  # dimensionality of output
        return (
            ops.summation(
                ops.logsumexp(logits, axes=(1,))
                - ops.summation(
                    logits * init.one_hot(k, y, device=logits.device), axes=(1,)
                )
            )
            / m
        )
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.running_mean = Tensor(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.running_var = Tensor(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(0,)) / x.shape[0]  # (dim, )
        centerlized_x = x - ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)
        var = ops.summation(ops.power_scalar(centerlized_x, 2), axes=(0,)) / x.shape[0]
        if self.training:
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
            return ops.divide(
                ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
                * (centerlized_x),
                ops.broadcast_to(
                    ops.reshape(ops.power_scalar(var + self.eps, 0.5), (1, self.dim)),
                    x.shape,
                ),
            ) + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        else:
            return ops.divide(
                ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
                * (
                    x
                    - ops.broadcast_to(
                        ops.reshape(self.running_mean, (1, self.dim)), x.shape
                    )
                ),
                ops.broadcast_to(
                    ops.reshape(
                        ops.power_scalar(self.running_var + self.eps, 0.5),
                        (1, self.dim),
                    ),
                    x.shape,
                ),
            ) + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(np.ones(dim), dtype=dtype)
        self.bias = Parameter(np.zeros(dim), dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # lesson : Explicitly write all the broadcast steps. If we omit the broadcast in element-wise division,
        # the backward method will crash. Same situation will happen if we omit the broadcast in computing
        # centerlized_x.
        mean = ops.reshape(ops.summation(x, axes=(1,)) / x.shape[1], (x.shape[0], 1))
        centerlized_x = x - ops.broadcast_to(mean, x.shape)
        var = ops.reshape(
            ops.summation(ops.power_scalar(centerlized_x, 2), axes=(1,)) / x.shape[1],
            (x.shape[0], 1),
        )
        return ops.divide(
            ops.broadcast_to(self.weight, x.shape) * (centerlized_x),
            ops.broadcast_to(ops.power_scalar(var + self.eps, 0.5), x.shape),
        ) + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return init.randb(*x.shape, p=1 - self.p) * x / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                self.kernel_size**2 * self.in_channels,
                self.kernel_size**2 * self.out_channels,
                shape=(
                    self.kernel_size,
                    self.kernel_size,
                    self.in_channels,
                    self.out_channels,
                ),
            ),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        bound = 1 / (self.in_channels * self.kernel_size**2) ** 0.5
        if bias:
            self.bias = Parameter(
                init.rand(
                    self.out_channels,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:  # x: NCHW
        ### BEGIN YOUR SOLUTION
        nchw_x = ops.transpose(ops.transpose(x, (1, 2)), (2, 3))
        nchw_out = ops.conv(
            nchw_x, self.weight, stride=self.stride, padding=(self.kernel_size) // 2
        )
        if self.bias:
            nchw_out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(
                nchw_out.shape
            )
        nhwc_out = ops.transpose(ops.transpose(nchw_out, (2, 3)), (1, 2))
        return nhwc_out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size
        bound = (1 / hidden_size) ** 0.5
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h == None:
            h = init.zeros(bs, self.hidden_size, dtype=X.dtype, device=X.device)
        if self.bias_ih:
            input = (
                h @ self.W_hh
                + X @ self.W_ih
                + ops.broadcast_to(
                    ops.reshape(self.bias_hh, (1, self.hidden_size)),
                    (bs, self.hidden_size),
                )
                + ops.broadcast_to(
                    ops.reshape(self.bias_ih, (1, self.hidden_size)),
                    (bs, self.hidden_size),
                )
            )
        else:
            input = h @ self.W_hh + X @ self.W_ih
        if self.nonlinearity == "tanh":
            return ops.tanh(input)
        elif self.nonlinearity == "relu":
            return ops.relu(input)
        else:
            raise ValueError("Not supported nonlinear module")
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
        ]
        for _ in range(num_layers - 1):
            self.rnn_cells.append(
                RNNCell(
                    hidden_size,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    device=device,
                    dtype=dtype,
                )
            )
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len = X.shape[0]
        bs = X.shape[1]
        X_t = ops.split(X, axis=0)
        h_t = []
        if h0:
            h_n = list(ops.split(h0, axis=0))
        else:
            h_n = [None] * self.num_layers
        for t in range(seq_len):
            X_in = X_t[t]
            for i, rnn_cell in enumerate(self.rnn_cells):
                h = rnn_cell(X_in, h_n[i])
                X_in = h
                h_n[i] = h
            h_t.append(X_in)
        return ops.stack(h_t, axis=0), ops.stack(h_n, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = (1 / hidden_size) ** 0.5
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None or h == (None, None):
            h0 = init.zeros(bs, self.hidden_size, dtype=X.dtype, device=X.device)
            c0 = init.zeros(bs, self.hidden_size, dtype=X.dtype, device=X.device)
        else:
            h0, c0 = h

        Z = X @ self.W_ih + h0 @ self.W_hh

        if self.bias_ih:
            Z += self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(
                (bs, 4 * self.hidden_size)
            ) + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(
                (bs, 4 * self.hidden_size)
            )

        Z_split = ops.split(Z, axis=1)
        hs = self.hidden_size
        i = ops.stack(tuple([Z_split[i] for i in range(0, hs)]), 1)
        f = ops.stack(tuple([Z_split[i] for i in range(hs, 2 * hs)]), 1)
        g = ops.stack(tuple([Z_split[i] for i in range(2 * hs, 3 * hs)]), 1)
        o = ops.stack(tuple([Z_split[i] for i in range(3 * hs, 4 * hs)]), 1)
        i = Sigmoid()(i)
        f = Sigmoid()(f)
        g = Tanh()(g)
        o = Sigmoid()(o)

        # i = Sigmoid()(ops.stack(Z_split[:self.hidden_size], axis=1))
        # f = Sigmoid()(ops.stack(Z_split[self.hidden_size: 2*self.hidden_size], axis=1))
        # g = Tanh()(ops.stack(Z_split[2*self.hidden_size:3*self.hidden_size], axis=1))
        # o = Sigmoid()(ops.stack(Z_split[3*self.hidden_size:], axis=1))

        c = f * c0 + i * g
        h = o * Tanh()(c)
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.
        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = [
            LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)
        ]
        for _ in range(num_layers - 1):
            self.lstm_cells.append(
                LSTMCell(
                    hidden_size, hidden_size, bias=bias, device=device, dtype=dtype
                )
            )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.
        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        X_list = ops.split(X, axis=0)

        out_list = []
        if h:
            h0, c0 = h
            h_list = list(ops.split(h0, axis=0))
            c_list = list(ops.split(c0, axis=0))
        else:
            h_list = [None] * self.num_layers
            c_list = [None] * self.num_layers

        for t in range(seq_len):
            X_in = X_list[t]
            for i, lstm_cell in enumerate(self.lstm_cells):
                h, c = lstm_cell(X_in, (h_list[i], c_list[i]))
                X_in = h
                h_list[i] = h
                c_list[i] = c
            out_list.append(X_in)

        return ops.stack(out_list, axis=0), (
            ops.stack(h_list, axis=0),
            ops.stack(c_list, axis=0),
        )
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                mean=0.0,
                std=1.0,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        one_hot_vec = init.one_hot(
            self.num_embeddings, x, device=x.device, dtype=x.dtype
        )
        embedded_x = (
            one_hot_vec.reshape((seq_len * bs, self.num_embeddings)) @ self.weight
        )
        return embedded_x.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION
