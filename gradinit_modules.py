import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init
from typing import Tuple, Union, TypeVar
import math

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]


class GradInitConv2d(torch.nn.modules.conv._ConvNd):
    __doc__ = r"""Convolutional layer with GradInit support.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GradInitConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # initialize the bias as zero
        if self.bias is not None:
            self.bias.data.zero_()
        self.opt_mode_ = False

    def _conv_forward(self, input: Tensor):
        if self.opt_mode_:
            # will be set externally
            weight = self.opt_weight.to(input)
            bias = self.opt_bias
            if bias is not None:
                bias = bias.to(input)
        else:
            weight = self.weight
            bias = self.bias

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input)

    def opt_mode(self, mode=True):
        self.opt_mode_ = mode


class GradInitBatchNorm2d(torch.nn.modules.batchnorm._BatchNorm):
    r"""Batch normalization layer with GradInit support.
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
    ) -> None:
        super(GradInitBatchNorm2d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats)
        self.gradinit_ = False
        self.opt_mode_ = False

    def forward(self, x):
        if self.gradinit_:
            if self.opt_mode_:
                weight = self.opt_weight.to(x)
                bias = self.opt_bias.to(x)
            else:
                weight = self.weight
                bias = self.bias
            return F.batch_norm(
                x,
                # If buffers are not to be tracked, ensure that they won't be updated
                None,
                None,
                weight, bias, True, self.momentum, self.eps)
        else:

            self._check_input_dim(x)

            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that it gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:  # type: ignore
                    self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            return F.batch_norm(
                x,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

    def gradinit(self, mode=True):
        self.gradinit_ = mode

    def opt_mode(self, mode=True):
        self.opt_mode_ = mode

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class GradInitLinear(torch.nn.Module):
    r"""Linear operator with GradInit support
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(GradInitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.opt_mode_ = False

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.opt_mode_:
            weight = self.opt_weight.to(input)
            bias = self.opt_bias
            if bias is not None:
                bias = bias.to(input)
        else:
            weight = self.weight
            bias = self.bias
        return F.linear(input, weight, bias)

    def opt_mode(self, mode=True):
        self.opt_mode_ = mode

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GradInitScale(torch.nn.Module):
    def __init__(self):
        super(GradInitScale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

        self.opt_mode_ = False

    def forward(self, x):
        if self.opt_mode_:
            weight = self.opt_weight.to(x)
        else:
            weight = self.weight

        return x * weight

    def opt_mode(self, mode=True):
        self.opt_mode_ = mode


class GradInitBias(torch.nn.Module):
    def __init__(self):
        super(GradInitBias, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.opt_mode_ = False

    def forward(self, x):
        if self.opt_mode_:
            bias = self.opt_bias.to(x)
        else:
            bias = self.bias

        return x + bias

    def opt_mode(self, mode=True):
        self.opt_mode_ = mode
