import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
import torch.nn as nn
# from .lazy import LazyModuleMixin
# from .module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
# from torch._torch_docs import reproducibility_notes

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union

from quat_base import _construct_matrix


class _QConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    r_weight: Tensor
    i_weight: Tensor
    j_weight: Tensor
    k_weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "in_channels and out_channels must be divisible by 4"

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.r_weight = Parameter(torch.empty((in_channels // 4, out_channels // (groups*4), *kernel_size), **factory_kwargs))
            self.i_weight = Parameter(torch.empty((in_channels // 4, out_channels // (groups*4), *kernel_size), **factory_kwargs))
            self.j_weight = Parameter(torch.empty((in_channels // 4, out_channels // (groups*4), *kernel_size), **factory_kwargs))
            self.k_weight = Parameter(torch.empty((in_channels // 4, out_channels // (groups*4), *kernel_size), **factory_kwargs))
        else:
            self.r_weight = Parameter(torch.empty((out_channels // 4, in_channels // (groups*4), *kernel_size), **factory_kwargs))
            self.i_weight = Parameter(torch.empty((out_channels // 4, in_channels // (groups*4), *kernel_size), **factory_kwargs))
            self.j_weight = Parameter(torch.empty((out_channels // 4, in_channels // (groups*4), *kernel_size), **factory_kwargs))
            self.k_weight = Parameter(torch.empty((out_channels // 4, in_channels // (groups*4), *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch.zeros((self.r_weight.size(0)*4, self.r_weight.size(1)*4), dtype=torch.bool))
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


# class QConv1d(_QConvNd):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: _size_1_t,
#         stride: _size_1_t = 1,
#         padding: Union[str, _size_1_t] = 0,
#         dilation: _size_1_t = 1,
#         groups: int = 1,
#         bias: bool = True,
#         padding_mode: str = 'zeros',  # TODO: refine this type
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         # we create new variables below to make mypy happy since kernel_size has
#         # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
#         kernel_size_ = _single(kernel_size)
#         stride_ = _single(stride)
#         padding_ = padding if isinstance(padding, str) else _single(padding)
#         dilation_ = _single(dilation)
#         super().__init__(
#             in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
#             False, _single(0), groups, bias, padding_mode, **factory_kwargs)

#     def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
#         if self.padding_mode != 'zeros':
#             return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#                             weight, bias, self.stride,
#                             _single(0), self.dilation, self.groups)
#         return F.conv1d(input, weight, bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def forward(self, input: Tensor) -> Tensor:
#         weight = _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)
#         return self._conv_forward(input, weight, self.bias)


class QConv2d(_QConvNd):
    """Quaternion convolution 2d.

    Examples:
        >>> model = QConv2d(20, 16, kernel_size=3, stride=1, padding=1)  # 20 and 16 are divisible by 4
        >>> x = torch.randn(128, 20, 32, 32)
        >>> output = model(x)
        >>> print(output.size())
        torch.Size([128, 16, 30, 30])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        weight = _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)
        return self._conv_forward(input, weight, self.bias)

# class QConv3d(_QConvNd):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: _size_3_t,
#         stride: _size_3_t = 1,
#         padding: Union[str, _size_3_t] = 0,
#         dilation: _size_3_t = 1,
#         groups: int = 1,
#         bias: bool = True,
#         padding_mode: str = 'zeros',
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         kernel_size_ = _triple(kernel_size)
#         stride_ = _triple(stride)
#         padding_ = padding if isinstance(padding, str) else _triple(padding)
#         dilation_ = _triple(dilation)
#         super().__init__(
#             in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
#             False, _triple(0), groups, bias, padding_mode, **factory_kwargs)

#     def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
#         if self.padding_mode != 'zeros':
#             return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#                             weight, bias, self.stride,
#                             _triple(0), self.dilation, self.groups)
#         return F.conv3d(input, weight, bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def forward(self, input: Tensor) -> Tensor:
#         weight = _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)
#         return self._conv_forward(input, weight, self.bias)


if __name__ == '__main__':
    # model = nn.Conv2d(20, 16, kernel_size=3, stride=1, padding=1)
    model = QConv2d(20, 16, kernel_size=3, stride=1, padding=1)  # 20 and 16 are divisible by 4
    x = torch.randn(128, 20, 32, 32)
    output = model(x)
    print(output.size())
    # print(f"{model.weight.size() = }")
    print(f"{model.r_weight.size() = },\n{model.i_weight.size() = },\n{model.j_weight.size() = },\n{model.k_weight.size() = }")
