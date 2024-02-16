"""
Quaternion layers:
    - [x] QLinear
    - [.] QConv1d
    - [.] QConv2d
    - [.] QConv3d
    - [ ] QMultiheadAttention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _construct_matrix(r, i, j, k):
    """Construct real weight matrix from the quaternion weights.

    Args:
        r, i, j, k (torch.tensor): parts of the quaternion weights

    Returns:
        torch.tensor: real weight matrix
    """
    weight = torch.cat([torch.cat([r, -i, -j, -k], dim=0),
                        torch.cat([i,  r, -k,  j], dim=0),  # noqa: E241
                        torch.cat([j,  k,  r, -i], dim=0),  # noqa: E241
                        torch.cat([k, -j,  i,  r], dim=0)], dim=1)  # noqa: E241

    return weight


def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1) // 4
    split_sizes = [dim] * 4
    r, i, j, k = torch.split(kernel, split_sizes, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton



# class QConv2d(nn.Module):
#     """Quaternion convolution 2d.

#     Args:
#         in_channels (int): number of input channels (must be divisible by 4)
#         out_channels (int): number of output channels (must be divisible by 4)
#         kernel_size (int): size of the kernel
#         stride (int): stride of the convolution. Default: 1
#         padding (int): padding of the convolution. Default: 0
#         groups (int): number of groups. Default: 1
#         bias (bool): whether to use bias. Default: True
#         dilation (int): dilation of the convolution. Default: 1

#     Raises:
#         AssertionError: if in_channels or out_channels are not divisible by 4
#     """

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, groups=1, bias=True, dilation=1):
#         assert in_channels % 4 == 0 and out_channels % 4 == 0, "in_channels and out_channels must be divisible by 4"
#         in_channels = in_channels // 4
#         out_channels = out_channels // 4
#         super(QConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups
#         self.bias = bias
#         self.dilation = dilation

#         self.r_weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
#         self.i_weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
#         self.j_weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
#         self.k_weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
#         self.bias = nn.Parameter(torch.empty(out_channels * 4)) if bias else None

#         self._reset_parameters()

#     def _reset_parameters(self):
#         nn.init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.r_weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def forward(self, x):  # noqa: D102

#         weight = _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)

#         ret = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

#         return ret
