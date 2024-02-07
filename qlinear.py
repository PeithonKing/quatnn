import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quat_base import _construct_matrix

class QLinear(nn.Module):
    r"""Quaternion Linear Layer.
    
    # All features of torch.nn.Linear layer from PyTorch are supported.
    
    # You need to remember these things when using this layer:
    #     - in_features and out_features must be divisible by 4.
    #     - We do not use quaternion for bias. bias count is not reduced. Only weight count is reduced to 25%.

    Examples:
        >>> model = QLinear(20, 16)  # 20 and 16 are divisible by 4
        >>> x = torch.randn(128, 20)
        >>> output = model(x)
        >>> print(output.size())
        torch.Size([128, 16])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    r_weight: torch.Tensor
    i_weight: torch.Tensor
    j_weight: torch.Tensor
    k_weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        assert in_features % 4 == 0 and out_features % 4 == 0, "in_channels and out_channels must be divisible by 4"
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_weight = nn.Parameter(torch.empty((out_features//4, in_features//4), **factory_kwargs))
        self.i_weight = nn.Parameter(torch.empty((out_features//4, in_features//4), **factory_kwargs))
        self.j_weight = nn.Parameter(torch.empty((out_features//4, in_features//4), **factory_kwargs))
        self.k_weight = nn.Parameter(torch.empty((out_features//4, in_features//4), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))
        if self.bias is not None:
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_in = self.in_features * 4
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


if __name__ == '__main__':
    # model = nn.Linear(20, 16)
    model = QLinear(20, 16)  # 20 and 16 are divisible by 4
    x = torch.randn(128, 20)
    output = model(x)
    print(output.size())
    # print(f"{model.weight.size() = }")
    print(f"{model.r_weight.size() = },\n{model.i_weight.size() = },\n{model.j_weight.size() = },\n{model.k_weight.size() = }")
