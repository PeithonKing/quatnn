# import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional as F
from quat_base import _construct_matrix


def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        return any(type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode for x in torch_dispatch_mode_stack)
    else:
        return False
    
nn.Transformer

class QMultiheadAttention(nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            # self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            # self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            # self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.r_q_proj_weight = Parameter(torch.empty((embed_dim//4, embed_dim//4), **factory_kwargs))
            self.i_q_proj_weight = Parameter(torch.empty((embed_dim//4, embed_dim//4), **factory_kwargs))
            self.j_q_proj_weight = Parameter(torch.empty((embed_dim//4, embed_dim//4), **factory_kwargs))
            self.k_q_proj_weight = Parameter(torch.empty((embed_dim//4, embed_dim//4), **factory_kwargs))
            
            self.r_k_proj_weight = Parameter(torch.empty((embed_dim//4, self.kdim), **factory_kwargs))
            self.i_k_proj_weight = Parameter(torch.empty((embed_dim//4, self.kdim), **factory_kwargs))
            self.j_k_proj_weight = Parameter(torch.empty((embed_dim//4, self.kdim), **factory_kwargs))
            self.k_k_proj_weight = Parameter(torch.empty((embed_dim//4, self.kdim), **factory_kwargs))
            
            self.r_v_proj_weight = Parameter(torch.empty((embed_dim//4, self.vdim), **factory_kwargs))
            self.i_v_proj_weight = Parameter(torch.empty((embed_dim//4, self.vdim), **factory_kwargs))
            self.j_v_proj_weight = Parameter(torch.empty((embed_dim//4, self.vdim), **factory_kwargs))
            self.k_v_proj_weight = Parameter(torch.empty((embed_dim//4, self.vdim), **factory_kwargs))
            self.register_parameter('r_in_proj_weight', None)
            self.register_parameter('i_in_proj_weight', None)
            self.register_parameter('j_in_proj_weight', None)
            self.register_parameter('k_in_proj_weight', None)
        else:
            # in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.r_in_proj_weight = Parameter(torch.empty((3 * embed_dim//4, embed_dim//4), **factory_kwargs))
            self.i_in_proj_weight = Parameter(torch.empty((3 * embed_dim//4, embed_dim//4), **factory_kwargs))
            self.j_in_proj_weight = Parameter(torch.empty((3 * embed_dim//4, embed_dim//4), **factory_kwargs))
            self.k_in_proj_weight = Parameter(torch.empty((3 * embed_dim//4, embed_dim//4), **factory_kwargs))
            self.register_parameter('r_q_proj_weight', None)
            self.register_parameter('i_q_proj_weight', None)
            self.register_parameter('j_q_proj_weight', None)
            self.register_parameter('k_q_proj_weight', None)
            
            self.register_parameter('r_k_proj_weight', None)
            self.register_parameter('i_k_proj_weight', None)
            self.register_parameter('j_k_proj_weight', None)
            self.register_parameter('k_k_proj_weight', None)
            
            self.register_parameter('r_v_proj_weight', None)
            self.register_parameter('i_v_proj_weight', None)
            self.register_parameter('j_v_proj_weight', None)
            self.register_parameter('k_v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.r_in_proj_weight)
            xavier_uniform_(self.i_in_proj_weight)
            xavier_uniform_(self.j_in_proj_weight)
            xavier_uniform_(self.k_in_proj_weight)
        else:                                     
            xavier_uniform_(self.r_q_proj_weight)
            xavier_uniform_(self.i_q_proj_weight)
            xavier_uniform_(self.j_q_proj_weight)
            xavier_uniform_(self.k_q_proj_weight)
            
            xavier_uniform_(self.r_k_proj_weight)
            xavier_uniform_(self.i_k_proj_weight)
            xavier_uniform_(self.j_k_proj_weight)
            xavier_uniform_(self.k_k_proj_weight)
            
            xavier_uniform_(self.r_v_proj_weight)
            xavier_uniform_(self.i_v_proj_weight)
            xavier_uniform_(self.j_v_proj_weight)
            xavier_uniform_(self.k_v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal : bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        

        if self._qkv_same_embed_dim:
            in_proj_weight = _construct_matrix(self.r_in_proj_weight, self.i_in_proj_weight, self.j_in_proj_weight, self.k_in_proj_weight)
            q_proj_weight = None
            k_proj_weight = None
            v_proj_weight = None
        else:
            in_proj_weight = None
            q_proj_weight = _construct_matrix(self.r_q_proj_weight, self.i_q_proj_weight, self.j_q_proj_weight, self.k_q_proj_weight)
            k_proj_weight = _construct_matrix(self.r_k_proj_weight, self.i_k_proj_weight, self.j_k_proj_weight, self.k_k_proj_weight)
            v_proj_weight = _construct_matrix(self.r_v_proj_weight, self.i_v_proj_weight, self.j_v_proj_weight, self.k_v_proj_weight)
        

        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != in_proj_weight.dtype:
            # this case will fail anyway, but at least we will get an useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and in_proj_weight ({in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

