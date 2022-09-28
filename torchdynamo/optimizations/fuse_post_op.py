import binascii
import copy
import operator

import torch
from torch import Tensor
from torch.nn import functional as F
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from torch.fx.experimental.optimization import matches_module_pattern
from torch.nn.modules.utils import _pair
from typing import Optional, List, Tuple, Union, Dict, Any
from torch.nn.common_types import _size_2_t
from torch.nn.modules.conv import _ConvNd

class LinearEltwise(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(LinearEltwise, self).__init__(in_features, out_features, bias=bias,
            device=device, dtype=dtype)
    
    def forward(self, input):
        y = torch.ops.mkldnn_prepacked.linear_eltwise(input, self.weight, self.bias, self.attr, self.scalars, self.algorithm)
        return y

    def update_status(self, eltwise, attr, extra_inputs):
        self.attr = attr

        scalars = []
        for item in extra_inputs.scalars:
            assert hasattr(eltwise, item)
            scalars.append(getattr(eltwise, item))
        self.scalars = scalars

        algorithm = ""
        if extra_inputs.algorithm:
            assert hasattr(eltwise, extra_inputs.algorithm)
            algorithm = getattr(eltwise, extra_inputs.algorithm) 
        self.algorithm = algorithm

class ConvEltwise(_ConvNd):
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
        padding_mode: str = 'zeros',  # TODO: refine this type,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(ConvEltwise, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return torch.ops.mkldnn_fusion.mkldnn_convolution_elementwise(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight, bias, self.stride, _pair(0), self.dilation, self.groups,
                self.attr, self.scalars, self.algorithm)
        return torch.ops.mkldnn_fusion.mkldnn_convolution_elementwise(input, weight, bias,
              self.padding, self.stride, self.dilation,
              self.groups, self.attr, self.scalars, self.algorithm)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    def update_status(self, eltwise, attr, extra_inputs):
        self.attr = attr

        scalars = []
        for item in extra_inputs.scalars:
            assert hasattr(eltwise, item)
            scalars.append(getattr(eltwise, item))
        self.scalars = scalars

        algorithm = ""
        if extra_inputs.algorithm:
            assert hasattr(eltwise, extra_inputs.algorithm)
            algorithm = getattr(eltwise, extra_inputs.algorithm) 
        self.algorithm = algorithm

class ConvBinary(_ConvNd):
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
        padding_mode: str = 'zeros',  # TODO: refine this type,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(ConvBinary, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, other: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return torch.ops.mkldnn_fusion.mkldnn_convolution_binary(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                other, weight, bias, self.stride, _pair(0), self.dilation, self.groups,
                self.attr,)
        return torch.ops.mkldnn_fusion.mkldnn_convolution_binary(input, other, weight, bias,
              self.padding, self.stride, self.dilation,
              self.groups, self.attr)

    def forward(self, input: Tensor, other: Tensor) -> Tensor:
        return self._conv_forward(input, other, self.weight, self.bias)

    def update_status(self, attr):
        self.attr = attr

def fuse_linear_eltwise_eval(linear, eltwise, attr, extra_inputs):
    linear_eltwise = LinearEltwise(linear.in_features,
                              linear.out_features,
                              linear.bias is not None,
                              linear.weight.device,
                              linear.weight.dtype)
    linear_eltwise.__dict__ = copy.deepcopy(linear.__dict__)
    # TODO: set this in init func is not working, due to copy __dict__??
    linear_eltwise.update_status(eltwise, attr, extra_inputs)
    return linear_eltwise

def fuse_conv_eltwise_eval(conv, eltwise, attr, extra_inputs):
    assert(not (conv.training)), "Fusion only for eval!"
    conv_eltwise = ConvEltwise(conv.in_channels,
                               conv.out_channels,
                               conv.kernel_size,
                               conv.stride,
                               conv.padding,
                               conv.dilation,
                               conv.groups,
                               conv.bias is not None,
                               conv.padding_mode,
                               conv.weight.device,
                               conv.weight.dtype)
    conv_eltwise.weight = torch.nn.Parameter(conv.weight.detach().clone(),requires_grad = conv.weight.requires_grad)
    if conv.bias is not None:
        conv_eltwise.bias = torch.nn.Parameter(conv.bias.detach().clone(),requires_grad = conv.bias.requires_grad)
    conv_eltwise.update_status(eltwise, attr, extra_inputs)
    return conv_eltwise


def fuse_conv_binary_eval(conv, attr):
    assert(not (conv.training)), "Fusion only for eval!"
    conv_binary = ConvBinary(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           conv.dilation,
                           conv.groups,
                           conv.bias is not None,
                           conv.padding_mode,
                           conv.weight.device,
                           conv.weight.dtype)
    conv_binary.weight = torch.nn.Parameter(conv.weight.detach().clone(),requires_grad = conv.weight.requires_grad)
    if conv.bias is not None:
        conv_binary.bias = torch.nn.Parameter(conv.bias.detach().clone(),requires_grad = conv.bias.requires_grad)
    conv_binary.update_status(attr)
    return conv_binary


class EltwiseFusionOp:
    def __init__(self, scalars=[], algorithm=""):
        self.scalars = scalars
        self.algorithm = algorithm

attr_names = {
    "relu": EltwiseFusionOp(),
    "sigmoid": EltwiseFusionOp(),
    "tanh": EltwiseFusionOp(),
    "hardswish": EltwiseFusionOp(),
    "leaky_relu": EltwiseFusionOp(scalars=["negative_slope"]),
    "hardtanh": EltwiseFusionOp(scalars=["min_val", "max_val"]),
    "gelu": EltwiseFusionOp(algorithm="approximate"),
}

linear_patterns = [
    (torch.nn.Linear, torch.nn.ReLU),
    (torch.nn.Linear, torch.nn.Sigmoid),
    (torch.nn.Linear, torch.nn.Tanh),
    (torch.nn.Linear, torch.nn.Hardswish),
    (torch.nn.Linear, torch.nn.LeakyReLU),
    (torch.nn.Linear, torch.nn.Hardtanh),
    (torch.nn.Linear, torch.nn.GELU),
]

conv_patterns = [
    (torch.nn.Conv2d, torch.nn.ReLU),
    (torch.nn.Conv2d, torch.nn.Sigmoid),
    (torch.nn.Conv2d, torch.nn.Tanh),
    (torch.nn.Conv2d, torch.nn.Hardswish),
    (torch.nn.Conv2d, torch.nn.LeakyReLU),
    (torch.nn.Conv2d, torch.nn.Hardtanh),
    (torch.nn.Conv2d, torch.nn.GELU),
]

def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)
    modules[node.target] = new_module

def fuse_post_op(gm, example_inputs):
    gm = fuse_eltwise(gm, example_inputs, linear_patterns)
    gm = fuse_eltwise(gm, example_inputs, conv_patterns)
    gm = conv_binary_fuse(gm)
    return gm

def fuse_eltwise(gm, example_inputs, patterns):
    modules = dict(gm.named_modules())
    new_graph = copy.deepcopy(gm.graph)
    assert len(patterns) == len(attr_names), "pattern and replacement length should be equal"
    for pattern, attr_name in zip(patterns, attr_names):
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of linear/conv2d is used by other nodes
                    continue
                conv_or_linear = modules[node.args[0].target]
                eltwise = modules[node.target]
                eval_mode = all(not n.training for n in [conv_or_linear, eltwise])

                tensors = example_inputs + [conv_or_linear.weight]
                if conv_or_linear.bias is not None:
                    tensors.append(conv_or_linear.bias)
                is_cpu = all(x.device == torch.device('cpu') for x in tensors)
                if eval_mode and is_cpu:
                    if type(conv_or_linear) is not torch.nn.Conv2d:
                        fused_module = fuse_linear_eltwise_eval(conv_or_linear, eltwise, attr_name, attr_names[attr_name])
                    else:
                        # TODO: check shape info.
                        if len(node.args[0].args) < 0:
                            continue
                        conv_input_meta = node.args[0].args[0].meta.get("tensor_meta")
                        if not conv_input_meta:
                            continue
                        if len(conv_input_meta.shape) != 4:
                            continue
                        fused_module = fuse_conv_eltwise_eval(conv_or_linear, eltwise, attr_name, attr_names[attr_name])
                    replace_node_module(node.args[0], modules, fused_module)
                    node.replace_all_uses_with(node.args[0])
                    new_graph.erase_node(node)
    gm =  fx.GraphModule(gm, new_graph)    
    return gm

def check_node_is_conv(current_node, modules):
    if not isinstance(current_node, fx.Node):
        return False
    if current_node.op != 'call_module':
        return False
    if not isinstance(current_node.target, str):
        return False
    if current_node.target not in modules:
        return False
    if type(modules[current_node.target]) is not torch.nn.Conv2d:
        return False
    return True

def check_node_is_binary(node):
    if (node.op == 'call_function' and node.target in [torch.add, torch.sub]) or \
            (node.op == 'call_function' and node.target in [operator.add, operator.sub]) or \
             (node.op == 'call_method' and node.target in [torch.Tensor.add, torch.Tensor.sub]):
        return True
    return False

binary_attr = {
    torch.add: "add",
    torch.Tensor.add:"add",
    operator.add: "add",
    torch.sub: "sub",
    torch.Tensor.sub: "sub",
    operator.sub: "sub",
}

def conv_binary_fuse(fx_model) -> torch.nn.Module:
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)
    for node in new_graph.nodes:
        if check_node_is_binary(node) and (len(node.kwargs) != 2 or node.kwargs['alpha'] == 1.0):
            if not isinstance(node.args[0], torch.fx.Node) or not isinstance(node.args[1], torch.fx.Node):
                continue
            tensor0_meta = node.args[0].meta.get("tensor_meta")
            tensor1_meta = node.args[1].meta.get("tensor_meta")
            if not tensor0_meta or not tensor1_meta:
                continue
            if tensor0_meta.shape != tensor1_meta.shape or tensor0_meta.dtype != tensor1_meta.dtype:
                continue
            if check_node_is_conv(node.args[0], modules):
                if len(node.args[0].users) > 1:
                    continue
                conv = modules[node.args[0].target]
                attr = binary_attr[node.target]
                fused_bn = fuse_conv_binary_eval(conv, attr)
                replace_node_module(node.args[0], modules, fused_bn)
                node.args[0].args =  node.args[0].args + (node.args[1], )
                node.replace_all_uses_with(node.args[0])
            elif check_node_is_conv(node.args[1], modules):
                if len(node.args[1].users) > 1:
                    continue
                conv = modules[node.args[1].target]
                attr = binary_attr[node.target]
                fused_conv = fuse_conv_binary_eval(conv, attr)
                replace_node_module(node.args[1], modules, fused_conv)
                node.args[1].args =  node.args[1].args + (node.args[0], )
                node.replace_all_uses_with(node.args[1])
            else:
                continue
            new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)

