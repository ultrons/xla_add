# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_xla.core.xla_model as xm
from .initialize import get_model_parallel_global_group as get_model_parallel_group
from .initialize import get_model_parallel_world_size
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size as get_world_size
from .initialize import get_model_parallel_rank as get_rank
from .utils import split_tensor_along_last_dim
from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    #if get_world_size(group=group) == 1:
    if get_world_size() == 1:
        return input_

    # All-reduce.
    #xm.all_reduce(xm.REDUCE_SUM, [input_], groups=group[1])
    xm.all_reduce(xm.REDUCE_SUM, [input_], groups=group, pin_layout=False)

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    #if get_world_size(group=group) == 1:
    if get_world_size() == 1:
        return input_

    # Split along last dimension.
    #world_size = get_world_size(group=group)
    world_size = get_world_size()
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    #rank = get_rank(group=group)
    rank = get_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    #if get_world_size(group=group) == 1:
    if get_world_size() == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1

    #tensor_list = all_gather(tensor=input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    #output = torch.cat(tensor_list, dim=last_dim).contiguous()
    #output = xm.all_gather(input_, groups=get_model_parallel_group()[1])
    output = xm.all_gather(input_, groups=get_model_parallel_group(), pin_layout=False)

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        ctx.mark_dirty(input_)
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------

def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)
