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

import os
os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
os.environ['PT_XLA_DEBUG'] = '1'
import random
import sys

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.nn.parameter import Parameter
import xla_add.mpu as mpu

from xla_add.mpu.tests.commons import set_random_seed, print_separator
from xla_add.mpu import layers

class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m , n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)
    def forward(self):
        return self.weight


def parallel_transformer(device, model_parallel_size, num_att_heads_per_partition,
                         hidden_size_per_att_head, batch_size, sequence_length):

    mpu.initialize_model_parallel(model_parallel_size)
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
                    xm.xrt_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads
    intermediate_size = 4 * hidden_size

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).to(device)
    transformer_layer = mpu.BertParallelTransformerLayer(
        hidden_size, intermediate_size, num_att_heads, 0.0, 0.0,
        torch.nn.functional.relu, 1.0e-5).to(device)

    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).to(device)
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).to(device)
    # Forward
    input_ = identity_layer()
    output = transformer_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, model_parallel_size, loss, \
        transformer_layer, identity_layer


def test_parallel_transformer_layer(model_parallel_size):

    xm.master_print('> testing ParallelTransformerLayer with model parallel '
              'size: {}'.format(model_parallel_size))

    device = xm.xla_device()

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    batch_size = 5
    sequence_length = 13

    rank_1, hidden_size_1, model_parallel_size_1, loss_1, \
        transformer_layer_1, identity_layer_1 = parallel_transformer(
            device,
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    rank, hidden_size, model_parallel_size, loss, \
        transformer_layer, identity_layer = parallel_transformer(
            device,
            model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    error = loss_1.sub(loss).abs().max()
    xm.rendezvous('all-ranks-sync')
    print('   loss error on global rank {}: {}'.format(
        xm.get_ordinal(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    xm.rendezvous('all-ranks-sync')
    print('   input gradient error on global rank {}: {}'.format(
        xm.get_ordinal(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    xm.rendezvous('all-ranks-sync')
    xm.master_print(' >> passed the test :-)')

def _mp_fn(index):
    model_parallel_size = 4
    test_parallel_transformer_layer(model_parallel_size)

if __name__ == '__main__':

    print('test parallel transformer')
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, nprocs=8, start_method='fork')
