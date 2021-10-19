import os
os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
os.environ['PT_XLA_DEBUG'] = '1'


import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch.nn as nn
import torch.optim as optim

import torch_xla.debug.metrics as met
from dataclasses import dataclass

#from fairseq.model_parallel.megatron.mpu import (
from xla_add.mpu.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

import xla_add.mpu as mpu
import xla_add.mpu.layers as layers


#lass sample_attention_block(nn.Module):
#   def __init__(self, config):
#       super('sample_mp_network', self).__init__()
#       self.query = nn.Linear(config.d_model, config.inner_dim)
#       self.key = nn.Linear(config.d_model, config.inner_dim)
#       self.value = nn.Linear(config.d_model, config.inner_dim)
#       self.out = nn.Linear(config.inner_dim, config.d_model)
#
#   def forward(self, x):
#       q = self.query(x) #(b,s,h,d) -> b,s,h,q
#       k = self.key(x) #(b,s,h,d) -> b,s,h,q
#       v = self.value(x) #(b,s,h,d) -> b,s,h,q
#
#       outer = torch.einsum('bshq,bthq->bsth', q, k) 
#       alpha = F.softmax(outer,1)
#       inner = torch.einsum('bsth,bthq->bshq', alpha, v)
#       y = self.out(inner)

class sample_network(nn.Module):
    def __init__(self):
        super(sample_network, self).__init__()
        #self.linear1 = nn.Linear(64, 32)
        #self.linear2 = nn.Linear(32, 10)
        self.linear1 = ColumnParallelLinear(64, 32, gather_output=False, bias=False)
        self.linear2 = RowParallelLinear(32, 10, input_is_parallel=True, bias=False)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.out(x)




def per_core_action(index):
    tensor_model_parallel_size = 4
    mpu.initialize_model_parallel(tensor_model_parallel_size)
    data_parallel_rank = mpu.initialize.get_data_parallel_rank()
    data_parallel_world_size = mpu.initialize.get_data_parallel_world_size()
    data_parallel_group = mpu.initialize.get_data_parallel_global_group()

    device = xm.xla_device()
    model = sample_network().to(device)

    # Create dataset
    data = torch.ones((128, 7, 64))
    dataset = torch.utils.data.TensorDataset(data)
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(
          dataset,
          num_replicas=(xm.xrt_world_size() // tensor_model_parallel_size),
          rank=data_parallel_rank,
          shuffle=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=distributed_sampler)
    
    parallel_loader = pl.MpDeviceLoader(dataloader, device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print(f"Number of elements in the dataset:{len(dataloader)}") 
    for sample, in parallel_loader:
        y_hat = model(sample)
        loss = y_hat.sum()
        print(f"Debug: Rank:{xm.get_ordinal()} Finished Forward Pass")
        print(torch_xla._XLAC._get_xla_tensors_text([loss]))
        
        loss.backward()
        xm.optimizer_step(optimizer, groups=data_parallel_group)


        print(f"Debug: Rank:{xm.get_ordinal()} DP Group:{data_parallel_group} loss :{loss}")

    xm.rendezvous('All processes meet here!')



if __name__ == '__main__':
    import sys
    print(sys.path)
    print("No News!")
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(per_core_action, nprocs=8, start_method='fork')
