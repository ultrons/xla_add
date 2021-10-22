import torch
import torch_xla.core.xla_model as xm
from  .initialize import get_model_parallel_global_group 

# Following method is an extention of xm.save method
# it has the same signature
# with one difference in the meaning
# Master only means the master replica only
# i.e. all the shards of the master replica
def save(data, file_or_path, master_only=True, global_master=False):
    master_group = get_model_parallel_global_group()[0]
    rank = xm.get_ordinal() if global_master else xm.get_local_ordinal()
    should_write_data = not master_only or xm.is_master_ordinal(
      local=not global_master) or (rank in master_group)  

 
    cpu_data = xm._maybe_convert_to_cpu(data, convert=should_write_data)
    if should_write_data:
      torch.save(cpu_data, file_or_path)
    xm.rendezvous('torch_xla.core.xla_model.save')
    
