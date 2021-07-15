import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from typing import List

class dist_init:
    """ 
    Sets static properties for distribution 
    Create an instance of this class during initialization.
    Methods:
    data_parallel_groups -> returns grouped ranks the input data will be shared into
    model_parallel_groups -> returns grouped ranks model is sharded into
    """

    _config = None

    @staticmethod
    def model_parallel_group():
        if dist_init._config is None:
            raise Exception("No instance of dist_init!!!")
        return dist_init._config['model_parallel_groups']

    @staticmethod
    def data_parallel_groups():
        if dist_init._config is None:
            raise Exception("No instance of dist_init!!!")
        return dist_init._config['data_parallel_groups']


    def __init__(self, model_parallel_size):
        if dist_init._config is not None:
            raise Exception("dist_init already exists")
        #world_size = xm.xrt_world_size()
        world_size = xmp._get_world_size() * xmp._get_devices_per_worker()
        default_group = [i for i in range(world_size)]
        
        if model_parallel_size == 1:
                   # Model Parallel Groups       # Data Parallel Group
            dist_init._config = {
                    'model_parallel_groups': [[i] for i in default_group],
                    'data_parallel_groups': [default_group] 
                    }
            return dist_init._config
        
        model_parallel_groups = []
        for i in range(world_size // model_parallel_size):
            grouped_ranks = default_group[i*model_parallel_size:model_parallel_size]
            model_parallel_groups.append(grouped_ranks)
        
        data_parallel_groups = []
        for i in range(model_parallel_size):
            grouped_ranks = default_group[i:world_size:model_parallel_size]
            data_parallel_groups.append(grouped_ranks)
        
        dist_init._config = {
                'model_parallel_groups': model_parallel_groups,
                'data_parallel_groups': data_parallel_groups
                }
        return dist_init._config

def find_group(groups):
    global_rank = xm.get_ordinal()
    for group in groups:
        if global_rank in group:
            return group, group.index(global_rank)


def get_model_parallel_rank():
    mp_groups = dist_init.model_parallel_group()
    return find_group(mp_groups)[1]

def get_model_parallel_world_size():
    mp_groups = dist_init.model_parallel_group()
    return len(find_group(mp_groups)[0])
