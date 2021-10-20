This repository has experimental code for running Megatron op sharding Model Parallelism.
The code is derived from: https://github.com/NVIDIA/Megatron-LM

Currently only ColumnParallelLinear and RowParallelLinear are tested.

For examples see:
parallel_transformer.py
spmd-experiments.py

With these two layers, MP implementation similar to https://arxiv.org/pdf/1909.08053.pdf is feasible.

Use mpu layers, please install the package:

`sudo -H pip install -e .`

mpu_ directory is derived from more recent megatron mpu code.
However it sill has some unresolved problems.

Please use: xla_add.mpu and ignore mpu_
