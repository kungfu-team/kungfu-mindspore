import argparse

import mindspore as ms
import numpy as np
from mindspore._c_expression import (kungfu_current_rank, kungfu_finalize,
                                     kungfu_current_cluster_size, kungfu_init,
                                     kungfu_nccl_finalize, kungfu_nccl_init)
from mindspore.ops.operations.kungfu_comm_ops import KungFuSubsetAllReduce

dtype_map = {
    'i32': np.int32,
    'f32': np.float32,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='CPU', choices=['GPU', 'CPU'])
    p.add_argument('--dtype', type=str, default='f32', choices=['i32', 'f32'])
    return p.parse_args()


def main():
    args = parse_args()
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target=args.device)

    kungfu_init()
    if args.device == 'GPU':
        kungfu_nccl_init()

    if kungfu_current_cluster_size() != 4:
        print('must test with 4 workers!')
        exit(1)

    # topology is a forest
    # topology must an int32 tensor of shape [n], where n is the cluster size
    # topology[i] represents the parent node of i
    # topology[i] == i means i is a root of one tree in the forest
    subset_allreduce = KungFuSubsetAllReduce()

    size = 10
    rank = kungfu_current_rank()
    value = 1 << rank
    dtype = dtype_map[args.dtype]

    x = ms.Tensor(np.array([value] * size).astype(dtype))
    print('x=%s' % (x))

    topology = ms.Tensor([0, 1, 2, 3], dtype=ms.int32)  # 4 trees
    y = subset_allreduce(x, topology)
    print('y=%s' % (y))

    topology = ms.Tensor([0, 0, 2, 2], dtype=ms.int32)  # 2 trees
    y = subset_allreduce(x, topology)
    print('y=%s' % (y))

    topology = ms.Tensor([0, 0, 1, 1], dtype=ms.int32)  # 1 tree
    y = subset_allreduce(x, topology)
    print('y=%s' % (y))

    if args.device == 'GPU':
        kungfu_nccl_finalize()
    kungfu_finalize()


main()
