"""kungfu comm_ops"""
import os

from ..._c_expression import (kungfu_current_cluster_size, kungfu_current_rank,
                              kungfu_finalize, kungfu_init,
                              kungfu_nccl_finalize, kungfu_nccl_init)
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register
from .comm_ops import ReduceOp


class KungFuAllReduce(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM):
        self.op = op
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x


class KungFuAllGather(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_dtype(self, x):
        return x

    def infer_shape(self, x):
        # print('KungFuAllGather::infer_shape %s' % (x))
        size = kungfu_current_cluster_size()
        dims = [size]
        dims.extend(x)
        # print('KungFuAllGather::infer_shape %s' % (dims))
        return dims


class KungFuBroadcast(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x


class KungFuResize(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['n'], outputs=['changed', 'detached'])

    def infer_shape(self, *args):
        return ([], [])

    def infer_dtype(self, *args):
        return (mstype.bool_, mstype.bool_)


def init(device):
    kungfu_init()
    if device == 'GPU':
        kungfu_nccl_init()


def finalize(device):
    if device == 'GPU':
        kungfu_nccl_finalize()
    kungfu_finalize()


class KungFuContext:
    def __init__(self, device='CPU'):
        if not device in ['CPU', 'GPU']:
            raise RuntimeError('invalid device %s' % (device))
        self._device = device

    def __enter__(self):
        init(self._device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        finalize(self._device)


class KungFuClusterSize(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=[], outputs=["size"])

    def infer_shape(self, *args):
        return []

    def infer_dtype(self, *args):
        return mstype.int32



from ..._c_expression import kungfu_debug_nccl


def using_kungfu():
    return bool(os.getenv('KUNGFU_SELF_SPEC'))


# debug ops


class KungFuLogTensor(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        return x
