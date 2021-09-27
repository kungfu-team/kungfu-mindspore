import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations.kungfu_comm_ops import KungFuAllReduce, KungFuClusterSize

cast = P.Cast()

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(cast(scale, mstype.float32))


class KungFuMomentum(ms.nn.Momentum):
    def __init__(self, *args, **kwargs):
        super(KungFuMomentum, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.all_reduce = KungFuAllReduce()

        # FIXME: make it dynamic
        cluster_size = float(kfops.kungfu_current_cluster_size())
        scalar_shape = []
        self.cluster_size = ms.Parameter(
            ms.Tensor(
                np.ones(scalar_shape) * cluster_size,
                ms.int32,
            ))

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        mean_grads = self.hyper_map(F.partial(grad_scale, self.cluster_size),
                                    gradients)
        return super(KungFuMomentum, self).construct(mean_grads)


class KungFuLamb(ms.nn.Lamb):
    def __init__(self, *args, **kwargs):
        super(KungFuLamb, self).__init__(*args, **kwargs)
        self.map_ = C.Map()
        self.all_reduce = KungFuAllReduce()

        # FIXME: make it dynamic
        cluster_size = float(kfops.kungfu_current_cluster_size())
        scalar_shape = []
        self.cluster_size = ms.Parameter(
            ms.Tensor(
                np.ones(scalar_shape) * cluster_size,
                ms.int32,
            ))

    def construct(self, gradients):
        gradients = self.map_(self.all_reduce, gradients)
        mean_grads = self.hyper_map(F.partial(grad_scale, self.cluster_size),
                                    gradients)
        return super(KungFuLamb, self).construct(mean_grads)


class ModOp(ms.nn.Cell):
    def __init__(self):
        super(ModOp, self).__init__()
        self.div_op = ms.ops.operations.math_ops.Div()

    def construct(self, x, y):
        q = self.div_op(x, y)
        r = x - q * y
        return r


add_grads = C.MultitypeFuncGraph("add_grads")


@add_grads.register("Tensor", "Tensor")
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, mstype.float32)


update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, cast(grad, mstype.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


class CumulativeLambOptimizer(ms.nn.Lamb):
    def __init__(self, parameters, *args, **kwargs):
        print('%s::%s' % ('CumulativeLambOptimizer', '__init__'))
        if 'apply_period' in kwargs:
            apply_period = kwargs['apply_period']
            del kwargs['apply_period']
        else:
            apply_period = 1  # FIXME: make it a Parameter

        super(CumulativeLambOptimizer, self).__init__(parameters, *args,
                                                      **kwargs)

        self.weights = self.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')

        print('%d grads in this model' % (len(self.accu_grads)))

        self.mod_op = ModOp()

        scalar_shape = []
        self.acc_step = ms.Parameter(
            ms.Tensor(
                np.zeros(scalar_shape),
                ms.int32,
            ))
        self.apply_period = ms.Parameter(
            ms.Tensor(
                np.ones(scalar_shape) * apply_period,
                ms.int32,
            ))

    @ms.ops.composite.add_flags(has_effect=True)
    def construct(self, gradients):
        # TODO: perform all_reduce
        #     gradients = self._map(self._all_reduce, gradients)
        self.acc_step = self.acc_step + 1
        q = self.mod_op(self.acc_step, self.apply_period)

        accu_grads = self.hyper_map(add_grads, self.accu_grads, gradients)
        accu_succ = self.hyper_map(update_accu_grads, self.accu_grads,
                                   accu_grads)

        if q == 0:
            mean_grads = self.hyper_map(
                F.partial(grad_scale, self.apply_period), accu_grads)
            apply_succ = super(CumulativeLambOptimizer,
                               self).construct(mean_grads)

            reset_succ = self.hyper_map(reset_accu_grads, self.accu_grads)

            succ = F.depend(reset_succ, apply_succ)
        else:
            succ = True
            succ = F.depend(succ, accu_succ)

        return F.depend(gradients, succ)

    def debug(self):
        print('BEGIN: %s::debug' % (self.__class__.__name__))

        acc_step = self.acc_step.asnumpy()
        print('acc_step: %s' % (acc_step))

        apply_period = self.apply_period.asnumpy()
        print('apply_period: %s' % (apply_period))

        accu_grads = [g.asnumpy() for g in self.accu_grads]
        for i, g in enumerate(accu_grads):
            print('accu_grads[%d]: %s' % (i, g))

        print('END: %s::debug' % (self.__class__.__name__))
