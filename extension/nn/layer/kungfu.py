import mindspore as ms
import mindspore.ops.operations.kungfu_comm_ops as kfops
import numpy as np
from mindspore._checkparam import Validator as validator
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer


class KungfuBatchNorm(ms.nn.Cell):
    @cell_attr_register
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9,
                 affine=True,
                 gamma_init="ones",
                 beta_init="zeros",
                 moving_mean_init="zeros",
                 moving_var_init="ones",
                 input_dims="2d",
                 data_format="NCHW"):
        super().__init__()
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        if num_features < 1:
            raise ValueError("num_features must be at least 1")
        self.num_features = num_features
        if momentum < 0 or momentum > 1:
            error_msg = "momentum should be a number in range [0, 1], but got {}".format(momentum)
            raise ValueError(error_msg)
        self.momentum = 1.0 - momentum
        self.input_dims = input_dims
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if ms.context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.eps = eps
        self.moving_mean = ms.Parameter(initializer(
            moving_mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = ms.Parameter(initializer(
            moving_var_init, num_features), name="variance", requires_grad=False)
        self.gamma = ms.Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = ms.Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)

        #  self._cluster_size_op = kfops.KungFuClusterSize()
        self._all_reduce_op = kfops.KungFuAllReduce()
        self._square_op = ms.ops.Square()
        self._sqrt_op = ms.ops.Sqrt()

        # HACK
        self._cluster_size_op = kfops.KungFuClusterSizeInput()
        self._cluster_size_input = ms.Tensor(np.ones((1,), dtype=np.int32))

    def construct(self, x):
        # Assume x of shape (N, C, H, W)
        batch_size = x.shape[0]
        if self.training:
            # calculate global batch size
            #  cluster_size = self._cluster_size_op()
            cluster_size = self._cluster_size_op(self._cluster_size_input) # HACK
            global_batch_size = batch_size * cluster_size

            # mean along N
            sum_local_batch = x.sum(axis=0)
            sum_global_batch = self._all_reduce_op(sum_local_batch)
            mean_global_batch = sum_global_batch / global_batch_size

            # calculate expected value
            expected_value = mean_global_batch.mean(axis=[1, 2])

            # calculate variance
            local_squared = x.copy()
            for i in range(batch_size):
                for j in range(self.num_features):
                    local_squared[i, j] = self._square_op(x[i, j] - expected_value[j])
            sum_local_var = local_squared.sum(axis=0)
            sum_global_var = self._all_reduce_op(sum_local_var)
            mean_variance = sum_global_var / global_batch_size
            variance = mean_variance.mean(axis=[1, 2])

            # normalise input
            x_norm = x.copy()
            for i in range(batch_size):
                for j in range(self.num_features):
                    zero_mean = (x[i, j] - expected_value[j])
                    one_var = self._sqrt_op(variance[j] + self.eps)
                    norm = zero_mean / one_var
                    x_norm[i, j] = self.gamma[j] * norm + self.beta[j]

            self.moving_mean = ((1 - self.momentum) * self.moving_mean
                               + self.momentum * expected_value)
            self.moving_variance = ((1 - self.momentum) * self.moving_variance
                                   + self.momentum * variance)

            return x_norm

        # inference
        x_norm = x.copy()
        for i in range(batch_size):
            for j in range(self.num_features):
                zero_mean = (x[i, j] - self.moving_mean[j])
                one_var = self._sqrt_op(self.moving_variance[j] + self.eps)
                norm = zero_mean / one_var
                x_norm[i, j] = self.gamma[j] * norm + self.beta[j]

        return x_norm
