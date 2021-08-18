#include "backend/kernel_compiler/gpu/kungfu/kungfu_cluster_size_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"

namespace mindspore
{
namespace kernel
{
MS_REG_GPU_KERNEL_ONE(
    KungFuClusterSize,
    KernelAttr().AddOutputAttr(kNumberTypeInt32),
    KungFuClusterSizeGpuKernel, int32_t)
}  // namespace kernel
}  // namespace   mindspore
