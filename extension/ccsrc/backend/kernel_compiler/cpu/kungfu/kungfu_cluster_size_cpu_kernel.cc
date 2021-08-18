#include "backend/kernel_compiler/cpu/kungfu/kungfu_cluster_size_cpu_kernel.h"

namespace mindspore
{
namespace kernel
{
MS_REG_CPU_KERNEL_T(
    KungFuClusterSize,
    KernelAttr().AddOutputAttr(kNumberTypeInt32),
    KungFuClusterSizeCpuKernel, int32_t)
}  // namespace kernel
}  // namespace   mindspore
