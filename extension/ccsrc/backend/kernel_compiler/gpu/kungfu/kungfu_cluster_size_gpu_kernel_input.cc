#include "backend/kernel_compiler/gpu/kungfu/kungfu_cluster_size_gpu_kernel_input.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"

namespace mindspore
{
namespace kernel
{
MS_REG_GPU_KERNEL_ONE(
    KungFuClusterSizeInput,
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KungFuClusterSizeInputGpuKernel, int32_t)
}  // namespace kernel
}  // namespace   mindspore
