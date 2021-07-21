#include "backend/kernel_compiler/gpu/kungfu/kungfu_all_gather_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"

namespace mindspore
{
namespace kernel
{
MS_REG_GPU_KERNEL_ONE(KungFuAllGather,
                      KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddOutputAttr(kNumberTypeFloat32),
                      KungFuAllGatherGpuKernel, float)

MS_REG_GPU_KERNEL_ONE(
    KungFuAllGather,
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KungFuAllGatherGpuKernel, int32_t)
}  // namespace kernel
}  // namespace   mindspore
