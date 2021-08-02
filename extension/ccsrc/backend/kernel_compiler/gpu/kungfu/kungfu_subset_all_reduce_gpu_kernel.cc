#include "backend/kernel_compiler/gpu/kungfu/kungfu_subset_all_reduce_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"

namespace mindspore
{
namespace kernel
{
MS_REG_GPU_KERNEL_ONE(KungFuSubsetAllReduce,
                      KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt32)  // the topology
                          .AddOutputAttr(kNumberTypeFloat32),
                      KungFuSubsetAllReduceGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(KungFuSubsetAllReduce,
                      KernelAttr()
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)  // the topology
                          .AddOutputAttr(kNumberTypeInt32),
                      KungFuSubsetAllReduceGpuKernel, int32_t)
}  // namespace kernel
}  // namespace   mindspore
