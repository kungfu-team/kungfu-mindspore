#pragma once

#include <vector>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/kungfu/kungfu_common.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore
{
namespace kernel
{
template <typename T>
class KungFuClusterSizeCpuKernel : public CPUKernel
{
  public:
    KungFuClusterSizeCpuKernel()
    {
    }
    ~KungFuClusterSizeCpuKernel() override = default;

    void InitKernel(const CNodePtr &kernel_node) override
    {
        LOG_InitKernel("KungFuClusterSizeCPUKernel");
    }

    bool Launch(const std::vector<AddressPtr> &inputs,
                const std::vector<AddressPtr> &workspace,
                const std::vector<AddressPtr> &outputs) override
    {
        std::unique_ptr<kungfu::Peer> _kungfu_peer;

        // int cluster_size = _kungfu_peer->Size();
        // MS_LOG(WARNING) << "Cluster size " << cluster_size;
        // const void *output_addr = outputs.at(0)->addr;
        // memcpy(output_addr, &cluster_size, sizeof(int)); // FIXME: not working

        return true;
    }
};
}  // namespace kernel
}  // namespace   mindspore
