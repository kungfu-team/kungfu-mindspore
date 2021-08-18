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
    KungFuClusterSizeCpuKernel() = default;
    ~KungFuClusterSizeCpuKernel() override = default;

    void InitKernel(const CNodePtr &kernel_node) override
    {
        output_size_ = sizeof(T);
        output_size_list_.push_back(output_size_);
    }

    bool Launch(const std::vector<AddressPtr> &inputs,
                const std::vector<AddressPtr> &workspace,
                const std::vector<AddressPtr> &outputs) override
    {
        std::unique_ptr<kungfu::Peer> _kungfu_peer;

        int cluster_size = _kungfu_peer->Size();
        MS_LOG(WARNING) << "A";
        T cluster_size_T = (T) cluster_size;
        MS_LOG(WARNING) << "B";
        T *output_addr = reinterpret_cast<T*>(outputs.at(0)->addr);
        MS_LOG(WARNING) << "C";
        std::memcpy(output_addr, &cluster_size_T, sizeof(int));

        MS_LOG(WARNING) << "CPU CPU CPU";

        return true;
    }

  private:
    std::vector<size_t> output_size_list_;
    size_t output_size_;
};
}  // namespace kernel
}  // namespace   mindspore
