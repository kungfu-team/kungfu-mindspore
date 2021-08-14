#pragma once
#include <map>
#include <string>
#include <vector>

#include "backend/kernel_compiler/cpu/kungfu/kungfu_common.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"

namespace mindspore
{
namespace kernel
{
template <typename T>
class KungFuClusterSizeGpuKernel : public GpuKernel
{
  public:
    KungFuClusterSizeGpuKernel()
          input_size_(0),
          output_size_(0),
          workspace_size_(0)
    {
    }

    ~KungFuClusterSizeGpuKernel() override = default;

    const std::vector<size_t> &GetInputSizeList() const override
    {
        return input_size_list_;
    }

    const std::vector<size_t> &GetOutputSizeList() const override
    {
        return output_size_list_;
    }

    const std::vector<size_t> &GetWorkspaceSizeList() const override
    {
        return workspace_size_list_;
    }

    bool Launch(const std::vector<AddressPtr> &inputs,
                const std::vector<AddressPtr> &workspace,
                const std::vector<AddressPtr> &outputs,
                void *stream_ptr) override
    {
        std::unique_ptr<kungfu::Peer> _kungfu_peer;

        int cluster_size = _kungfu_peer->Size();
        T *output_addr = GetDeviceAddress<T>(outputs, 0);
        *output_addr = cluster_size;

        return true;
    }

    bool Init(const CNodePtr &kernel_node) override
    {
        InitSizeLists();

        return true;
    }

  protected:
    void InitSizeLists() override
    {
        input_size_list_.push_back(input_size_);
        output_size_list_.push_back(output_size_);
        return;
    }

  private:
    std::vector<size_t> input_size_list_;
    std::vector<size_t> output_size_list_;
    std::vector<size_t> workspace_size_list_;
    size_t input_size_;
    size_t output_size_;
    size_t workspace_size_;
};
}  // namespace kernel
}  // namespace   mindspore
