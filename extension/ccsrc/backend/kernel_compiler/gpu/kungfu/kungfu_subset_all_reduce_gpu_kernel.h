#pragma once
#include <map>
#include <sstream>
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
class KungFuSubsetAllReduceGpuKernel : public GpuKernel
{
    // constexpr static auto kAttrTopology = "topology";

  public:
    KungFuSubsetAllReduceGpuKernel()
        : nccl_controller_(nullptr),
          nccl_scheduler_(nullptr),
          reduce_op_(KungFu_SUM),
          comm_stream_(nullptr),
          input_count_(0),
          topology_count_(0),
          output_count_(0),
          input_size_(0),
          topology_size_(0),
          output_size_(0),
          workspace_size_(0)
    {
    }

    ~KungFuSubsetAllReduceGpuKernel() override
    {
        DestroyResource();
    }

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
        LOG_Kernel_Launch("KungFuAllReduceGpuKernel", inputs, workspace,
                          outputs);
        KUNGFU_PROFILE_SITE(KungFuAllReduceGpuKernel::Launch);

        const T *input_addr = GetDeviceAddress<T>(inputs, 0);
        const int32_t *topology_addr = GetDeviceAddress<int32_t>(inputs, 1);
        T *output_addr = GetDeviceAddress<T>(outputs, 0);

        cudaStream_t stream = comm_stream_
                                  ? comm_stream_
                                  : reinterpret_cast<cudaStream_t>(stream_ptr);
        // MS_LOG(WARNING) << "using stream " << stream;

        std::vector<int32_t> topology(topology_count_);
        auto result = cudaMemcpy(topology.data(), topology_addr, topology_size_,
                                 cudaMemcpyDeviceToHost);  // TODO: check result
        if (result != cudaSuccess) {
            MS_LOG(ERROR) << "cudaStreamSynchronize failed";
        }
        fprintf(stderr, "topology copied to CPU\n");

        auto w = make_kungfu_workspace(input_addr, output_addr, input_count_);

        auto controller = _kungfu_nccl_helper->EnsureGroupController(topology);
        controller->InitOnce(_kungfu_peer.get());

        fprintf(stderr, "NCCL controller created\n");

        // TODO: support async
        if (kungfu_use_nccl_scheduler) {
            nccl_scheduler_->Do([=] {
                KF_LOG_CALL(controller->AllReduce(w, reduce_op_, stream));
            });
        } else {
            controller->AllReduce(w, reduce_op_, stream);
        }
        return true;
    }

    bool Init(const CNodePtr &kernel_node) override
    {
        LOG_InitKernel("KungFuAllReduceGpuKernel");
        KUNGFU_PROFILE_SITE(KungFuAllReduceGpuKernel::Init);

        InitOp(kernel_node);

        InitResource();
        data_type_ = GetCudnnDataType(
            TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
        size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
        if (input_num != 2) {
            MS_LOG(ERROR) << "Input number is " << input_num
                          << ", but requires 2 input.";
            return false;
        }
        size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
        if (output_num != 1) {
            MS_LOG(ERROR) << "Output number is " << output_num
                          << ", but requires needs 1 output.";
            return false;
        }

        auto inputA_shape =
            AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
        auto inputB_shape =
            AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
        auto outputC_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);

        InferInAndOutDesc(inputA_shape, inputB_shape, outputC_shape);

        InitSizeLists();

        auto comm_stream_attr =
            AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stream_id");
        if (comm_stream_attr) {
            comm_stream_ = reinterpret_cast<cudaStream_t>(
                GetValue<uintptr_t>(comm_stream_attr));
            MS_EXCEPTION_IF_NULL(comm_stream_);
            MS_LOG(WARNING) << "got kernel_node stream_id: " << comm_stream_;
        }

        return true;
    }

  protected:
    void InitResource() override
    {
        const auto nccl_scope_ = KungFu_NCCL_GLOBAL;
        nccl_scheduler_ = _kungfu_nccl_helper->EnsureScheduler(nccl_scope_);
        nccl_controller_ = _kungfu_nccl_helper->EnsureController(nccl_scope_);
    }

    void InitSizeLists() override
    {
        input_size_list_.push_back(input_size_);
        input_size_list_.push_back(topology_size_);

        output_size_list_.push_back(output_size_);
        return;
    }

  private:
    void InitOp(const CNodePtr &kernel_node)
    {
        auto reduce_op =
            AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(kAttrOp);
        if (reduce_op) {
            std::string op_name = GetValue<std::string>(reduce_op);
            static std::map<std::string, KungFu_Op> _name_to_op({
                {"sum", KungFu_SUM},
                {"min", KungFu_MIN},
                {"max", KungFu_MAX},
                {"prod", KungFu_PROD},
            });
            auto it = _name_to_op.find(op_name);
            if (it == _name_to_op.end()) {
                MS_LOG(EXCEPTION)
                    << "reduce op " << op_name << " is not supported.";
            } else {
                reduce_op_ = it->second;
            }
        }
    }

    // void InitTopology(const CNodePtr &kernel_node)
    // {
    //     auto topologyAttr =
    //         AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(kAttrTopology);
    //     auto topologyStr = GetValue<std::string>(topologyAttr);
    //     std::stringstream ss(topologyStr);
    //     for (std::string s; std::getline(ss, s, '|');) {
    //         topology_.push_back(std::stoi(s));
    //     }
    //     fprintf(stderr,
    //             "KungFuSubsetAllReduceGpuKernel using topology: [%d]{...}\n",
    //             (int)topology_.size());
    // }

    void DestroyResource() noexcept
    {
    }

    void InferInAndOutDesc(const std::vector<size_t> &input_shape,
                           const std::vector<size_t> &topology_shape,
                           const std::vector<size_t> &output_shape)
    {
        input_count_ = std::accumulate(input_shape.begin(), input_shape.end(),
                                       1, std::multiplies<size_t>());
        topology_count_ =
            std::accumulate(topology_shape.begin(), topology_shape.end(), 1,
                            std::multiplies<size_t>());
        output_count_ =
            std::accumulate(output_shape.begin(), output_shape.end(), 1,
                            std::multiplies<size_t>());
        input_size_ = input_count_ * sizeof(T);
        topology_size_ = topology_count_ * sizeof(int32_t);
        output_size_ = output_count_ * sizeof(T);
    }

    kungfu::NCCLController *nccl_controller_;
    kungfu::NCCLScheduler *nccl_scheduler_;

    KungFu_Op reduce_op_;
    // std::vector<int32_t> topology_;

    cudaStream_t comm_stream_;
    cudnnDataType_t data_type_;
    std::string group_name_;

    size_t input_count_;
    size_t topology_count_;
    size_t output_count_;

    std::vector<size_t> input_size_list_;
    std::vector<size_t> output_size_list_;
    std::vector<size_t> workspace_size_list_;

    size_t input_size_;
    size_t topology_size_;
    size_t output_size_;
    size_t workspace_size_;
};
}  // namespace kernel
}  // namespace   mindspore
