#include <kungfu/nccl/helper.hpp>

#include "backend/kernel_compiler/cpu/kungfu/kungfu_common.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "mindspore/core/utils/log_adapter.h"
#include "pybind_api/api_register.h"

static std::string safe_getenv(const char *name)
{
    const char *ptr = std::getenv(name);
    if (ptr) {
        return std::string(ptr);
    }
    return "";
}

static bool parse_env_bool(const char *name)
{
    auto v = safe_getenv(name);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](char c) { return std::tolower(c); });
    if (v == "true" || v == "on") {
        return true;
    }
    if (v == "1") {
        return true;
    }
    return false;
}

std::unique_ptr<kungfu::NCCLHelper> _kungfu_nccl_helper;

namespace mindspore
{
namespace kernel
{
bool _kungfu_use_nccl_scheduler()
{
    bool f = parse_env_bool("KUNGFU_USE_NCCL_SCHEDULER");
    // std::cerr << std::boolalpha << "KUNGFU_USE_NCCL_SCHEDULER=" << f
    //           << std::endl;
    return f;
}

bool kungfu_use_nccl_scheduler = _kungfu_use_nccl_scheduler();

void kungfu_nccl_init()
{
    log_func_call(__func__);
    // kungfu_show_cuda_version();
    // kungfu_show_nccl_version();
    const auto nccl_scope = KungFu_NCCL_GLOBAL;
    _kungfu_nccl_helper.reset(new kungfu::NCCLHelper);
    auto nccl_scheduler = _kungfu_nccl_helper->EnsureScheduler(nccl_scope);
    auto nccl_controller = _kungfu_nccl_helper->EnsureController(nccl_scope);
    kungfu::Peer *peer = _kungfu_peer.get();
    fprintf(stderr, "using _kungfu_peer=%p\n", peer);
    if (kungfu_use_nccl_scheduler) {
        nccl_scheduler->Do(
            [&] { KF_LOG_CALL(nccl_controller->InitOnce(peer)); });
    } else {
        nccl_controller->InitOnce(peer);
    }
}

void kungfu_nccl_finalize()
{
    log_func_call(__func__);
    _kungfu_nccl_helper.reset(nullptr);
}

REGISTER_PYBIND_DEFINE(KungFuNccl, ([](py::module *m) {
                           m->def("kungfu_nccl_init", &kungfu_nccl_init);
                           m->def("kungfu_nccl_finalize",
                                  &kungfu_nccl_finalize);
                       }));
}  // namespace kernel
}  // namespace   mindspore
