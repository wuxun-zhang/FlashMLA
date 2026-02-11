#include "fwd.hpp"
#include "kernel.hpp"

// for kernel template instantiations

namespace xe2 {

void run_fwd_kernel(const SparseAttnFwdParams& params) {
    const bool have_topk_length = params.topk_length != nullptr;

    // Dispatch based on d_qk dimension and presence of topk_length
    if (params.d_qk == 512) {
        if (have_topk_length) {
            xe2::fwd::run_fwd_kernel_impl<512, true>(params);
        } else {
            xe2::fwd::run_fwd_kernel_impl<512, false>(params);
        }
    } else if (params.d_qk == 576) {
        if (have_topk_length) {
            xe2::fwd::run_fwd_kernel_impl<576, true>(params);
        } else {
            xe2::fwd::run_fwd_kernel_impl<576, false>(params);
        }
    } else {
        throw std::runtime_error("Unsupported d_qk value in sparse attention fwd kernel");
    }
}
} // namespace xe2
