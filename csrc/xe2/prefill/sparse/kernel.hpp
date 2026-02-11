#pragma once

#include "../../../params.h"

namespace xe2::fwd {

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_kernel_impl(const SparseAttnFwdParams& params) {
    // KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(params);
}

} // namespace xe2::fwd
