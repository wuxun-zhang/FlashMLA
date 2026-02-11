#pragma once

#include "../../../params.h"

namespace xe2::fwd {

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_kernel(const SparseAttnFwdParams& params);

} // namespace xe2::fwd
