#include "../kernel.hpp"

namespace xe2::fwd {

// NOTE (intlsy): We instantiate run_fwd_phase1_kernel in two .cu files as functions with HAVE_TOPK_LENGTH
// = true / false respectively, to compile them in parallel.
template void run_fwd_kernel_impl<576, false>(const XPUSparseAttnFwdParams& params);

}
