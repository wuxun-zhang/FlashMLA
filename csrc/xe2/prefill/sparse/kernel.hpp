#pragma once

#include "../../../params.h"
#include "../../utils.hpp"

#include "cutlass/device_kernel.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

#include <cute/util/compat/device.hpp>
#include <cute/util/compat/dims.hpp>
#include <cute/util/compat/launch_policy.hpp>

// WA: avoid issue of not supported device-only copy
struct XPUSparseAttnFwdParams : public SparseAttnFwdParams {
    sycl::queue queue;
};

namespace xe2::fwd {

using namespace cute;

constexpr int Subgroup_Size = 16;
constexpr int B_H = 32;       // h_q block size
constexpr int B_TOPK = 64;    // topk_length block size

template<int D_QK, bool HAVE_TOPK_LENGTH>
class KernelTemplate {
public:
    using Params = SparseAttnFwdParams;
    struct SharedStorage {
        // This is just a placeholder. The actual shared memory declarations will be here.
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        // This is just a placeholder. The actual kernel implementation will be here.
    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
void launch_kernel(const XPUSparseAttnFwdParams& params) {
    KU_ASSERT(params.h_kv == 1, "h_kv must be 1");
    KU_ASSERT(params.h_q % B_H == 0, "h_q must be divisible by ", B_H);
    KU_ASSERT(params.topk % B_TOPK == 0, "topk must be divisible by ", B_TOPK);
    KU_ASSERT(params.d_qk == D_QK, "Invalid d_qk for this kernel instantiation");

    dim3 block(256, 1, 1);
    dim3 grid((params.h_q / B_H) * params.s_q, 1, 1);

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    using Kernel = KernelTemplate<D_QK, HAVE_TOPK_LENGTH>;
    const int smem_size = Kernel::SharedStorageSize;

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    auto event = launch<cutlass::device_kernel<Kernel>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<Subgroup_Size>}},
        params.queue, static_cast<SparseAttnFwdParams>(params));
#else
    compat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<Subgroup_Size>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel>(policy, params.queue, static_cast<SparseAttnFwdParams>(params));
#endif

}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_kernel_impl(const XPUSparseAttnFwdParams& params) {
    launch_kernel<D_QK, HAVE_TOPK_LENGTH>(params);
}

} // namespace xe2::fwd
