#pragma once

#include "../../../params.h"
#include "../../utils.hpp"

#include "cutlass/device_kernel.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

#include <cute/util/compat/device.hpp>
#include <cute/util/compat/dims.hpp>
#include <cute/util/compat/launch_policy.hpp>

#include "mainloop.hpp"
#include "epilogue.hpp"
#include "tile_scheduler.hpp"

// WA: avoid issue of not supported device-only copy
struct XPUSparseAttnFwdParams : public SparseAttnFwdParams {
    sycl::queue queue;
};

namespace xe2::fwd {

using namespace cute;

struct KernelConfig {
    using ElementQ = cutlass::bfloat16_t;
    using ElementK = cutlass::bfloat16_t;
    using ElementV = cutlass::bfloat16_t;
    using ElementO = cutlass::bfloat16_t;

    using StrideQ = cute::tuple<int, 1, int, int>;
    using StrideKV = cute::tuple<int, 1, int, int>;
    using StrideO = cute::tuple<int, 1, int, int>;

    static constexpr int NUM_THREADS = 256;
    static constexpr int SUBGROUP_SIZE = 16;
    static constexpr int B_H = 8;       // h_q block size
    static constexpr int B_TOPK = 64;    // topk_length block size

    // 576 / 32 = 18
    // Q head packing size = B_H
    using TileShapeQK = Shape<B_H, 256, 32>;
    using SubgroupLayoutQK = Shape<1, 16, 1>;

    using TileShapePV = Shape<B_H, 32, 256>;
    using SubgroupLayoutPV = decltype(xe2::fwd::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

    using TileShapeOut = Shape<B_H, _128>;
};

template<int D_QK, bool HAVE_TOPK_LENGTH, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class KernelTemplate {
public:
    //
    // Type Aliases
    //
    using CollectiveMainloop = CollectiveMainloop_;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;

    using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
    using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
    using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
    using TileShapePV = typename CollectiveMainloop::TileShapePV;
    using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
    using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
    using ElementK = typename CollectiveMainloop::TensorK::element_type;
    using ElementV = typename CollectiveMainloop::TensorV::element_type;

    using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
    using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
    using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

    using SGPerWG = typename CollectiveMainloop::SGPerWG;

    using FragA = typename CollectiveMainloop::FragA;
    using FragARow = typename CollectiveMainloop::FragARow;

    // Tile scheduler derived types
    using TileScheduler = TileScheduler_;
    using TileSchedulerParams = typename TileScheduler::Params;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    using TileShapeO = typename CollectiveEpilogue::TileShapeO;
    using ElementO = typename CollectiveEpilogue::TensorO::element_type;
    using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

    // dtype for storing intermediate exp sums and max logits
    using ElementLSE = typename CollectiveEpilogue::ElementLSE;

    // Kernel level shared memory storage
    using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
    using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;

    union SharedStorage {
        MainloopSharedStorage mainloop;
        EpilogueSharedStorage epilogue;
    };

    static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                        : sizeof(SharedStorage);

    using KernelArguments = SparseAttnFwdParams;
    using KernelParams = KernelArguments;

    struct Arguments {
        KernelArguments kernel{};
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        KernelHardwareInfo hw_info{};
    };

    // Kernel entry point API
    struct Params {
        KernelParams kernel;
        MainloopParams mainloop;
        EpilogueParams epilogue;
        TileSchedulerParams scheduler;
    };

    //
    // Methods
    //

    static Params to_underlying_arguments(Arguments const &args, void *workspace) {
        return {args.kernel,
                CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
                CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
                TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{})};
    }

    // FIXME(wuxun): move to python interface to raise error early
    static bool can_implement(Arguments const &args) {
        return CollectiveMainloop::can_implement(args.mainloop)
            && CollectiveEpilogue::can_implement(args.epilogue);
    }

    static int get_workspace_size(Arguments const &args) { return 0; }

    static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                                cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
        return Status::kSuccess;
    }

    static dim3 get_grid_shape(Params const &params) {
        return TileScheduler::get_grid_shape(params.scheduler);
    }

    static dim3 get_block_shape() { return dim3(KernelConfig::NUM_THREADS, 1, 1); }

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        using namespace sycl::ext::oneapi::this_work_item;
        int thr_id = int(ThreadIdxX());
        int wg_id = int(BlockIdxX());

        int sg_id = thr_id / SUBGROUP_SIZE;
        int tid_in_sg = thr_id % SUBGROUP_SIZE;

        int num_heads_blocks = params.h_q / B_H;
        int seq_idx = wg_id / num_heads_blocks;
        int head_bid = wg_id % num_heads_blocks;
        // start idx of current head block
        int cur_head_idx = head_bid * B_H;

    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
void launch_kernel(const XPUSparseAttnFwdParams& params) {
    KU_ASSERT(params.h_kv == 1, "h_kv must be 1");
    KU_ASSERT(params.h_q % B_H == 0, "h_q must be divisible by ", B_H);
    KU_ASSERT(params.topk % B_TOPK == 0, "topk must be divisible by ", B_TOPK);
    KU_ASSERT(params.d_qk == D_QK, "Invalid d_qk for this kernel instantiation");

    // create kernel
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    constexpr int SGTileQ = get<0>(shape_div(KernelConfig::TileShapeQK{}, shape(KernelConfig::SubgroupLayoutQK{})))();
    // bf16 dpas m8n16k16
    // (8, 256, 32) / ((8, 16, 16) * (1, 16, 1)) = (1, 1, 2) iterations per subgroup
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, bfloat16_t>;
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<KernelConfig::TileShapeQK>, KernelConfig::SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<KernelConfig::TileShapePV>, KernelConfig::SubgroupLayoutPV>::TiledMMA;

    constexpr int VTiles = get<1>(KernelConfig::TileShapeOutput{}) / get<1>(KernelConfig::TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val),
                         make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(KernelConfig::ElementQ{}, KernelConfig::StrideQ{}));
    using TensorKV = decltype(make_dummy_tensor(KernelConfig::ElementK{}, KernelConfig::StrideKV{}));
    using TensorO = decltype(make_dummy_tensor(KernelConfig::ElementO{}, KernelConfig::StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = xe2::fwd::XeDefault<PipelineStages>;
    using CollectiveMainloop = xe2::fwd::collective::Xe2FwdMainloop<
        MainloopDispatchPolicy, Causal, CachedKV, PagedKV,
        TiledMMAQK, TiledMMAPV, VTiles,
        TensorQ, TensorKV, TensorKV,
        // TensorK_cache, TensorV_cache,
        GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV,
        // GmemTiledCopyK_cache, GmemTiledCopyV_cache
    >;

    // Epilogue
    using CollectiveEpilogue = xe2::fwd::collective::Xe2FwdEpilogue<
        CollectiveMainloop,
        KernelConfig::TileShapeOutput,
        TensorO,
        void
    >;

    using Kernel = KernelTemplate<D_QK, HAVE_TOPK_LENGTH,
        CollectiveMainloop, CollectiveEpilogue, xe2::fwd::XeFHMAIndividualTileScheduler>;

    dim3 block(KernelConfig::NUM_THREADS, 1, 1);
    dim3 grid((params.h_q / KernelConfig::B_H) * params.s_q, 1, 1);

    Kernel::Params kernel_params {
        params,
        {params.sm_scale, nullptr, 0, nullptr},
        {},
        hw_info
    };

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    const int smem_size = Kernel::SharedStorageSize;

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    auto event = launch<cutlass::device_kernel<Kernel>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<KernelConfig::SUBGROUP_SIZE>}},
        params.queue, kernel_params);
#else
    compat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<KernelConfig::SUBGROUP_SIZE>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel>(policy, params.queue, kernel_params);
#endif

}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_kernel_impl(const XPUSparseAttnFwdParams& params) {
    launch_kernel<D_QK, HAVE_TOPK_LENGTH>(params);
}

} // namespace xe2::fwd
