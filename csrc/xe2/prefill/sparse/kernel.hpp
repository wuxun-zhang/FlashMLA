#pragma once

#include "../../../params.h"
#include "../../utils.hpp"

#include "cutlass/device_kernel.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>

#include <cute/util/compat/device.hpp>
#include <cute/util/compat/dims.hpp>
#include <cute/util/compat/launch_policy.hpp>

// #include "mainloop.hpp"
// #include "epilogue.hpp"
// #include "tile_scheduler.hpp"

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

    using StrideQ = cute::tuple<int, _1, int>;
    using StrideKV = cute::tuple<int, _1, int>;
    // using StrideV = cute::tuple<1, int, int>;
    using StrideO = cute::tuple<int, _1, int>;

    static constexpr int NUM_THREADS = 256;
    static constexpr int SUBGROUP_SIZE = 16;
    static constexpr int NUM_SUBGROUPS = NUM_THREADS / SUBGROUP_SIZE;
    static constexpr int B_H = 8;       // h_q block size
    static constexpr int B_TOPK = 64;    // topk_length block size

    static constexpr int D_QK = 576;
    static constexpr int D_PE = 64;
    static constexpr int D_V = 512;

    static constexpr int stages = 1;

    // 576 / 32 = 18
    // Q head packing size = B_H
    using TileShapeQK = Shape<Int<B_H>, Int<4*B_TOPK>, _64>;
    using SubgroupLayoutQK = Layout<Shape<_1, _16, _1>>;

    using TileShapePV = Shape<Int<B_H>, _64, Int<4*B_TOPK>>;
    // using SubgroupLayoutPV = decltype(xe2::fwd::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
    using SubgroupLayoutPV = Layout<Shape<_1, _1, _16>>;

    using TileShapeOut = Shape<Int<B_H>, Int<D_V>>;

    using SmemLayoutK = Layout<Shape<Int<B_TOPK>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;

    constexpr static int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    // bf16 dpas m8n16k16
    // (8, 256, 32) / ((8, 16, 16) * (1, 16, 1)) = (1, 1, 2) iterations per subgroup
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, bfloat16_t>;
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

};

template<int D_QK, bool HAVE_TOPK_LENGTH>
class KernelTemplate_1 {
public:
    struct SharedStorage {
        cute::array_aligned<cutlass::bfloat16_t, D_QK * KernelConfig::B_TOPK> k[KernelConfig::stages];
    };

    static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                        : sizeof(SharedStorage);
    using Params = SparseAttnFwdParams;

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        using namespace sycl::ext::oneapi::this_work_item;

        SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

        const KernelConfig::ElementQ* q = params.q;
        const KernelConfig::ElementK* kv = params.kv;
        const float* attn_sink = params.attn_sink;
        const int* topk_length = params.topk_length;
        KernelConfig::ElementO* out = params.out;
        float* max_logits = params.max_logits;
        float* lse = params.lse;

        int thr_id = int(ThreadIdxX());
        int wg_id = int(BlockIdxX());

        int sg_id = thr_id / KernelConfig::SUBGROUP_SIZE;
        int tid_in_sg = thr_id % KernelConfig::SUBGROUP_SIZE;

        int num_head_blocks = params.h_q / KernelConfig::B_H;
        int seq_idx = wg_id / num_head_blocks;
        int head_bid = wg_id % num_head_blocks;
        // start idx of current head block
        int cur_head_start_idx = head_bid * KernelConfig::B_H;

        using LayoutQ = Layout<Shape<Int<KernelConfig::B_H>, Int<D_QK>>, Stride<Int<KernelConfig::D_QK>, _1>>;
        using LayoutO = Layout<Shape<Int<KernelConfig::B_H>, Int<KernelConfig::D_V>>, Stride<Int<KernelConfig::D_V>, _1>>;

        Layout layout_K = make_layout(make_shape(params.s_kv, params.d_qk), make_stride(params.stride_kv_s_kv, _1{}));

        Tensor Q = make_tensor(make_gmem_ptr(q + cur_head_start_idx * D_QK), LayoutQ{});
        Tensor K = make_tensor(make_gmem_ptr(kv), layout_K);
        // Tensor V = make_tensor(make_gmem_ptr(kv), layout_V);
        Tensor O = make_tensor(make_gmem_ptr(out + cur_head_start_idx * KernelConfig::D_V), LayoutO{});

        Tensor proxyQ = make_identity_tensor(Q.shape());
        Tensor proxyK = make_identity_tensor(make_shape(params.topk, Int<D_QK>{}));
        // Tensor proxyV = make_identity_tensor(V.shape());
        Tensor proxyP = make_identity_tensor(take<0,2>(KernelConfig::TileShapeQK{})); // (h,k)
        Tensor proxyO = make_identity_tensor(O.shape());

        // FIXME: correct coord
        Tensor gQ = local_tile(proxyQ, select<0,2>(KernelConfig::TileShapeQK{}), make_coord(cur_head_start_idx,_)); // (h,d,D)
        Tensor sK = local_tile(proxyK, KernelConfig::SmemLayoutK{}, make_coord(0,_)); // (k,d,D)

        static_assert(is_same_v<decltype(sK), float>, "dtype mixmatch");

        KernelConfig::TiledMMAQK mma_qk{};
        KernelConfig::TiledMMAPV mma_pv{};

        // // Shared memory buffers
        // Layout K_slm_layout = make_layout(append<3>(typename decltype(r2s_K)::Tiler_MN{}, Int<Stages>{}));
        // [B_TOPK, D_QK]
        auto K_slm = make_tensor(make_smem_ptr(shared_storage.k[0].data()), KernelConfig::SmemLayoutK{});

        // create tiled copy for loading Q from gmem
        auto tiled_copy_Q = make_block_2d_copy_A(mma_qk, Q);
        // creat s2r copy for load K from SLM to register
        auto s2r_copy_K = make_B_slm_copies(mma_qk);

        // get copy slice for each work item
        auto thr_copy_q = tiled_copy_Q.get_slice(thr_id);
        auto thr_copy_s2r_k = s2r_copy_K.get_slice(thr_id);
        auto tKsK = thr_copy_s2r_k.partition_S(K_slm);

        auto thr_mma_qk = mma_qk.get_slice(thr_id);
        auto thr_mma_pv = mma_pv.get_slice(thr_id);

        /* Create register fragments for MMA and copies */
        // auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0,0));
        auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
        // retile for per-thread view from subgroup tensor
        auto tQrQ = thr_copy_q.retile_D(tSrQ);

        auto tSrK = thr_mma_qk.partition_sg_fragment_B(sK(_,_,0));
        auto tKrK = thr_copy_s2r_k.retile_D(tSrK);

        // FIXME: subgroup tensor or per-thread tensor???
        auto tSrS = thr_mma_qk.partition_sg_fragment_C(proxyP);


        // FIXME: use asm for loading gmem?
        int real_topk_length = HAVE_TOPK_LENGTH ? *(topk_length + seq_idx) : params.topk;

        const int* indices = params.indices + seq_idx * params.stride_indices_s_q;   // [topk]
        // assume each thead reads only one row of [B_TOPK] indices
        constexpr int NUM_SUBGROUPS = KernelConfig::NUM_THREADS / KernelConfig::SUBGROUP_SIZE;
        constexpr int NUM_ROWS_PER_SUBGROUP = (KernelConfig::B_TOPK + NUM_SUBGROUPS - 1) / NUM_SUBGROUPS;
        int token_indice[NUM_ROWS_PER_SUBGROUP];
        bool is_token_valid[NUM_ROWS_PER_SUBGROUP];
        auto load_token_indices = [&](int block_idx) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_ROWS_PER_SUBGROUP; ++i) {
                int offs = block_idx * KernelConfig::B_TOPK + sg_id * NUM_ROWS_PER_SUBGROUP + i;
                int cur_indice = *(indices + offs);
                token_indice[i] = cur_indice * params.stride_kv_s_kv;
                is_token_valid[i] = cur_indice >= 0 && cur_indice < params.s_kv;
                if constexpr (HAVE_TOPK_LENGTH) {
                    is_token_valid[i] &= offs < real_topk_length;
                }
            }
        };

        static_assert(KernelConfig::D_QK % KernelConfig::SUBGROUP_SIZE == 0, "D_QK must be divisible by SUBGROUP_SIZE");
        constexpr int NUM_VALS_PER_THREAD = KernelConfig::D_QK / KernelConfig::SUBGROUP_SIZE;
        static_assert(NUM_VALS_PER_THREAD % 4 == 0, "NUM_VALS_PER_THREAD must be divisible by 4 for vectorized copy");
        constexpr int NUM_BYTES_PER_VAL = 2; // bfloat16
        constexpr int VEC_SIZE = 8; // assuming using sycl::vec<uint8_t, 8> for copy
        constexpr int NUM_VEC_PER_THREAD = NUM_VALS_PER_THREAD / (VEC_SIZE / NUM_BYTES_PER_VAL);
        auto copy_k_tiles_g2s = [&](int block_idx) {
            // each subgroup loads multiple rows of K [B_TOPK, D_QK]
            for (int i = 0; i < NUM_ROWS_PER_SUBGROUP; ++i) {
                // int offs = token_indice[i] * D_QK;
                int offs = token_indice[i];
                CUTE_UNROLL
                for (int n = 0; n < NUM_VEC_PER_THREAD; ++n) {
                    int vec_offs = offs + n * KernelConfig::SUBGROUP_SIZE * VEC_SIZE + tid_in_sg * VEC_SIZE;
                    // FIXME: bf16 zero is 0x0000, correct conversion???
                    sycl::vec<uint8_t, VEC_SIZE> k_vec{0x0};
                    if (is_token_valid[i]) {
                        k_vec = reinterpret_cast<const sycl::vec<uint8_t, VEC_SIZE>*>(kv + vec_offs)[0];
                    }
                    int slm_vec_offs = (sg_id * NUM_ROWS_PER_SUBGROUP + i) * D_QK + n * KernelConfig::SUBGROUP_SIZE * VEC_SIZE + tid_in_sg * VEC_SIZE;
                    reinterpret_cast<sycl::vec<uint8_t, VEC_SIZE>*>(K_slm.data() + slm_vec_offs)[0] =  k_vec;
                }
            }

            // memory fence for SLM copy
            barrier_arrive(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
            barrier_wait(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);
        };

        // copy one tile of Q and K to call mma
        auto qkt_gemm_one_tile = [&](int tile_idx) {
            copy(tiled_copy_Q, gQ(_,_,tile_idx), tQrQ);
            copy(s2r_copy_K, tKsK(_,_,tile_idx), tKrK);
            cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        };

        // registers for each work item
        // Question:
        //  1) will cute::gemm only accept subgroup tensor?

        int num_topk_blocks = ceil_div(real_topk_length, KernelConfig::B_TOPK);
        CUTE_NO_UNROLL
        for (int topk_idx = 0; topk_idx < num_topk_blocks; ++topk_idx) {
            load_token_indices(topk_idx);
            copy_k_tiles_g2s(topk_idx);

            qkt_gemm_one_tile(0);
            qkt_gemm_one_tile(1);
            qkt_gemm_one_tile(2);
            qkt_gemm_one_tile(3);
            qkt_gemm_one_tile(4);
            qkt_gemm_one_tile(5);
            qkt_gemm_one_tile(6);
            qkt_gemm_one_tile(7);
            // D_QK / size<2>(TileShapeQK{}) = 576 / 64 = 9 tiles
            if (D_QK == 576) {
                qkt_gemm_one_tile(8);
            }
        }
    }
};


template<int D_QK, bool HAVE_TOPK_LENGTH, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class KernelTemplate_2 {
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

        using LayoutQ = Layout<Shape<Int<KernelConfig::B_H>, Int<D_QK>>, Stride<Int<KernelConfig::D_QK>, _1>>;
        using LayoutK = Layout<Shape<Int<KernelConfig::B_H>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;
        using LayoutO = Layout<Shape<Int<KernelConfig::B_H>, Int<KernelConfig::D_V>>, Stride<Int<KernelConfig::D_V>, _1>>;
        using LayoutIndices = Layout<Shape<Int<KernelConfig::B_H>, Int<KernelConfig::B_TOPK>>, Stride<Int<KernelConfig::B_TOPK>, _1>>;

        Tensor Q = make_tensor(make_gmem_ptr(q + cur_head_start_idx * D_QK), LayoutQ{});
        Tensor K = make_tensor(make_gmem_ptr(kv + cur_head_start_idx * D_QK), LayoutK{});
        Tensor O = make_tensor(make_gmem_ptr(kv + cur_head_start_idx * D_V), LayoutO{});

        Tensor indices = make_tensor(make_gmem_ptr(params.indices + seq_idx * num_heads_blocks * KernelConfig::B_TOPK + head_bid * KernelConfig::B_TOPK), LayoutIndices{});
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

    using Kernel = KernelTemplate_2<D_QK, HAVE_TOPK_LENGTH,
        CollectiveMainloop, CollectiveEpilogue, xe2::fwd::XeFHMAIndividualTileScheduler>;

    // using Kernel = KernelTemplate_1<D_QK, HAVE_TOPK_LENGTH>;

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
