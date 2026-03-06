#pragma once

#include "../../../params.h"
#include "../../utils.hpp"

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/device_kernel.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/algorithm/subgroup_algorithms.hpp>

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
    // FIXME: try bigger number, e.g 256
    static constexpr int B_TOPK = 128;    // topk_length block size

    // static constexpr int D_QK = 576;
    static constexpr int D_PE = 64;
    static constexpr int D_V = 512;

    // FIXME: whether any models has topk value other than 2048
    static constexpr int INDEX_TOPK = 2048;

    static constexpr int stages = 1;

    // 576 / 32 = 18
    // Q head packing size = B_H
    using TileShapeQK = Shape<Int<B_H>, Int<B_TOPK>, _64>;
    using SubgroupLayoutQK = Layout<Shape<_1, Int<NUM_SUBGROUPS>, _1>>;

    using TileShapePV = Shape<Int<B_H>, _64, Int<B_TOPK>>;
    // using SubgroupLayoutPV = decltype(xe2::fwd::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
    using SubgroupLayoutPV = Layout<Shape<_1, _1, Int<NUM_SUBGROUPS>>>;

    // D_V / 64 = 8 tiles for v_dim
    using TileShapeOut = Shape<Int<B_H>, _64>;

    // using SmemLayoutK = Layout<Shape<Int<B_TOPK>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<B_TOPK>, _64>, Stride<_64, _1>>;
    // using SmemLayoutVTransposed = Layout<Shape<_64, Int<B_TOPK>>, Stride<_1, Int<B_TOPK>>>;

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
        bool is_kv_valid[KernelConfig::B_TOPK];
    };

    template <typename TiledMMA>
    using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
                                make_identity_tensor(select<0,1>(TiledMMA{}.tile_mnk()))));

    static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                        : sizeof(SharedStorage);
    using Params = SparseAttnFwdParams;

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        using namespace sycl::ext::oneapi::this_work_item;

        SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

        using ElementQ = KernelConfig::ElementQ;
        using ElementK = KernelConfig::ElementK;
        using ElementO = KernelConfig::ElementO;
        using ElementS = typename KernelConfig::TiledMMAQK::ValTypeD;

        const ElementQ* q = params.q;
        const ElementK* kv = params.kv;
        const float* attn_sink = params.attn_sink;
        const int* topk_length = params.topk_length;
        ElementO* out = params.out;
        float* max_logits = params.max_logits;
        float* lse = params.lse;

        const int thr_id = int(ThreadIdxX());
        const int wg_id = int(BlockIdxX());

        const int sg_id = thr_id / KernelConfig::SUBGROUP_SIZE;
        const int tid_in_sg = thr_id % KernelConfig::SUBGROUP_SIZE;

        int num_head_blocks = params.h_q / KernelConfig::B_H;
        int seq_idx = wg_id / num_head_blocks;
        int head_bid = wg_id % num_head_blocks;
        // start idx of current head block
        int cur_head_start_idx = head_bid * KernelConfig::B_H;

        using LayoutQ = Layout<Shape<Int<KernelConfig::B_H>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;
        using LayoutK = Layout<Shape<Int<KernelConfig::B_TOPK>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;
        using LayoutO = Layout<Shape<Int<KernelConfig::B_H>, Int<KernelConfig::D_V>>, Stride<Int<KernelConfig::D_V>, _1>>;

        Layout layout_K = make_layout(make_shape(params.s_kv, params.d_qk), make_stride(params.stride_kv_s_kv, _1{}));

        Tensor Q = make_tensor(make_gmem_ptr(q + cur_head_start_idx * D_QK), LayoutQ{});
        Tensor O = make_tensor(make_gmem_ptr(out + cur_head_start_idx * KernelConfig::D_V), LayoutO{});

        Tensor proxyQ = make_identity_tensor(Q.shape());
        Tensor proxyK = make_identity_tensor(LayoutK{}.shape());
        // Tensor proxyV = make_identity_tensor(V.shape());
        Tensor proxyP = make_identity_tensor(take<0,2>(KernelConfig::TileShapeQK{})); // (h,k)
        Tensor proxyO = make_identity_tensor(O.shape());

        // FIXME: correct coord
        // (8,64,9)
        Tensor gQ = local_tile(proxyQ, select<0,2>(KernelConfig::TileShapeQK{}), make_coord(cur_head_start_idx,_)); // (h,d,D)
        Tensor gO = local_tile(proxyO, shape(KernelConfig::TileShapeOut{}), make_coord(cur_head_start_idx,_));
        // (128,64,9)
        Tensor sK = local_tile(proxyK, shape(KernelConfig::SmemLayoutK{}), make_coord(0,_)); // (k,d,D)

        // static_assert(is_same_v<decltype(sK.shape()), float>, "dtype mismatch");

        KernelConfig::TiledMMAQK mma_qk{};
        KernelConfig::TiledMMAPV mma_pv{};

        // // Shared memory buffers
        // Layout K_slm_layout = make_layout(append<3>(typename decltype(r2s_K)::Tiler_MN{}, Int<Stages>{}));
        // [B_TOPK, D_QK]
        auto K_slm = make_tensor(make_smem_ptr(shared_storage.k[0].data()), KernelConfig::SmemLayoutK{});

        // create tiled copy for loading Q from gmem
        auto tiled_copy_Q = make_block_2d_copy_A(mma_qk, Q);
        // get copy slice for each work item
        auto thr_copy_q = tiled_copy_Q.get_slice(thr_id);

        // dummy one to help create proper SLM copy
        auto tiled_copy_k = make_block_2d_copy_B(mma_qk, make_tensor(make_gmem_ptr(static_cast<ElementK*>(nullptr)), LayoutK{}));
        // create s2r copy for load K from SLM to register
        // FIXME: how to choose copy atom???
        using Copy_Atom_s2r = Copy_Atom<UniversalCopy<cutlass::AlignedArray<cutlass::bfloat16_t, 16, 32>>, cutlass::AlignedArray<cutlass::bfloat16_t, 16, 32>>;
        using TVLayoutLoad = typename decltype(tiled_copy_k)::TiledLayout_TV;
        using Tiler_MN_load = typename decltype(tiled_copy_k)::Tiler_MN;
        auto s2r_copy_K = TiledCopy<Copy_Atom_s2r, TVLayoutLoad, Tiler_MN_load>{};
        auto thr_copy_s2r_k = s2r_copy_K.get_slice(thr_id);

        // static_assert(is_same_v<decltype(s2r_copy_K), float>, "dtype mismatch");

        auto thr_mma_qk = mma_qk.get_slice(thr_id);
        auto thr_mma_pv = mma_pv.get_slice(thr_id);

        auto tQgQ = thr_copy_q.partition_S(gQ);
        // ((16,4),1,1)
        auto tKsK = thr_copy_s2r_k.partition_S(K_slm);

        /* Create register fragments for MMA and copies */
        // auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
        // (8,1,4) <- (8,128,64) / ((8,16,16) * (1,16,1))
        auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
        // retile for per-thread view from subgroup tensor
        // ((16,2),1,1)
        auto tQrQ = thr_copy_q.retile_D(tSrQ);
        // static_assert(is_same_v<decltype(tQrQ.shape()), float>, "dtype mismatch");

        // ((2,8),1,4)
        auto tSrK = thr_mma_qk.partition_sg_fragment_B(sK(_,_,0));
        // ((1,64),1,1)
        auto tKrK = thr_copy_s2r_k.retile_D(tSrK);

        // static_assert(is_same_v<decltype(tKrK.shape()), float>, "dtype mismatch");

        // FIXME: subgroup tensor or per-thread tensor???
        // (8,1,1)
        auto tSrS = partition_fragment_C(thr_mma_qk, select<0,1>(KernelConfig::TileShapeQK{})); // (ATOM, MMA_M, MMA_N)

        // static_assert(is_same_v<decltype(tSrS.shape()), float>, "dtype mismatch");
        // static_assert(is_same_v<Int<size(tSrS)>, float>, "dtype mismatch");

        // FIXME: use inline asm for loading gmem?
        int real_topk_length = HAVE_TOPK_LENGTH ? *(topk_length + seq_idx) : params.topk;

        const int* indices = params.indices + seq_idx * params.stride_indices_s_q;   // [topk]
        // assume each thead reads only one row of [B_TOPK] indices
        constexpr int NUM_SUBGROUPS = KernelConfig::NUM_THREADS / KernelConfig::SUBGROUP_SIZE;
        constexpr int NUM_ROWS_PER_SUBGROUP = (KernelConfig::B_TOPK + NUM_SUBGROUPS - 1) / NUM_SUBGROUPS;
        int token_indice[NUM_ROWS_PER_SUBGROUP];
        bool is_token_valid[NUM_ROWS_PER_SUBGROUP];
        auto load_token_indices_and_save_to_slm = [&](int block_idx) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_ROWS_PER_SUBGROUP; ++i) {
                int offs = block_idx * KernelConfig::B_TOPK + sg_id * NUM_ROWS_PER_SUBGROUP + i;
                int cur_indice = *(indices + offs);
                token_indice[i] = cur_indice * params.stride_kv_s_kv;
                is_token_valid[i] = cur_indice >= 0 && cur_indice < params.s_kv;
                if constexpr (HAVE_TOPK_LENGTH) {
                    is_token_valid[i] &= offs < real_topk_length;
                }
                if (tid_in_sg == 0) {
                    // only one thread writes to shared memory
                    shared_storage.is_kv_valid[sg_id * NUM_ROWS_PER_SUBGROUP + i] = is_token_valid[i];
                }
            }
        };

        static_assert(D_QK % KernelConfig::SUBGROUP_SIZE == 0, "D_QK must be divisible by SUBGROUP_SIZE");
        constexpr int NUM_VALS_PER_THREAD = D_QK / KernelConfig::SUBGROUP_SIZE;
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
                    reinterpret_cast<sycl::vec<uint8_t, VEC_SIZE>*>(shared_storage.k[0].data() + slm_vec_offs)[0] =  k_vec;
                }
            }

            // memory fence for SLM copy
            barrier_arrive(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
            barrier_wait(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);
        };

        // copy one tile of Q and K to call mma
        auto qkt_gemm_one_tile = [&](int tile_idx) {
            copy(tiled_copy_Q, tQgQ(_,_,_,tile_idx), tQrQ);
            copy(s2r_copy_K, tKsK(_,_,_), tKrK);
            cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        };

        auto mask_rS = [&]() {
            // for those invalid tokens, we need mask value to -INF for later softmax calculation
            Tensor coord_S = make_identity_tensor(select<0,1>(KernelConfig::TileShapeQK{}));
            Tensor tCgC = thr_mma_qk.partition_C(coord_S);
            CUTE_UNROLL
            for (int i = 0; i < size(tSrS); ++i) {
                // [B_H, B_TOPK]
                int col_idx = get<1>(tCgC(i));
                // if the column index corresponds to an invalid token, mask it out
                if (!shared_storage.is_kv_valid[sg_id * NUM_ROWS_PER_SUBGROUP + col_idx]) {
                    tSrS(i) = ElementS(-INFINITY);
                }
            }
        };
        
        // (8,1,1)
        using FragS = FragC<KernelConfig::TiledMMAQK>;
        // (1) - thr_id 0 ~ B_H-1 holds max of each row
        using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));

        // (8,1,4)
        using SingleFragA = FragC<KernelConfig::TiledMMAPV>;       // (atom val,q',v')
        // (8,1,4,8)
        using FragA = expand_sg_fragment_t<SingleFragA, 1, 8>;     // (atom val,q',v',VV)
        // (1)
        using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
        
        // Tensor rO = partition_fragment_C(KernelConfig::TiledMMAPV{}, Shape<Int<KernelConfig::B_H>, Int<KernelConfig::D_V>>{}); // (h,VV)
        // (8,1,4)
        Tensor rO = thr_mma_pv.partition_sg_fragment_C(gO(_,_,0));
        static_assert(is_same_v<decltype(rO.shape()), float>, "dtype mismatch");

        auto online_softmax = [&]() {

        };

        auto pv_gemm_one_tile = [&]() {

        };

        int num_topk_blocks = ceil_div(real_topk_length, KernelConfig::B_TOPK);
        CUTE_NO_UNROLL
        for (int topk_idx = 0; topk_idx < num_topk_blocks; ++topk_idx) {
            load_token_indices_and_save_to_slm(topk_idx);
            copy_k_tiles_g2s(topk_idx);

            clear(tSrS);
            qkt_gemm_one_tile(0);
            qkt_gemm_one_tile(1);
            qkt_gemm_one_tile(2);
            qkt_gemm_one_tile(3);
            qkt_gemm_one_tile(4);
            qkt_gemm_one_tile(5);
            qkt_gemm_one_tile(6);
            qkt_gemm_one_tile(7);
            // D_QK / size<2>(TileShapeQK{}) = 576 / 64 = 9 tiles
            if constexpr (D_QK == 576) {
                qkt_gemm_one_tile(8);
            }

            mask_rS();
            online_softmax();
            pv_gemm_one_tile();
        }
    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
void launch_kernel(const XPUSparseAttnFwdParams& params) {
    KU_ASSERT(params.h_kv == 1, "h_kv must be 1");
    KU_ASSERT(params.h_q % KernelConfig::B_H == 0, "h_q must be divisible by ", KernelConfig::B_H);
    KU_ASSERT(params.topk % KernelConfig::B_TOPK == 0, "topk must be divisible by ", KernelConfig::B_TOPK);
    KU_ASSERT(params.d_qk == D_QK, "Invalid d_qk for this kernel instantiation");

    // create kernel
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using Kernel = KernelTemplate_1<D_QK, HAVE_TOPK_LENGTH>;

    dim3 block(KernelConfig::NUM_THREADS, 1, 1);
    dim3 grid((params.h_q / KernelConfig::B_H) * params.s_q, 1, 1);

    // typename Kernel::Params kernel_params {
    //     params,
    //     {params.sm_scale, nullptr, 0, nullptr},
    //     {},
    //     hw_info
    // };

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    const int smem_size = Kernel::SharedStorageSize;

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    auto event = launch<cutlass::device_kernel<Kernel>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<KernelConfig::SUBGROUP_SIZE>}},
        params.queue, static_cast<SparseAttnFwdParams>(params));
#else
    compat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<KernelConfig::SUBGROUP_SIZE>
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
