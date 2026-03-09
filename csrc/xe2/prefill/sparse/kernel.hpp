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

// WA: avoid issue of not supported device-only copy
struct XPUSparseAttnFwdParams : public SparseAttnFwdParams {
    sycl::queue queue;
};

namespace xe2::fwd {

using namespace cute;

struct KernelConfig {
    using ElementQ = cutlass::bfloat16_t;
    using ElementKV = cutlass::bfloat16_t;
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
    static constexpr int HEAD_DIM_TILE_SIZE = 64;

    // FIXME: whether any models has topk value other than 2048
    static constexpr int INDEX_TOPK = 2048;

    static constexpr int stages = 1;

    // 576 / 32 = 18
    // Q head packing size = B_H
    using TileShapeQK = Shape<Int<B_H>, Int<B_TOPK>, Int<HEAD_DIM_TILE_SIZE>>;
    using SubgroupLayoutQK = Layout<Shape<_1, Int<NUM_SUBGROUPS>, _1>>;

    using TileShapePV = Shape<Int<B_H>, Int<HEAD_DIM_TILE_SIZE>, Int<B_TOPK>>;
    // using SubgroupLayoutPV = decltype(xe2::fwd::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
    using SubgroupLayoutPV = Layout<Shape<_1, _1, Int<NUM_SUBGROUPS>>>;

    // D_V / 64 = 8 tiles for v_dim
    using TileShapeOut = Shape<Int<B_H>, Int<HEAD_DIM_TILE_SIZE>>;

    // using SmemTileLayoutK = Layout<Shape<Int<B_TOPK>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;
    using SmemTileLayoutK = Layout<Shape<Int<B_TOPK>, Int<HEAD_DIM_TILE_SIZE>>, Stride<Int<HEAD_DIM_TILE_SIZE>, _1>>;
    using SmemTileLayoutV = Layout<Shape<Int<HEAD_DIM_TILE_SIZE>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>;

    constexpr static int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    // bf16 dpas m8n16k16
    // (8, 128, 64) / ((8, 16, 16) * (1, 16, 1)) = (1, 1, 4) iterations per subgroup
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, bfloat16_t>;
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;
};

template<int D_QK, bool HAVE_TOPK_LENGTH>
class KernelTemplate_1 {
public:
    using ElementQ = KernelConfig::ElementQ;
    using ElementKV = KernelConfig::ElementKV;
    using ElementO = KernelConfig::ElementO;
    using ElementS = typename KernelConfig::TiledMMAQK::ValTypeD;
    using ElementA = typename KernelConfig::TiledMMAPV::ValTypeD;

    // Split k-reduced tiles between participating subgroups.
    using ReduceK = decltype(size<3>(typename KernelConfig::TiledMMAPV::ThrLayoutVMNK{}));

     // (8,1,4)
    using SingleFragA = decltype(KernelConfig::TiledMMAPV{}.get_slice(0).partition_sg_fragment_C(make_identity_tensor(select<0,1>(KernelConfig::TileShapePV{}))));
    // （8,1,4,8)
    using FragA = expand_sg_fragment_t<SingleFragA, 1, KernelConfig::D_V / get<1>(KernelConfig::TileShapePV{})>;     // (atom val,q',v',VV)
    // storing updated global row-max and row-sum
    // (1) - thr_id 0 ~ B_H-1 holds max of each row
    using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
    // static_assert(is_same_v<decltype(FragA{}.shape()), float>, "dtype mismatch");

    static auto reduce_sg_v_helper() {
        constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
        constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
        return Int<(v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg) : v_total_sg>{};
    }

    using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
    using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
    using ReduceSGV = decltype(reduce_sg_v_helper());
    using ReduceSGLayout = decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

    using SGTileShapeO = decltype(shape_div(take<0,2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

    using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
        make_layout(select<1,0>(SGTileShapeO{}),
                    Stride<E<1>, E<0>>{})
    ));
    using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

    using SGPerWG = decltype(product(take<1,4>(shape(typename KernelConfig::TiledMMAPV::ThrLayoutVMNK{}))));

    struct SharedStorageNonReduceK {
        cute::array_aligned<cutlass::bfloat16_t, D_QK * KernelConfig::B_TOPK> k[KernelConfig::stages];
        bool is_kv_valid[KernelConfig::B_TOPK];
    };

    // Shared memory storage
    // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe block load infrastructure.
    using AlignedSGTileA_Q = C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) * intel::sg_size>;

    struct SharedStorageReduceK {
        cute::array_aligned<cutlass::bfloat16_t, D_QK * KernelConfig::B_TOPK> k[KernelConfig::stages];
        bool is_kv_valid[KernelConfig::B_TOPK];

        cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
        cute::array<ElementA,   AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data, a_max_data;
    };

    using SharedStorage = conditional_t<(ReduceK{} > 1), SharedStorageReduceK, SharedStorageNonReduceK>;
    static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                        : sizeof(SharedStorage);
    using Params = SparseAttnFwdParams;

    CUTLASS_DEVICE
    void operator()(const Params& params, char* smem_buf) const {
        using namespace sycl::ext::oneapi::this_work_item;

        // fix HEAD_DIM_TILE_SIZE to 64 now
        static_assert(KernelConfig::D_V / KernelConfig::HEAD_DIM_TILE_SIZE == 8, "Head dim must be divisible by tile size");

        SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

        const ElementQ* q = params.q;
        const ElementKV* kv = params.kv;
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

        using GmemLayoutQ = Layout<Shape<Int<KernelConfig::B_H>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;
        using SmemLayoutK = Layout<Shape<Int<KernelConfig::B_TOPK>, Int<D_QK>>, Stride<Int<D_QK>, _1>>;
        using SmemLayoutVTransposed = decltype(composition(SmemLayoutK{},
            Layout<Shape<Int<D_QK>, Int<KernelConfig::B_TOPK>>, Stride<Int<KernelConfig::B_TOPK>, _1>>{}));
        using GmemLayoutO = Layout<Shape<Int<KernelConfig::B_H>, Int<KernelConfig::D_V>>, Stride<Int<KernelConfig::D_V>, _1>>;

        Tensor Q = make_tensor(make_gmem_ptr(q + cur_head_start_idx * D_QK), GmemLayoutQ{});
        Tensor O = make_tensor(make_gmem_ptr(out + cur_head_start_idx * KernelConfig::D_V), GmemLayoutO{});

        Tensor proxyQ = make_identity_tensor(Q.shape());
        // (128,576)
        Tensor proxyK = make_identity_tensor(SmemLayoutK{}.shape());
        // S = Q@K^T
        // P = Softmax(S)
        // (8,128)
        Tensor proxyP = make_identity_tensor(select<0,1>(KernelConfig::TileShapeQK{})); // (h,k)
        // (576,128)
        Tensor proxyV = make_identity_tensor(SmemLayoutVTransposed{}.shape()); // (v,k)
        // O = P@V
        Tensor proxyO = make_identity_tensor(O.shape());

        // static_assert(is_same_v<decltype(proxyV.shape()), float>, "dtype mismatch");

        // FIXME: correct coord
        // (8,64,9)
        Tensor gQ = local_tile(proxyQ, select<0,2>(KernelConfig::TileShapeQK{}), make_coord(cur_head_start_idx,_)); // (h,d,D)
        // (128,64,9)
        Tensor sK = local_tile(proxyK, shape(KernelConfig::SmemTileLayoutK{}), make_coord(0,_)); // (k,d,D)
        // (64,128,9)
        Tensor sV = local_tile(proxyV, shape(KernelConfig::SmemTileLayoutV{}), make_coord(_,0)); // (d,k,D)
        Tensor gO = local_tile(proxyO, KernelConfig::TileShapeOut{}, make_coord(cur_head_start_idx,_));

        // static_assert(is_same_v<decltype(sV.shape()), float>, "dtype mismatch");

        KernelConfig::TiledMMAQK mma_qk{};
        KernelConfig::TiledMMAPV mma_pv{};

        // (V,M,N,K)
        //  V -> num threads per MMA atom
        //  M -> repetition of M dimension in MMA operation
        //  N -> repetition of N dimension in MMA operation
        //  K -> repetition of reduction dimension in MMA operation
        // V*M*N*K == total threads participating in the MMA operation
        //
        // (16,1,1,16)
        static_assert(is_same_v<decltype(typename KernelConfig::TiledMMAQK::ThrLayoutVMNK{}), float>, "dtype mismatch");

        // // Shared memory buffers
        // Layout K_slm_layout = make_layout(append<3>(typename decltype(r2s_K)::Tiler_MN{}, Int<Stages>{}));
        // [B_TOPK, D_QK]
        auto K_slm = make_tensor(make_smem_ptr(shared_storage.k[0].data()), SmemLayoutK{});
        auto V_slm = make_tensor(make_smem_ptr(shared_storage.k[0].data()), SmemLayoutVTransposed{});

        // create tiled copy for loading from gmem or storing to gmem
        auto tiled_copy_Q = make_block_2d_copy_A(mma_qk, Q);
        // auto tiled_copy_O = make_block_2d_copy_D(mma_pv, O);
        auto tiled_copy_O = (ReduceK{} > _1{}) ? make_block_2d_copy_D_subtiled(mma_pv, ReduceFragA{}.tv_layout(), ReduceSGLayout{}, proxyO)
                                            : make_block_2d_copy_A(mma_pv, O);
        // get copy slice for each work item
        auto thr_copy_q = tiled_copy_Q.get_slice(thr_id);
        auto thr_copy_o = tiled_copy_O.get_slice(thr_id);

        // dummy one to help create proper SLM copy
        auto tiled_copy_k = make_block_2d_copy_B(mma_qk, make_tensor(make_gmem_ptr(static_cast<ElementKV*>(nullptr)), SmemLayoutK{}));
        // create s2r copy for load K from SLM to register
        // FIXME: how to choose copy atom??? try with XE_1D_LDSM
        // using Copy_Atom_s2r_k = Copy_Atom<UniversalCopy<cutlass::AlignedArray<ElementKV, 16, 32>>, cutlass::AlignedArray<ElementKV, 16, 32>>;
        using Copy_Atom_s2r_k = Copy_Atom<XE_1D_LDSM<ElementKV>, ElementKV>;
        using TVLayoutLoad_k = typename decltype(tiled_copy_k)::TiledLayout_TV;
        using Tiler_MN_load_k = typename decltype(tiled_copy_k)::Tiler_MN;
        auto s2r_copy_K = TiledCopy<Copy_Atom_s2r_k, TVLayoutLoad_k, Tiler_MN_load_k>{};
        auto thr_copy_s2r_k = s2r_copy_K.get_slice(thr_id);

        auto tiled_copy_v = make_block_2d_copy_B(mma_pv, make_tensor(make_gmem_ptr(static_cast<ElementKV*>(nullptr)), SmemLayoutVTransposed{}));
        // using Copy_Atom_s2r_v = Copy_Atom<UniversalCopy<cutlass::AlignedArray<ElementKV, 16, 32>>, cutlass::AlignedArray<ElementKV, 16, 32>>;
        // FIXME: V is transposed in SLM, cannot use vectorized copy atom???
        // using Copy_Atom_s2r_v = Copy_Atom<UniversalCopy<ElementKV>, ElementKV>;
        using Copy_Atom_s2r_v = Copy_Atom<XE_1D_LDSM<ElementKV>, ElementKV>;
        using TVLayoutLoad_v = typename decltype(tiled_copy_v)::TiledLayout_TV;
        using Tiler_MN_load_v = typename decltype(tiled_copy_v)::Tiler_MN;
        auto s2r_copy_V = TiledCopy<Copy_Atom_s2r_v, TVLayoutLoad_v, Tiler_MN_load_v>{};
        auto thr_copy_s2r_v = s2r_copy_V.get_slice(thr_id);

        // static_assert(is_same_v<decltype(s2r_copy_V), float>, "dtype mismatch");

        auto thr_mma_qk = mma_qk.get_slice(thr_id);
        auto thr_mma_pv = mma_pv.get_slice(thr_id);

        // (((8,2),2),1,1,9)
        auto tQgQ = thr_copy_q.partition_S(gQ);
        // ((1,(2,32)),1,8)
        auto tKsK = thr_copy_s2r_k.partition_S(K_slm);

        /* Create register fragments for MMA and copies */
        // auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
        // (8,1,4) <- (8,128,64) / ((8,16,16) * (1,16,1))
        auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
        // retile for per-thread view from subgroup tensor
        // ((16,2),1,1)
        auto tQrQ = thr_copy_q.retile_D(tSrQ);

        // ((8,4),1,1,8)
        auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
        // ((8,4),1,1,8)
        auto tOgO = thr_copy_o.partition_D(gO);

        // static_assert(is_same_v<decltype(tOgO.shape()), float>, "dtype mismatch");

        // ((2,8),1,4)
        auto tSrK = thr_mma_qk.partition_sg_fragment_B(sK(_,_,0));
        // ((1,64),1,1)
        auto tKrK = thr_copy_s2r_k.retile_D(tSrK);

        // static_assert(is_same_v<decltype(tKsK.shape()), float>, "dtype mismatch");

        // FIXME: subgroup tensor or per-thread tensor???
        // (8,1,1)
        // auto tSrS = partition_fragment_C(thr_mma_qk, select<0,1>(KernelConfig::TileShapeQK{})); // (ATOM, MMA_M, MMA_N)
        // auto tSrS = thr_mma_qk.partition_fragment_C(gS(_,_,0));
        auto tSrS = thr_mma_qk.partition_sg_fragment_C(proxyP);

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
        auto qk_gemm_one_tile = [&](int tile_idx) {
            copy(tiled_copy_Q, tQgQ(_,_,_,tile_idx), tQrQ);
            copy(s2r_copy_K, tKsK(_,_,tile_idx), tKrK(_,_,0));
            cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        };

        auto mask_rS = [&]() {
            // for those invalid tokens, we need mask value to -INF for later softmax calculation
            Tensor coord_S = make_identity_tensor(select<0,1>(KernelConfig::TileShapeQK{}));
            Tensor tCgC = thr_mma_qk.partition_C(coord_S);
            CUTE_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
                // [B_H, B_TOPK]
                int col_idx = get<1>(tCgC(i));
                // if the column index corresponds to an invalid token, mask it out
                if (!shared_storage.is_kv_valid[sg_id * NUM_ROWS_PER_SUBGROUP + col_idx]) {
                    tSrS(i) = cutlass::platform::numeric_limits<ElementS>::lowest();
                }
            }
        };

        // storing block row-max and row-sum
        // (8,1,1)
        // using FragS = decltype(tSrS);
        // (1) - thr_id 0 ~ B_H-1 holds max of each row
        // using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));

        // (8,1,4)
        // using SingleFragA = decltype(thr_mma_pv.partition_sg_fragment_C(make_identity_tensor(select<0,1>(KernelConfig::TileShapePV{}))));
        // （8,1,4,8)
        // using FragA = expand_sg_fragment_t<SingleFragA, 1, KernelConfig::D_V / get<1>(KernelConfig::TileShapePV{})>;     // (atom val,q',v',VV)
        // storing updated global row-max and row-sum
        // (1) - thr_id 0 ~ B_H-1 holds max of each row
        // using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
        // static_assert(is_same_v<decltype(FragA{}.shape()), float>, "dtype mismatch");

        FragA tArA;
        FragARow tA_max, tA_sum;
        cute::clear(tArA);
        cute::fill(tA_max, cutlass::platform::numeric_limits<float>::lowest());
        cute::fill(tA_sum, 0.0f);
        auto online_softmax_and_rescale_o = [&](int block_idx) {
            auto tA_bmax = reduce<1>(tSrS, sycl::maximum{});
            auto tA_prev_max = tA_max;
            CUTE_UNROLL
            for (int i = 0; i < tA_max.size(); i++) {
                tA_max(i) = sycl::max(tA_max(i), params.sm_scale_div_log2 * tA_bmax(i));
            }

            CUTE_UNROLL
            for (int i = 0; i < tSrS.size(); i++) {
                tSrS(i) = sycl::native::exp2(params.sm_scale_div_log2 * tSrS(i) - broadcast<0>(tA_max, tSrS, i));
            }

            // rescale accumulated row-sum and output
            if (block_idx > 0) {
                FragARow rescale;

                CUTE_UNROLL
                for (int i = 0; i < tA_max.size(); i++) {
                    rescale(i) = sycl::native::exp2(tA_prev_max(i) - tA_max(i));
                    tA_sum(i) *= rescale(i);
                }

                CUTE_UNROLL
                for (int i = 0; i < tArA.size(); i++) {
                    tArA(i) *= broadcast<0>(rescale, tArA, i);
                }
            }

            // update global row-sum
            auto tA_bsum = reduce<1>(tSrS, sycl::plus<void>{});
            for (int i = 0; i < tA_sum.size(); i++) {
                tA_sum(i) += tA_bsum(i);
            }
        };

        // (8,1,1)
        auto tArP = thr_mma_pv.partition_sg_fragment_A(proxyP);
        // ((2,8),4,1)
        auto tArV = thr_mma_pv.partition_sg_fragment_B(sV(_,_,0));
        // ((1,(2,8,4)),9,1)
        auto tVsV = thr_copy_s2r_v.partition_S(V_slm);
        // ((1,64),1,1)
        auto tVrV = thr_copy_s2r_v.retile_D(tArV);

        // static_assert(is_same_v<decltype(tVsV.shape()), float>, "dtype mismatch");

        auto pv_gemm_one_tile = [&](int v_tile_idx) {
            copy(s2r_copy_V, tVsV(_,v_tile_idx,_), tVrV(_,_,0));
            // FIXME: can remove?
            reorder(tSrS, tArP);
            cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,v_tile_idx));
        };

        auto reduce_L = [&]() -> std::tuple<FragA, FragARow, bool> {
            // refer to https://github.com/intel/sycl-tla/blob/main/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp#L188
            if constexpr (ReduceK{} == _1{}) {
                return std::make_tuple(tArA, tA_sum, true);
            } else {
                /* Identify A tile ID and k block for this subgroup. */
                auto thr_vak = group<1,3>(mma_pv.get_thr_layout_vmnk()).get_flat_coord(assert_uniform(thr_id));
                auto a_tile = get<1>(thr_vak);
                auto k_blk = get<2>(thr_vak);

                /* Set up SLM tensors and partition A tiles among participating subgroups */
                auto shape_A     = append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{}/ReduceK{});
                auto shape_A_row = make_shape(get<0>(SGTileShapeO{}), shape(ReduceSGLayout{}), ReduceK{}, SGPerWG{}/ReduceK{});

                /* Physical layouts, with subtile modes broken out */
                auto sA_layout = group<2,4>(flat_divide(make_ordered_layout(shape_A, Step<_1,_0,_2,_3>{}), SGTileShapeO{}));
                auto sA_row_stride = make_stride(_1{}, make_stride(get<0>(shape_A_row), _0{}),
                                                AlignedSGTileA_Q{}, AlignedSGTileA_Q{} * ReduceK{});
                auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

                /* Coordinate layouts, with subtile modes broken out */
                auto basis2 = make_basis_like(SGTileShapeO{});
                auto sA_coords = make_layout(append(SGTileShapeO{}, shape(ReduceSGLayout{})),
                                            append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

                auto sA     = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_data),     sA_layout);      // (q,v,rblk_dst,rblk_src,a_tile)
                auto sA_max = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_max_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
                auto sA_sum = make_tensor(make_smem_ptr<ElementA>(&shared_storage.a_sum_data), sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

                /* Write my contributions to SLM. */
                copy_block_r2s(tA_max, sA_max(_,_,k_blk,a_tile));
                barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
                copy_block_r2s(tA_sum, sA_sum(_,_,k_blk,a_tile));
                copy_block_r2s(tArA, sA(_,_,_,k_blk,a_tile), sA_coords);

                bool active = (k_blk      < size(ReduceSGLayout{}))
                            || (ReduceK{} == size(ReduceSGLayout{}));    // help compiler out

                /* Wait for maxima to be available, signal other data available */
                barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
                barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

                ReduceFragA rA;
                ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

                if (active) {
                    /* Read A_max back from SLM and reduce. */
                    CUTLASS_PRAGMA_UNROLL
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        copy_block_s2r(sA_max(_,k_blk,kr,a_tile), rA_kmax[kr]);
                    }

                    rA_max = rA_kmax[0];
                    for (int kr = 1; kr < ReduceK{}; kr++)
                        cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});

                    /* Calculate scale factors for aligning per-block maxima. */
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        cute::transform(rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) {
                            return sycl::native::exp2(kmax - gmax);
                        });
                    }
                }

                /* Wait for A/A_sum data to be available */
                barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

                if (active) {
                    /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce. */
                    clear(rA_sum);

                    CUTLASS_PRAGMA_UNROLL
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        ReduceFragARow rA_sum_read;
                        copy_block_s2r(sA_sum(_,k_blk,kr,a_tile), rA_sum_read);

                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < rA_sum_read.size(); i++) {
                            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
                        }
                    }

                    clear(rA);

                    CUTLASS_PRAGMA_UNROLL
                    for (int kr = 0; kr < ReduceK{}; kr++) {
                        ReduceFragA rA_read;
                        copy_block_s2r(sA(_,_,k_blk,kr,a_tile), sA_coords(_,_,0), rA_read);

                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < rA_read.size(); i++) {
                            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
                        }
                    }
                }
                return std::make_tuple(rA, rA_sum, active);
            }
        };

        auto final_rescale_and_store_o = [&]() {
            // reduce global row-sum
            auto [tArA, tA_sum, active] = reduce_L();

            if (!active) return;

            // rescale output
            CUTE_UNROLL
            for (int i = 0; i < tA_sum.size(); ++i) {
                tA_sum(i) =  ElementA(1) / tA_sum(i);
            }

            CUTE_UNROLL
            for (int i = 0; i < tArA.size(); ++i) {
                tArA(i) *= broadcast<0>(tA_sum, tArA, i);
            }

            reorder(tArA, tOrO);
            copy(tiled_copy_O, tOrO, tOgO);
        };

        int num_topk_blocks = ceil_div(real_topk_length, KernelConfig::B_TOPK);
        // mainloop
        CUTE_NO_UNROLL
        for (int topk_idx = 0; topk_idx < num_topk_blocks; ++topk_idx) {
            load_token_indices_and_save_to_slm(topk_idx);
            copy_k_tiles_g2s(topk_idx);

            clear(tSrS);
            qk_gemm_one_tile(0);
            qk_gemm_one_tile(1);
            qk_gemm_one_tile(2);
            qk_gemm_one_tile(3);
            qk_gemm_one_tile(4);
            qk_gemm_one_tile(5);
            qk_gemm_one_tile(6);
            qk_gemm_one_tile(7);
            // D_QK / size<2>(TileShapeQK{}) = 576 / 64 = 9 tiles
            if constexpr (D_QK == 576) {
                qk_gemm_one_tile(8);
            }

            mask_rS();
            online_softmax_and_rescale_o(topk_idx);

            // D_V / 64 = 512 / 64 = 8 tiles
            pv_gemm_one_tile(0);
            pv_gemm_one_tile(1);
            pv_gemm_one_tile(2);
            pv_gemm_one_tile(3);
            pv_gemm_one_tile(4);
            pv_gemm_one_tile(5);
            pv_gemm_one_tile(6);
            pv_gemm_one_tile(7);
        }

        // epilogue
        final_rescale_and_store_o();
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
