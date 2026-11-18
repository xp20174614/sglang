/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SM100 sparse MLA with e4m3 Q and *fp4 KV read directly as the MMA B operand*: the 512 NoPE dims live
// in the pool as e2m1 nibbles and are never expanded, the 64 RoPE dims stay e4m3, and one ue8m0 exponent
// per KV row carries a per-row power-of-two scale (321 B/row instead of 576). Both halves of a row are
// stored pre-scaled by 2^-k, so the kernel restores 2^k once per token, in the softmax: on the logits
// (QK^T's N dim) and on S (PV's K dim). Everything else is the e4m3 kernel with b_format 5 in the two
// descriptors: TMA 16U4_ALIGN16B reproduces the byte layout an E2M1 B operand expects while still
// occupying 1 B per element in smem, so every stage size, address, LBO, SBO and barrier count is
// unchanged.
//
// Inputs follow TRT-LLM-gen's FP8 sparse MLA: merged e4m3 Q, 576 B e4m3 KV rows, token-level top-k
// slots, and fused bmm1 / bmm2 scales. The bmm1 scale is folded into the softmax log2 constant.
//
// Persistent CTAs walk work items (query token, 64-head block, split). Per 64-token block, QK^T is
// split between two issuers into two partial-logit tiles: TS MMAs over NoPE (Q in TMEM, duplicated in
// both lane halves so each lane holds a full dot product) and SS MMAs over RoPE. P is e4m3 with a
// lazy O rescale, and PV reads V as the K NoPE tile in MN-major order. O stays in TMEM. Issuer A
// issues the next item's Q copy and first QK^T before the last PV of the current item.
//
// Warp roles (512 threads):
//   warps 0-3    softmax, O rescale
//   warps 4-7    TMA gather4 of K NoPE and K RoPE
//   warp  8      MMA issuer A: TMEM alloc, Q UTCCP, QK^T NoPE dims 0-255, PV
//   warp  9      top-k validity masks
//   warp  10     Q NoPE / RoPE TMA for the next item
//   warp  11     MMA issuer B: QK^T NoPE dims 256-511 and RoPE
//   warps 12-15  epilogue, overlapping the softmax of the next item
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "helpers.h"
#include "params.h"
#include <algorithm>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace sglang::q8kv8_sm100 {

constexpr int D_NOPE = 512;
constexpr int D_ROPE = 64;
constexpr int D_QK = D_NOPE + D_ROPE;
constexpr int D_V = 512;
constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int NUM_STAGES = 4;
// NoPE K-steps (of 16) issued by issuer A; issuer B takes the rest plus RoPE.
constexpr int KA = 16;  // PT: issuer A runs all nope K-steps into P_A, B only rope into P_B
constexpr int NUM_THREADS = 512;
constexpr int MMA_A_WARP = 8;
constexpr int MASK_WARP = 9;
constexpr int Q_LOAD_WARP = 10;
constexpr int MMA_B_WARP = 11;
constexpr int EPI_WARP0 = 12;
constexpr float MAX_INIT_VAL = -1e30f;
// e4m3 tops out at 448: P * 2^P_SHIFT never exceeds 2^(P_SHIFT + RESCALE_THRESHOLD) = 256.
constexpr float P_SHIFT = 4.0f;
constexpr float RESCALE_THRESHOLD = 4.0f;

// Timing ablation only (results are wrong when != 0): the per-token r_t machinery is the only softmax
// work this variant adds, so switching it off separates "what fp4 TMA/MMA costs" from "what the rescale
// costs".
//   0 = full, correct fp4 path
//   2 = no exponent map, no rescale (pure fp4 TMA/MMA data path)
//   5 = exponent map on, rescale off (the cost of the extra TMA map alone)
//   6 = rescale on, exponent map off (garbage results: the softmax side in isolation)
#ifndef PT_MASK32
#define PT_MASK32 1
#endif
#ifndef FP4_MMA_MODE
#define FP4_MMA_MODE 0
#endif
// v2: V2_SF_LDG=1 stores the exponents as a dense [s_kv] byte array (321 B/row pool, rope rows 64 B) that
// the mask warp loads itself, ahead of time, from the indices it already prefetches; no exponent TMA map.
#ifndef V2_SF_LDG
#define V2_SF_LDG 1  // PT step 2: r_t from a dense e4m3 array, loaded by the mask warp
#endif
#define FP4_STAGE_SF (!V2_SF_LDG && (FP4_MMA_MODE == 0 || FP4_MMA_MODE == 5))
#define FP4_LDG_SF (V2_SF_LDG && (FP4_MMA_MODE == 0 || FP4_MMA_MODE == 5))
#define FP4_READ_SF (FP4_MMA_MODE == 0 || FP4_MMA_MODE == 6)
#define FP4_APPLY_SF (FP4_MMA_MODE == 0 || FP4_MMA_MODE == 6)
// v2 timing ablations (results wrong when either is non-default):
//   ABL_V_U8=1   : NoPE/V tile via a plain UINT8 map over a 512 B/row pool (e4m3-sized TMA traffic)
//   ABL_BFMT     : b_format of the NoPE QK and PV descriptors (5 = E2M1, 0 = E4M3)
#ifndef ABL_V_U8
#define ABL_V_U8 0
#endif
#ifndef ABL_BFMT
#define ABL_BFMT 5
#endif
// v2: who turns the per-token exponent into r_t.
//   V2_RT=1 (default): the mask warp (otherwise idle) waits for the stage's TMA, converts its 8 tokens'
//            exponents to floats and stores them packed in kv_rt[st]; the softmax reads its 32 r_t with
//            8 broadcast LDS.128 (no shuffles, off the softmax's critical path).
//   V2_RT=0: reference fp4_mma behaviour (1 LDS + 32 shfl + 32 shifts inside the softmax).
#ifndef V2_RT
#define V2_RT 1
#endif
#define V2_RT_SMEM (V2_RT == 1 && (FP4_STAGE_SF || FP4_LDG_SF))
#ifndef V2_ABL_RT_CONST
#define V2_ABL_RT_CONST 0
#endif

constexpr int K_NOPE_STAGE_BYTES = B_TOPK * D_NOPE;  // 4 SW128 column tiles of 64 rows x 128B
constexpr int K_ROPE_STAGE_BYTES = B_TOPK * D_ROPE;  // 1 SW64 tile of 64 rows x 64B
constexpr int Q_ROPE_BYTES = B_H * D_ROPE;           // same layout as a K RoPE stage
constexpr int S_BYTES = B_H * B_TOPK;                // 64 rows x 64B, SW64
// Bytes TMA reports to the mbarrier for one row group (4 tokens) of this pool: the e2m1 half writes only
// 8 B of data per 16 B ALIGN16B slot, so the reported count is the packed payload, and NOT the 4 * 576 =
// 2304 B of smem footprint the e4m3 baseline reported. The rope map contributes 64 B per token (its box
// is 64 elements wide on an 80 B row) and the exponent map 16 B.
constexpr int KV_BYTES_PER_ROW_GROUP = 4 * ((ABL_V_U8 ? D_NOPE : D_NOPE / 2) + D_ROPE) + (FP4_STAGE_SF ? 4 * 16 : 0);
// The fp4 pool the MMA reads directly: 256 B of packed e2m1 for the 512 NoPE dims plus an 80 B row that
// holds the e4m3 RoPE dims and this row's ue8m0 exponent = 336 B/row instead of 576.
// The e2m1 half goes through TMA 16U4_ALIGN16B, whose smem image is 1 B per element nominal (8 B of data
// + 8 B of gap per 16 B slot), so every byte offset, span, LBO and SBO below stays exactly what it was
// for the e4m3 tile: the K NoPE stage is still 4 tiles of 64 rows x 128 B, and one 128-element box still
// reports/writes 128 B per row.
constexpr int KV4_V_ROW_BYTES = ABL_V_U8 ? D_NOPE : D_NOPE / 2;  // 256
// The RoPE half is 80 B/row: 64 B of e4m3 rope followed by 16 B that hold this row's ue8m0 exponent in
// the first byte. TMA global strides must be multiples of 16 B, so the exponent cannot be packed tighter
// than a 16 B granule; folding it into the rope row means the same two tensor maps that already fetch the
// KV also fetch the scale, and the kernel needs no scattered per-token LDG.U8 at all
// (that LDG was +10.4% end to end). Pool: 256 + 80 = 336 B/row instead of 576.
constexpr int KV4_ROPE_ROW_BYTES = V2_SF_LDG ? D_ROPE : D_ROPE + 16;  // 80 (64 with V2_SF_LDG)
constexpr int KV4_SF_OFFSET = D_ROPE;            // 64: the exponent's 16 B window starts here
namespace tmem_cols {
constexpr uint32_t O = 0;         // 256 columns: two .ws N=256 halves of the 64 x 512 output
constexpr uint32_t P_A = 256;     // 2 x 32 columns: logits over NoPE dims 0-255
constexpr uint32_t P_B = 320;     // 2 x 32 columns: logits over NoPE dims 256-511 plus RoPE
constexpr uint32_t Q_NOPE = 384;  // 128 columns: 512 e4m3 per lane, rows duplicated in both halves
}  // namespace tmem_cols

// QK^T and PV now multiply against the E2M1 pool tile, so their B format is 5 (E2M1 in TMA's 8-bit
// container); the A operands stay e4m3 (Q for QK^T, S for PV). The RoPE tile is *not* quantized -- it is
// still e4m3 in the pool -- so its SS MMA keeps b_format 0 and needs its own descriptor.
constexpr uint32_t IDESC_QK = make_idesc(0, ABL_BFMT, false, B_H, B_TOPK);
constexpr uint32_t IDESC_QK_ROPE = make_idesc(0, 0, false, B_H, B_TOPK);
constexpr uint32_t IDESC_PV = make_idesc(0, ABL_BFMT, true, B_H, 256);

enum NamedBarriers : uint32_t {
  // Warps w and w^2 own the two token halves of the same 32 rows.
  pair_sync_base = 1,
};

struct alignas(1024) SharedStorage {
  uint8_t k_nope[NUM_STAGES][K_NOPE_STAGE_BYTES];
  uint8_t q_nope[B_H * D_NOPE];  // K NoPE stage layout, UTCCP source
  uint8_t k_rope[NUM_STAGES][K_ROPE_STAGE_BYTES];
  uint8_t q_rope[2][Q_ROPE_BYTES];  // SS operand for the whole item, so double-buffered by item
  uint8_t s[2][S_BYTES];
  float max_buf[2][128];
  float sum_buf[128];
  uint8_t is_k_valid[NUM_STAGES][B_TOPK / 8];
  // Per-token ue8m0 exponent of the fp4 pool (r_t = 2^(kv4_sf - 127)), delivered by the same TMA gather4
  // as the KV it belongs to: 4 tokens x 16 B per row group, exponent in the first byte of each window.
  // TMA wants a 128B-aligned destination (a 16B-aligned one faults with "misaligned address"), so each
  // row group's 64 B window sits at the bottom of its own 128 B granule.
#if !V2_SF_LDG
  alignas(128) uint8_t kv4_sf[NUM_STAGES][B_TOPK * 32];
#endif
  // v2: r_t = 2^k per token of each stage, packed, written by the mask warp (V2_RT_SMEM).
  alignas(16) float kv_rt[NUM_STAGES][B_TOPK];
  // Per-lane output scale of item it for the epilogue. 3 deep: softmax of item it+2 waits only on PV
  // of item it, so it can run two items ahead of the epilogue.
  float epi_scale[3][128];
  // q_full alternates by item: issuer B may still be waiting on item i when item i+1 lands.
  transac_bar_t bar_q_full[2], bar_q_free, bar_o_free, bar_epi_ready[3];
  transac_bar_t bar_kv_full[NUM_STAGES], bar_sv_done[NUM_STAGES];
  transac_bar_t bar_k_valid_ready[NUM_STAGES], bar_k_valid_free[NUM_STAGES];
  transac_bar_t bar_qk_done[2], bar_p_free[2], bar_s_ready[2];
  uint32_t tmem_base;
};

struct TmaMaps {
  CUtensorMap q;        // u8 {576, h_q, s_q}, box {128, 64, 1}, SW128 (NoPE column tiles)
  CUtensorMap q_rope;   // u8 {576, h_q, s_q}, box {64, 64, 1},  SW64  (column 512)
  CUtensorMap kv;       // 16U4_ALIGN16B {512, s_kv}, box {128, 1}, SW128, gather4 (packed e2m1, 256 B/row)
  CUtensorMap kv_rope;  // u8 {80, s_kv},          box {64, 1},  SW64,  gather4 (e4m3 rope, 64 B/row)
  CUtensorMap kv_sf;    // u8 {80, s_kv},          box {16, 1},  NONE,  gather4 (ue8m0, col 64)
};

// SW64 K-major tile with 64B rows: physical offset of 16B chunk `c` in row `r`.
__device__ __forceinline__ uint32_t sw64_offset(int r, int c) {
  return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
}

struct WorkItem {
  int s_q_idx, head_block, split, num_blocks, topk_length;
};

// Items run (query token, 64-head block, split) with the split fastest.
template <bool HAVE_TOPK_LENGTH>
__device__ __forceinline__ WorkItem get_item(const SparseMlaQ8Kv8Sm100Params& params, int item) {
  const int items_per_token = (params.h_q / B_H) * params.num_splits;
  WorkItem w;
  w.s_q_idx = item / items_per_token;
  const int r = item % items_per_token;
  w.head_block = r / params.num_splits;
  w.split = r % params.num_splits;
  const int total = HAVE_TOPK_LENGTH ? max(0, min(__ldg(params.topk_length + w.s_q_idx), params.topk)) : params.topk;
  w.topk_length = max(0, min(total - w.split * params.split_len, params.split_len));
  // At least one block keeps the pipeline shape fixed; an empty row is fully masked.
  w.num_blocks = max((w.topk_length + B_TOPK - 1) / B_TOPK, 1);
  return w;
}

__device__ __forceinline__ const int* item_indices(const SparseMlaQ8Kv8Sm100Params& params, const WorkItem& w) {
  return params.indices + int64_t(w.s_q_idx) * params.stride_indices_s_q + w.split * params.split_len;
}

// PT: r_t is stored as one e4m3 byte (positive, normal or subnormal).
__device__ __forceinline__ float pt_e4m3_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

// PT step 11: exp2 on the FMA pipe for a fraction of the softmax elements
#define PT_EMU_R 4
#define PT_EMU_MASK 0
#define PT_EMU_DEG 2
__device__ __forceinline__ float2 pt_exp2_emu(float2 x) {
  x.x = fmaxf(x.x, -125.0f);
  x.y = fmaxf(x.y, -125.0f);
  const float2 magic = make_float2(12582912.0f, 12582912.0f);
  const float2 t = float2_add(x, magic);
  const float2 j = float2_add(t, make_float2(-12582912.0f, -12582912.0f));
  const float2 f = float2_fma(j, make_float2(-1.0f, -1.0f), x);
#if PT_EMU_DEG == 2
  // least-squares degree 2 on [-0.5, 0.5] (relative weight): max rel. error 2.7e-3, mean bias -1e-6
  float2 p = float2_fma(f, make_float2(0.23734999f, 0.23734999f), make_float2(0.70128021f, 0.70128021f));
  p = float2_fma(p, f, make_float2(1.00035157f, 1.00035157f));
#else
  float2 p = float2_fma(f, make_float2(0.0555041087f, 0.0555041087f), make_float2(0.2402265070f, 0.2402265070f));
  p = float2_fma(p, f, make_float2(0.6931471806f, 0.6931471806f));
  p = float2_fma(p, f, make_float2(1.0f, 1.0f));
#endif
  return make_float2(__int_as_float(__float_as_int(p.x) + (__float_as_int(t.x) << 23)),
                     __int_as_float(__float_as_int(p.y) + (__float_as_int(t.y) << 23)));
}

// QK^T of one block from one issuer: NoPE K-steps [0, KA) for issuer A, [KA, 16) plus RoPE for B.
template <bool WITH_ROPE>
__device__ __forceinline__ void issue_qk(SharedStorage& ss, uint32_t tmem_base, int st, int pb, const uint8_t* q_rope) {
  const uint32_t tmem_p = tmem_base + (WITH_ROPE ? tmem_cols::P_B : tmem_cols::P_A);  // PT_P1BUF: single buffer
  (void)pb;
  constexpr int s0 = WITH_ROPE ? KA : 0;
  constexpr int s1 = WITH_ROPE ? D_NOPE / 32 : KA;
  CUTE_UNROLL
  for (int s = s0; s < s1; ++s) {
    mma_ws_ts_f8(
        tmem_p,
        tmem_base + tmem_cols::Q_NOPE + s * 8,
        make_smem_desc(
            ss.k_nope[st] + (s / 4) * (B_TOPK * 128) + (s % 4) * 32, cute::UMMA::LayoutType::SWIZZLE_128B, 16, 1024),
        IDESC_QK,
        s > s0 ? 1u : 0u);
  }
  if constexpr (WITH_ROPE) {
    CUTE_UNROLL
    for (int s = 0; s < D_ROPE / 32; ++s) {
      mma_ws_ss_f8(
          tmem_p,
          make_smem_desc(q_rope + s * 32, cute::UMMA::LayoutType::SWIZZLE_64B, 16, 512),
          make_smem_desc(ss.k_rope[st] + s * 32, cute::UMMA::LayoutType::SWIZZLE_64B, 16, 512),
          IDESC_QK_ROPE,
          (s0 < s1 || s > 0) ? 1u : 0u);  // PT: without a nope step before it, the first rope step initializes P_B
    }
  }
}

// Split merge of one (query token, head) row: lane l combines dims [16 l, 16 l + 16) of the partials
// by their natural-log lse; the sink enters the denominator once, here.
template <int NUM_SPLITS, bool HAVE_ATTN_SINK>
__device__ __forceinline__ void merge_row(const SparseMlaQ8Kv8Sm100Params& params, int64_t row, int lane) {
  const int64_t rows = int64_t(params.s_q) * params.h_q;
  // The partial rows do not depend on the lse: load both in one round trip (+1-5% on split decode).
  uint4 pre[NUM_SPLITS][2];
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    const uint4* src = reinterpret_cast<const uint4*>(params.out + (s * rows + row) * D_V + lane * 16);
    pre[s][0] = src[0];
    pre[s][1] = src[1];
  }
  float l[NUM_SPLITS];
  float m = -CUDART_INF_F, ml = -CUDART_INF_F;
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    l[s] = params.lse[s * rows + row];
    m = fmaxf(m, l[s]);
    ml = fmaxf(ml, params.max_logits[s * rows + row]);
  }
  uint4* dst = reinterpret_cast<uint4*>(params.merged_out + row * D_V + lane * 16);
  if (m == -CUDART_INF_F) {
    dst[0] = dst[1] = make_uint4(0, 0, 0, 0);
    if (lane == 0) {
      params.merged_lse[row] = CUDART_INF_F;
      params.merged_max_logits[row] = -CUDART_INF_F;
    }
    return;
  }
  float w[NUM_SPLITS];
  float sum = 0.0f;
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    w[s] = __expf(l[s] - m);
    sum += w[s];
  }
  float denom = sum;
  if constexpr (HAVE_ATTN_SINK) {
    denom += __expf(__ldg(params.attn_sink + row % params.h_q) - m);
  }
  float acc[16];
  CUTE_UNROLL
  for (int i = 0; i < 16; ++i)
    acc[i] = 0.0f;
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    const float ws = w[s] / denom;
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const uint4 raw = pre[s][j];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        acc[j * 8 + 2 * i] = fmaf(ws, f.x, acc[j * 8 + 2 * i]);
        acc[j * 8 + 2 * i + 1] = fmaf(ws, f.y, acc[j * 8 + 2 * i + 1]);
      }
    }
  }
  CUTE_UNROLL
  for (int j = 0; j < 2; ++j) {
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      h[i] = __floats2bfloat162_rn(acc[j * 8 + 2 * i], acc[j * 8 + 2 * i + 1]);
    }
    dst[j] = raw;
  }
  if (lane == 0) {
    params.merged_lse[row] = m + logf(sum);
    params.merged_max_logits[row] = ml;
  }
}

// Split merge as its own kernel: one warp per row.
template <int NUM_SPLITS, bool HAVE_ATTN_SINK>
__global__ void __launch_bounds__(256) merge_splits_kernel(__grid_constant__ const SparseMlaQ8Kv8Sm100Params params) {
  const int64_t row = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
  if (row < int64_t(params.s_q) * params.h_q) {
    merge_row<NUM_SPLITS, HAVE_ATTN_SINK>(params, row, threadIdx.x % 32);
  }
}

// Item epilogue for TMEM lane quadrant `tmem_lane`: O * out_scale as BF16; out_scale 0 writes zeros.
__device__ __forceinline__ void store_o(uint32_t tmem_lane, bf16* g_out, int half, float out_scale) {
  CUTE_UNROLL
  for (int nh = 0; nh < 2; ++nh) {
    CUTE_UNROLL
    for (int c = 0; c < 4; ++c) {
      float o[32];
      tmem_ld_32x32b_x32(tmem_lane + tmem_cols::O + nh * 128 + c * 32, o);
      tmem_wait_ld();
      uint32_t packed[16];
      CUTE_UNROLL
      for (int i = 0; i < 16; ++i) {
        const float x = out_scale != 0.0f ? o[2 * i] * out_scale : 0.0f;
        const float y = out_scale != 0.0f ? o[2 * i + 1] * out_scale : 0.0f;
        const __nv_bfloat162 h2 = __floats2bfloat162_rn(x, y);
        packed[i] = *reinterpret_cast<const uint32_t*>(&h2);
      }
      // .ws N=256 half nh: lanes 0-63 hold dims [256 nh, +128), lanes 64-127 the next 128.
      bf16* dst = g_out + nh * 256 + half * 128 + c * 32;
      stg_256(dst, packed);
      stg_256(dst + 16, packed + 8);
    }
  }
}

template <bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
__global__ void __launch_bounds__(NUM_THREADS, 1) sparse_mla_q8kv8_sm100_kernel(
    __grid_constant__ const SparseMlaQ8Kv8Sm100Params params, __grid_constant__ const TmaMaps maps) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100)) || \
    (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
  const int num_items = params.s_q * (params.h_q / B_H) * params.num_splits;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_idx = threadIdx.x % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

  if (threadIdx.x == 0) {
    // Swizzled tiles need a 1024B-aligned base.
    if ((cute::cast_smem_ptr_to_uint(smem_raw) & 1023u) != 0) {
      asm volatile("trap;");
    }
    ss.bar_q_full[0].init(1);
    ss.bar_q_full[1].init(1);
    ss.bar_q_free.init(1);
    ss.bar_o_free.init(4);
    CUTE_UNROLL
    for (int i = 0; i < NUM_STAGES; ++i) {
      ss.bar_kv_full[i].init(4);
      ss.bar_sv_done[i].init(1);
      ss.bar_k_valid_ready[i].init(PT_MASK32 ? 1 : B_TOPK / 8);
      ss.bar_k_valid_free[i].init(4);
    }
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i) {
      ss.bar_qk_done[i].init(2);  // one commit per issuer
      ss.bar_p_free[i].init(4);
      ss.bar_s_ready[i].init(4);
    }
    CUTE_UNROLL
    for (int i = 0; i < 3; ++i) {
      ss.bar_epi_ready[i].init(4);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == MMA_A_WARP) {
    cute::TMEM::Allocator1Sm().allocate(512, &ss.tmem_base);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  if (warp_idx == Q_LOAD_WARP && lane_idx == 0) {
    cute::prefetch_tma_descriptor(&maps.q);
    cute::prefetch_tma_descriptor(&maps.q_rope);
    cute::prefetch_tma_descriptor(&maps.kv);
    cute::prefetch_tma_descriptor(&maps.kv_rope);
    if (!V2_SF_LDG) cute::prefetch_tma_descriptor(&maps.kv_sf);
  }
  __syncthreads();
  tcgen05_after_thread_sync();
  const uint32_t tmem_base = ss.tmem_base;
  // Everything above only touched this CTA; inputs written by the previous kernel are read below.
  device::PDLWaitPrimary<true>();

  // Every role walks the same item sequence; g counts 64-token blocks across items.
  if (warp_idx < 4) {
    // Thread t owns row t % 64 and token half t / 64, i.e. exactly TMEM lane t of P and O.
    const int t = threadIdx.x;
    const int row = t & 63;
    const int half = t >> 6;
    const uint32_t tmem_lane = tmem_base + (uint32_t(warp_idx * 32) << 16);
    // TMEM logits are raw e4m3 sums; this maps them straight to the log2 domain.
    const float scale_log2 = params.scale_log2;
    const float2 scale2 = make_float2(scale_log2, scale_log2);
    const uint32_t pair_bar = NamedBarriers::pair_sync_base + (warp_idx & 1);
    const bool partial = params.num_splits > 1;

    int g = 0, it = 0;
    for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
      const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
      const int head = w.head_block * B_H + row;
      const int64_t o_row = (int64_t(w.split) * params.s_q + w.s_q_idx) * params.h_q + head;
      float mi = MAX_INIT_VAL;  // reference max of the stored P (log2 domain), updated lazily
      float li = 0.0f;          // sum of 2^(x - mi + P_SHIFT) over this thread's tokens
      float real_mi = -CUDART_INF_F;

      CUTE_NO_UNROLL
      for (int k = 0; k < w.num_blocks; ++k, ++g) {
        const int pb = g & 1;
        const int st = g % NUM_STAGES;
#if FP4_READ_SF && V2_RT_SMEM
        // v2: r_t and the mask are ready once the mask warp has seen this stage's TMA, which is normally
        // well before QK^T lands, so their shared-memory loads go out before the qk_done wait and their
        // latency hides behind it (issued after qk_done they cost ~5% end to end).
        ss.bar_k_valid_ready[st].wait((g / NUM_STAGES) & 1);
        const uint32_t valid = *reinterpret_cast<const uint32_t*>(ss.is_k_valid[st] + half * 4);
        // v2: the mask warp already turned this stage's exponents into packed floats; `half` is uniform
        // per warp, so these are 8 broadcast loads. They must complete before k_valid_free is arrived
        // (the mask warp rewrites kv_rt[st] after it), exactly like `valid` above.
        float rt[32];
#if V2_ABL_RT_CONST
        // timing ablation: apply without the 8 loads (one value read per block, not foldable)
        {
          const float c = ss.kv_rt[st][half * 32];
          CUTE_UNROLL
          for (int i = 0; i < 32; ++i) rt[i] = c;
        }
#else
        {
          const float4* r4 = reinterpret_cast<const float4*>(ss.kv_rt[st] + half * 32);
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) {
            const float4 x = r4[i];
            rt[4 * i] = x.x;
            rt[4 * i + 1] = x.y;
            rt[4 * i + 2] = x.z;
            rt[4 * i + 3] = x.w;
          }
        }
#endif
        ss.bar_qk_done[pb].wait((g >> 1) & 1);
        tcgen05_after_thread_sync();
        float p[32], pr[32];
        tmem_ld_32x32b_x32(tmem_lane + tmem_cols::P_A, p);  // PT_P1BUF
        tmem_ld_32x32b_x32(tmem_lane + tmem_cols::P_B, pr);  // PT_P1BUF
#else
        ss.bar_qk_done[pb].wait((g >> 1) & 1);
        ss.bar_k_valid_ready[st].wait((g / NUM_STAGES) & 1);
        tcgen05_after_thread_sync();

        float p[32], pr[32];
        tmem_ld_32x32b_x32(tmem_lane + tmem_cols::P_A, p);  // PT_P1BUF
        tmem_ld_32x32b_x32(tmem_lane + tmem_cols::P_B, pr);  // PT_P1BUF
        // Shared-memory read overlaps the TMEM loads.
        const uint32_t valid = *reinterpret_cast<const uint32_t*>(ss.is_k_valid[st] + half * 4);
#if FP4_READ_SF
        // Token `half * 32 + lane_idx` (this lane's own token) sits at byte 0 of its 16 B window, i.e. at
        // (token / 4) * 128 + (token % 4) * 16; every lane needs all 32 exponents, so one load per lane
        // plus a warp broadcast.
        const int my_tok = half * 32 + lane_idx;
        const uint32_t my_sf = ss.kv4_sf[st][(my_tok / 4) * 128 + (my_tok % 4) * 16];
#endif
#endif
        tmem_wait_ld();
        tcgen05_before_thread_sync();
        // One arrival per warp: wait::ld is warp-collective and __syncwarp orders the lanes.
        __syncwarp();
        if (lane_idx == 0) {
          ss.bar_p_free[pb].arrive();
          ss.bar_k_valid_free[st].arrive();
        }

        // r_t = 2^k of this thread's 32 tokens (PV's K dim), exactly a power of two: the ue8m0 byte is
        // already k + 127, so its float bit pattern is (k + 127) << 23. Both halves of the row are stored
        // pre-scaled by 2^-k (e2m1 for NoPE, e4m3 for RoPE), so this one factor restores all 576 dims.
#if FP4_READ_SF && !V2_RT_SMEM
        float rt[32];
        CUTE_UNROLL
        for (int i = 0; i < 32; ++i) {
          rt[i] = __int_as_float(__shfl_sync(0xffffffffu, my_sf, i) << 23);
        }
#endif
        CUTE_UNROLL
        for (int i = 0; i < 16; ++i) {
          // Both issuers' logits go through the same E2M1 operand, so the raw sum is 2^-k times the real
          // dot product: fold r_t back in before the max / exp (this is QK's N dim = the token dim).
#if FP4_APPLY_SF
          // PT: only the nope logits carry the per-token scale; the rope logits (P_B) are unscaled
          const float2 v = float2_fma(make_float2(p[2 * i], p[2 * i + 1]), make_float2(rt[2 * i], rt[2 * i + 1]),
                                      make_float2(pr[2 * i], pr[2 * i + 1]));
#else
          const float2 v = float2_add(make_float2(p[2 * i], p[2 * i + 1]), make_float2(pr[2 * i], pr[2 * i + 1]));
#endif
          p[2 * i] = v.x;
          p[2 * i + 1] = v.y;
        }
        // Fully valid blocks (the common case) skip the per-element selects.
        if (!__all_sync(0xffffffff, valid == 0xffffffffu)) {
          CUTE_UNROLL
          for (int i = 0; i < 32; ++i) {
            p[i] = (valid >> i & 1u) ? p[i] : -CUDART_INF_F;
          }
        }
        float cur_max = -CUDART_INF_F;
        CUTE_UNROLL
        for (int i = 0; i < 32; i += 4) {
          cur_max = fmaxf(cur_max, fmaxf(fmaxf(p[i], p[i + 1]), fmaxf(p[i + 2], p[i + 3])));
        }
        cur_max *= scale_log2;
        ss.max_buf[g & 1][t] = cur_max;
        NamedBarrier::arrive_and_wait(64, pair_bar);
        cur_max = fmaxf(cur_max, ss.max_buf[g & 1][t ^ 64]);
        real_mi = fmaxf(real_mi, cur_max);
        // Warps w and w^2 see the same rows and maxima, so both O lanes of a row agree.
        const bool need_rescale = __any_sync(0xffffffff, cur_max - mi > RESCALE_THRESHOLD);
        float scale_old = 1.0f;
        if (need_rescale) {
          const float new_max = fmaxf(cur_max, mi);
          scale_old = exp2f(mi - new_max);
          mi = new_max;
        }

        // P = 2^(x - mi + P_SHIFT).
        const float bias = P_SHIFT - mi - params.pt_eL;  // PT: stored P = d * r_t
        const float2 neg_bias = make_float2(bias, bias);
        float2 sum2 = make_float2(0.0f, 0.0f);
        uint32_t packed[8];
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
          float2 d0 = float2_fma(make_float2(p[4 * i], p[4 * i + 1]), scale2, neg_bias);
          float2 d1 = float2_fma(make_float2(p[4 * i + 2], p[4 * i + 3]), scale2, neg_bias);
          if (PT_EMU_MASK ? ((PT_EMU_MASK >> i) & 1) : ((i % PT_EMU_R) == PT_EMU_R - 1)) {  // PT step 11: compile-time
            d0 = pt_exp2_emu(d0);
            d1 = pt_exp2_emu(d1);
          } else {
            d0.x = exp2f(d0.x);
            d0.y = exp2f(d0.y);
            d1.x = exp2f(d1.x);
            d1.y = exp2f(d1.y);
          }
          // The running sum stays the true softmax denominator; only the stored operand carries r_t,
          // because PV's B operand is e2m1 = v * 2^-k as well, so P must be P * 2^k.
          sum2 = float2_add(sum2, float2_add(d0, d1));
#if FP4_APPLY_SF
          const float2 s0 = float2_mul(d0, make_float2(rt[4 * i], rt[4 * i + 1]));
          const float2 s1 = float2_mul(d1, make_float2(rt[4 * i + 2], rt[4 * i + 3]));
#else
          const float2 s0 = d0;
          const float2 s1 = d1;
#endif
          const uint32_t lo = __nv_cvt_float2_to_fp8x2(s0, __NV_SATFINITE, __NV_E4M3);
          const uint32_t hi = __nv_cvt_float2_to_fp8x2(s1, __NV_SATFINITE, __NV_E4M3);
          packed[i] = lo | (hi << 16);
        }
        li = fmaf(li, scale_old, sum2.x + sum2.y);

        // S[pb] was last read by PV(g - 2).
        // PT step 18: no wait for PV(g - 2) here -- issuer A's commit to qk_done(g), waited above, was issued after
        // PV(g - 2) and tracks all of A's prior tcgen05 operations, so S[g & 1] is already free.
        *reinterpret_cast<uint4*>(ss.s[pb] + sw64_offset(row, half * 2)) =
            make_uint4(packed[0], packed[1], packed[2], packed[3]);
        *reinterpret_cast<uint4*>(ss.s[pb] + sw64_offset(row, half * 2 + 1)) =
            make_uint4(packed[4], packed[5], packed[6], packed[7]);

        if (k > 0 && need_rescale) {
          // PV(g - 1) must land before its accumulator is rescaled; PV(g) waits for s_ready.
          ss.bar_sv_done[(g - 1) % NUM_STAGES].wait(((g - 1) / NUM_STAGES) & 1);
          tcgen05_after_thread_sync();
          const float2 s2 = make_float2(scale_old, scale_old);
          CUTE_UNROLL
          for (int c = 0; c < (D_V / 2) / 32; ++c) {
            float o[32];
            tmem_ld_32x32b_x32(tmem_lane + tmem_cols::O + c * 32, o);
            tmem_wait_ld();
            CUTE_UNROLL
            for (int i = 0; i < 16; ++i) {
              const float2 v = float2_mul(make_float2(o[2 * i], o[2 * i + 1]), s2);
              o[2 * i] = v.x;
              o[2 * i + 1] = v.y;
            }
            tmem_st_32x32b_x32(tmem_lane + tmem_cols::O + c * 32, o);
          }
          tmem_wait_st();
          tcgen05_before_thread_sync();
        }
        fence_view_async_shared();
        __syncwarp();
        if (lane_idx == 0) {
          ss.bar_s_ready[pb].arrive();
        }
      }

      ss.sum_buf[t] = li;
      NamedBarrier::arrive_and_wait(64, pair_bar);
      const float li_total = li + ss.sum_buf[t ^ 64];
      // real_mi - mi <= RESCALE_THRESHOLD, so any valid token makes li_total >= 1.
      const bool has_valid = real_mi != -CUDART_INF_F;
      if (half == 0) {
        params.max_logits[o_row] = has_valid ? real_mi * CUDART_LN2_F : -CUDART_INF_F;
        // Empty partials get weight 0 in the split merge; an empty final row reports +inf.
        params.lse[o_row] =
            has_valid ? logf(li_total) + (mi - P_SHIFT + params.pt_eL) * CUDART_LN2_F : (partial ? -CUDART_INF_F : CUDART_INF_F);
      }
      float out_scale = 0.0f;
      if (has_valid) {
        float denom = li_total;
        if (HAVE_ATTN_SINK && !partial) {
          // +inf sink zeroes the row, -inf sink is a no-op; split partials get it in the merge.
          denom += exp2f(__ldg(params.attn_sink + head) * CUDART_L2E_F - (mi - P_SHIFT + params.pt_eL));
        }
        out_scale = params.bmm2_scale / denom;
      }
      ss.epi_scale[it % 3][t] = out_scale;
      __syncwarp();
      if (lane_idx == 0) {
        ss.bar_epi_ready[it % 3].arrive();
      }
    }
  } else if (warp_idx >= EPI_WARP0) {
    // Epilogue warpgroup: thread e owns TMEM lane e, like softmax thread e.
    const int e = threadIdx.x - EPI_WARP0 * 32;
    const uint32_t tmem_lane = tmem_base + (uint32_t((warp_idx - EPI_WARP0) * 32) << 16);
    int g = 0, it = 0;
    for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
      const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
      g += w.num_blocks;
      const int64_t o_row = (int64_t(w.split) * params.s_q + w.s_q_idx) * params.h_q + w.head_block * B_H + (e & 63);
      ss.bar_epi_ready[it % 3].wait((it / 3) & 1);
      const int g_last = g - 1;
      ss.bar_sv_done[g_last % NUM_STAGES].wait((g_last / NUM_STAGES) & 1);
      tcgen05_after_thread_sync();
      store_o(tmem_lane, params.out + o_row * D_V, e >> 6, ss.epi_scale[it % 3][e]);
      tcgen05_before_thread_sync();
      __syncwarp();
      if (lane_idx == 0) {
        ss.bar_o_free.arrive();
      }
    }
  } else if (warp_idx < 8) {
    // Producer warp pw gathers row groups {pw, pw + 4, pw + 8, pw + 12} (4 tokens each).
    const int pw = warp_idx - 4;
    if (cute::elect_one_sync()) {
      const auto in_range = [&](int idx) { return uint32_t(idx) < uint32_t(params.s_kv); };
      int g = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        const int* g_indices = item_indices(params, w);
        int4 rows[4];
        CUTE_UNROLL
        for (int r = 0; r < 4; ++r) {
          rows[r] = __ldg(reinterpret_cast<const int4*>(g_indices) + pw + 4 * r);
        }
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          bool issue[4];
          int num_issue = 0;
          CUTE_UNROLL
          for (int r = 0; r < 4; ++r) {
            // After its first fill a stage only holds finite data, so masked groups may keep stale rows.
            issue[r] = g < NUM_STAGES || in_range(rows[r].x) || in_range(rows[r].y) || in_range(rows[r].z) ||
                       in_range(rows[r].w);
            num_issue += issue[r] ? 1 : 0;
          }
          ss.bar_sv_done[st].wait(((g / NUM_STAGES) & 1) ^ 1);
          ss.bar_kv_full[st].arrive_and_expect_tx(num_issue * KV_BYTES_PER_ROW_GROUP);
          CUTE_UNROLL
          for (int r = 0; r < 4; ++r) {
            if (!issue[r]) continue;
            const int rg = pw + 4 * r;
            CUTE_UNROLL
            for (int c = 0; c < D_NOPE / 128; ++c) {
              tma_gather4(
                  &maps.kv,
                  ss.bar_kv_full[st],
                  ss.k_nope[st] + c * (B_TOPK * 128) + rg * 512,
                  c * 128,
                  rows[r],
                  int64_t(cute::TMA::CacheHintSm90::EVICT_LAST));
            }
            tma_gather4(
                &maps.kv_rope,
                ss.bar_kv_full[st],
                ss.k_rope[st] + rg * 256,
                0,
                rows[r],
                int64_t(cute::TMA::CacheHintSm90::EVICT_LAST));
#if FP4_STAGE_SF
            // The exponent of a token rides in its own 16 B window of the rope row, so the KV gather also
            // brings the scale in: no scattered per-token load anywhere in the kernel.
            tma_gather4(
                &maps.kv_sf,
                ss.bar_kv_full[st],
                ss.kv4_sf[st] + rg * 128,
                KV4_SF_OFFSET,
                rows[r],
                int64_t(cute::TMA::CacheHintSm90::EVICT_LAST));
#endif
          }
          if (k + 1 < w.num_blocks) {
            CUTE_UNROLL
            for (int r = 0; r < 4; ++r) {
              rows[r] = __ldg(reinterpret_cast<const int4*>(g_indices + (k + 1) * B_TOPK) + pw + 4 * r);
            }
          }
        }
      }
    }
  } else if (warp_idx == MMA_A_WARP) {
    if (cute::elect_one_sync()) {
      // PV(gp) of block gp; the first block of item it_p overwrites O once item it_p - 1 has left it.
      const auto issue_pv = [&](int gp, bool first_of_item, int it_p) {
        const int st = gp % NUM_STAGES;
        const int sb = gp & 1;
        ss.bar_s_ready[sb].wait((gp >> 1) & 1);
        if (first_of_item && it_p > 0) {
          ss.bar_o_free.wait((it_p - 1) & 1);
        }
        tcgen05_after_thread_sync();
        CUTE_UNROLL
        for (int nh = 0; nh < 2; ++nh) {
          CUTE_UNROLL
          for (int s = 0; s < B_TOPK / 32; ++s) {
            // V = the K NoPE tile read MN-major: 128-dim atoms 8KB apart, 8-token atoms 1KB apart.
            mma_ws_ss_f8(
                tmem_base + tmem_cols::O + nh * 128,
                make_smem_desc(ss.s[sb] + s * 32, cute::UMMA::LayoutType::SWIZZLE_64B, 16, 512),
                make_smem_desc(
                    ss.k_nope[st] + nh * (2 * B_TOPK * 128) + s * (32 * 128),
                    cute::UMMA::LayoutType::SWIZZLE_128B,
                    B_TOPK * 128,
                    1024),
                IDESC_PV,
                (!first_of_item || s > 0) ? 1u : 0u);
          }
        }
        umma_commit(ss.bar_sv_done[st]);
      };
      // One flat block sequence across items: QK^T(g) is issued before PV(g - 1), also when g starts
      // a new item, so the next item's UTCCP and first QK^T do not wait for the last softmax.
      int g = 0;
      int it = 0;
      int pend_g = -1, pend_it = 0;
      bool pend_first = false;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        if (g > 0) {
          // Both issuers' QK^T of the previous block (the previous item's last) must be done before the
          // TMEM copy of Q is replaced.
          ss.bar_qk_done[(g - 1) & 1].wait(((g - 1) >> 1) & 1);
        }
        ss.bar_q_full[it & 1].wait((it >> 1) & 1);
        tcgen05_after_thread_sync();
        // 16B per row per copy; warpx2::02_13 duplicates the 64 rows into both lane halves.
        CUTE_UNROLL
        for (int i = 0; i < D_NOPE / 16; ++i) {
          utccp_64x128b_warpx2_0213(
              tmem_base + tmem_cols::Q_NOPE + i * 4,
              make_smem_desc(
                  ss.q_nope + (i / 8) * (B_H * 128) + (i % 8) * 16, cute::UMMA::LayoutType::SWIZZLE_128B, 16, 1024));
        }
        // Releases the Q NoPE staging buffer and tells issuer B that TMEM Q is ready.
        umma_commit(ss.bar_q_free);

        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          const int pb = g & 1;
          ss.bar_kv_full[st].wait((g / NUM_STAGES) & 1);
          if (g > 0) ss.bar_p_free[(g - 1) & 1].wait(((g - 1) >> 1) & 1);  // PT_P1BUF: softmax(g-1) loaded P
          tcgen05_after_thread_sync();
          issue_qk<false>(ss, tmem_base, st, pb, nullptr);
          umma_commit(ss.bar_qk_done[pb]);
          if (pend_g >= 0) {
            issue_pv(pend_g, pend_first, pend_it);
          }
          pend_g = g;
          pend_first = k == 0;
          pend_it = it;
        }
      }
      if (pend_g >= 0) {
        issue_pv(pend_g, pend_first, pend_it);
      }
    }
  } else if (warp_idx == MMA_B_WARP) {
    if (cute::elect_one_sync()) {
      int g = 0;
      int it = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        ss.bar_q_full[it & 1].wait((it >> 1) & 1);  // Q RoPE landed
        ss.bar_q_free.wait(it & 1);                 // issuer A's UTCCP of Q NoPE landed
        const uint8_t* q_rope = ss.q_rope[it & 1];
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k) {
          const int gq = g + k;
          const int st = gq % NUM_STAGES;
          const int pb = gq & 1;
          ss.bar_kv_full[st].wait((gq / NUM_STAGES) & 1);
          if (gq > 0) ss.bar_p_free[(gq - 1) & 1].wait(((gq - 1) >> 1) & 1);  // PT_P1BUF
          tcgen05_after_thread_sync();
          issue_qk<true>(ss, tmem_base, st, pb, q_rope);
          umma_commit(ss.bar_qk_done[pb]);
        }
        g += w.num_blocks;
      }
    }
  } else if (warp_idx == MASK_WARP) {
#if V2_RT_SMEM
#if PT_MASK32
    static_assert(FP4_LDG_SF, "PT_MASK32 needs the dense r_t array (V2_SF_LDG)");
    {
#ifndef V2_MASK_LA
#define V2_MASK_LA 3  // PT step 6
#endif
      constexpr int LA = V2_MASK_LA;
      int la_item = blockIdx.x, la_k = 0;
      WorkItem la_w{};
      if (la_item < num_items) la_w = get_item<HAVE_TOPK_LENGTH>(params, la_item);
      struct Ent { int a, b, pos, len; uint32_t sa, sb; };
      const auto load_sf = [&](Ent& e) {
        e.sa = uint32_t(e.a) < uint32_t(params.s_kv) ? uint32_t(__ldg(params.kv4_sf + e.a)) : 0x38u;  // e4m3 1.0
        e.sb = uint32_t(e.b) < uint32_t(params.s_kv) ? uint32_t(__ldg(params.kv4_sf + e.b)) : 0x38u;
      };
      const auto load_next = [&](Ent& e) {
        if (la_item >= num_items) return;
        const int* gi = item_indices(params, la_w);
        e.pos = la_k * B_TOPK;
        e.len = la_w.topk_length;
        e.a = __ldg(gi + e.pos + lane_idx);
        e.b = __ldg(gi + e.pos + 32 + lane_idx);
        if (++la_k == la_w.num_blocks) {
          la_k = 0;
          la_item += gridDim.x;
          if (la_item < num_items) la_w = get_item<HAVE_TOPK_LENGTH>(params, la_item);
        }
      };
      Ent ring[LA];
      CUTE_UNROLL
      for (int i = 0; i < LA; ++i) {
        ring[i] = Ent{};
        load_next(ring[i]);
      }
      CUTE_UNROLL
      for (int i = 0; i + 1 < LA; ++i) load_sf(ring[i]);
      const float pt_mul = exp2f(params.pt_eL);
      int g = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          const Ent e0 = ring[0];
          const bool oka = uint32_t(e0.a) < uint32_t(params.s_kv) && e0.pos + lane_idx < e0.len;
          const bool okb = uint32_t(e0.b) < uint32_t(params.s_kv) && e0.pos + 32 + lane_idx < e0.len;
          const uint32_t ma = __ballot_sync(0xffffffffu, oka);
          const uint32_t mb = __ballot_sync(0xffffffffu, okb);
          CUTE_UNROLL
          for (int i = 0; i + 1 < LA; ++i) ring[i] = ring[i + 1];
          load_next(ring[LA - 1]);
          load_sf(ring[LA - 2]);
          const float ra = pt_e4m3_to_float(uint8_t(e0.sa)) * pt_mul;
          const float rb = pt_e4m3_to_float(uint8_t(e0.sb)) * pt_mul;
          ss.bar_k_valid_free[st].wait(((g / NUM_STAGES) & 1) ^ 1);
          if (lane_idx == 0) {
            reinterpret_cast<uint32_t*>(ss.is_k_valid[st])[0] = ma;
            reinterpret_cast<uint32_t*>(ss.is_k_valid[st])[1] = mb;
          }
          ss.kv_rt[st][lane_idx] = ra;
          ss.kv_rt[st][32 + lane_idx] = rb;
          __syncwarp();
          if (lane_idx == 0) ss.bar_k_valid_ready[st].arrive();
        }
      }
    }
#else
    // v2: the mask warp also produces r_t, which makes it wait for each stage's TMA. Its index loads
    // must then not sit behind that wait (a DRAM round trip per block puts the warp on the critical
    // path: 0.70 -> 0.85 ms), so the indices of the next MASK_LA blocks are loaded ahead, across item
    // boundaries, into a register ring.
    if (lane_idx < B_TOPK / 8) {
#ifndef V2_MASK_LA
#define V2_MASK_LA 3  // PT step 6
#endif
      constexpr int MASK_LA = V2_MASK_LA;
      int la_item = blockIdx.x, la_k = 0;
      WorkItem la_w{};
      if (la_item < num_items) la_w = get_item<HAVE_TOPK_LENGTH>(params, la_item);
      // One lookahead entry: 8 indices, the position of the first and the item's valid length.
      struct Ent { int4 a, b; int pos, len; uint32_t sf[8]; };
#if FP4_LDG_SF
      // ue8m0 of the entry's 8 tokens, straight from the dense array; out-of-range slots are masked anyway.
      const auto load_sf = [&](Ent& e) {
        const int idx[8] = {e.a.x, e.a.y, e.a.z, e.a.w, e.b.x, e.b.y, e.b.z, e.b.w};
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
          e.sf[i] = uint32_t(idx[i]) < uint32_t(params.s_kv) ? uint32_t(__ldg(params.kv4_sf + idx[i])) : 0x38u;  // PT: e4m3 1.0
        }
      };
#endif
      const auto load_next = [&](Ent& e) {
        if (la_item >= num_items) return;
        const int* gi = item_indices(params, la_w);
        e.pos = la_k * B_TOPK + lane_idx * 8;
        e.len = la_w.topk_length;
        e.a = __ldg(reinterpret_cast<const int4*>(gi + e.pos));
        e.b = __ldg(reinterpret_cast<const int4*>(gi + e.pos) + 1);
        if (++la_k == la_w.num_blocks) {
          la_k = 0;
          la_item += gridDim.x;
          if (la_item < num_items) la_w = get_item<HAVE_TOPK_LENGTH>(params, la_item);
        }
      };
      // Register ring (compile-time indices only): ring[0] is block g, ring[MASK_LA - 1] block g + LA - 1.
      Ent ring[MASK_LA];
      CUTE_UNROLL
      for (int i = 0; i < MASK_LA; ++i) {
        ring[i] = Ent{};
        load_next(ring[i]);
      }
#if FP4_LDG_SF
      CUTE_UNROLL
      for (int i = 0; i + 1 < MASK_LA; ++i) {
        load_sf(ring[i]);
      }
#endif
      int g = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          const Ent& e0 = ring[0];
#if FP4_LDG_SF
          uint32_t e0sf[8];
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) e0sf[i] = e0.sf[i];
#endif
          const int idx[8] = {e0.a.x, e0.a.y, e0.a.z, e0.a.w, e0.b.x, e0.b.y, e0.b.z, e0.b.w};
          uint32_t mask = 0;
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) {
            const bool ok = uint32_t(idx[i]) < uint32_t(params.s_kv) && e0.pos + i < e0.len;
            mask |= (ok ? 1u : 0u) << i;
          }
          CUTE_UNROLL
          for (int i = 0; i + 1 < MASK_LA; ++i) {
            ring[i] = ring[i + 1];
          }
          load_next(ring[MASK_LA - 1]);  // block g + LA, in flight during the waits below
#if FP4_LDG_SF
          // Its indices were issued one block ago; its exponents are used LA - 2 blocks from now.
          load_sf(ring[MASK_LA - 2]);
          const float pt_mul_ldg = exp2f(params.pt_eL);
          float r[8];
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) {
            r[i] = pt_e4m3_to_float(uint8_t(e0sf[i])) * pt_mul_ldg;  // PT: e4m3 r_t times 2^eL
          }
#endif
          ss.bar_k_valid_free[st].wait(((g / NUM_STAGES) & 1) ^ 1);
          ss.is_k_valid[st][lane_idx] = uint8_t(mask);
          // r_t of this lane's 8 tokens: the exponents ride the stage's KV gather (row group rg = token / 4
          // owns a 128 B granule, token % 4 picks the 16 B window, exponent in byte 0). The softmax waits
          // on k_valid_ready after qk_done, which already implies this kv_full.
#if !FP4_LDG_SF
          ss.bar_kv_full[st].wait((g / NUM_STAGES) & 1);
#endif
          {
#if !FP4_LDG_SF
            const uint8_t* sf = ss.kv4_sf[st] + lane_idx * 256;
            const float pt_mul = exp2f(params.pt_eL);
            float r[8];
            CUTE_UNROLL
            for (int i = 0; i < 8; ++i) {
              r[i] = pt_e4m3_to_float(sf[(i / 4) * 128 + (i % 4) * 16]) * pt_mul;  // PT: e4m3 r_t times 2^eL
            }
#endif
            float4* dst = reinterpret_cast<float4*>(ss.kv_rt[st] + lane_idx * 8);
            dst[0] = make_float4(r[0], r[1], r[2], r[3]);
            dst[1] = make_float4(r[4], r[5], r[6], r[7]);
          }
          ss.bar_k_valid_ready[st].arrive();
        }
      }
    }
#endif  // PT_MASK32
#else
    if (lane_idx < B_TOPK / 8) {
      int g = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        const int* g_indices = item_indices(params, w);
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          const int pos = k * B_TOPK + lane_idx * 8;
          const int4 a = __ldg(reinterpret_cast<const int4*>(g_indices + pos));
          const int4 b = __ldg(reinterpret_cast<const int4*>(g_indices + pos) + 1);
          const int idx[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
          uint32_t mask = 0;
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) {
            const bool ok = uint32_t(idx[i]) < uint32_t(params.s_kv) && pos + i < w.topk_length;
            mask |= (ok ? 1u : 0u) << i;
          }
          ss.bar_k_valid_free[st].wait(((g / NUM_STAGES) & 1) ^ 1);
          ss.is_k_valid[st][lane_idx] = uint8_t(mask);
          ss.bar_k_valid_ready[st].arrive();
        }
      }
    }
#endif
  } else if (warp_idx == Q_LOAD_WARP) {
    // Q of item i+1 is staged while item i runs. UTCCP(i) frees the NoPE staging; the RoPE staging
    // is an SS operand for the whole item, so it alternates by item, and UTCCP(i) also implies that
    // item i-1 (the previous user of buffer (i+1) % 2) has finished its QK^T.
    if (lane_idx == 0) {
      int it = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        if (it > 0) {
          ss.bar_q_free.wait((it - 1) & 1);
        }
        transac_bar_t& bar = ss.bar_q_full[it & 1];
        bar.arrive_and_expect_tx(B_H * D_QK);
        CUTE_UNROLL
        for (int j = 0; j < D_NOPE / 128; ++j) {
          cute::SM90_TMA_LOAD_3D::copy(
              &maps.q,
              reinterpret_cast<uint64_t*>(&bar),
              uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
              ss.q_nope + j * (B_H * 128),
              j * 128,
              w.head_block * B_H,
              w.s_q_idx);
        }
        cute::SM90_TMA_LOAD_3D::copy(
            &maps.q_rope,
            reinterpret_cast<uint64_t*>(&bar),
            uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
            ss.q_rope[it & 1],
            int(D_NOPE),
            w.head_block * B_H,
            w.s_q_idx);
      }
    }
  }

  // This CTA has issued all its work, so the next kernel may launch; its griddepcontrol.wait still
  // waits for this whole grid to finish.
  device::PDLTriggerSecondary<true>();
  // The epilogue fenced its last tcgen05.ld before this barrier.
  __syncthreads();
  if (warp_idx == MMA_A_WARP) {
    tcgen05_after_thread_sync();
    cute::TMEM::Allocator1Sm().free(tmem_base, 512);
  }
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
  }
#endif
}

inline void encode_tma(
    CUtensorMap* map,
    CUtensorMapDataType dtype,
    uint32_t rank,
    const void* gaddr,
    const uint64_t* dims,
    const uint64_t* strides_bytes,
    const uint32_t* box,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion promotion,
    const char* name) {
  const uint32_t elem_strides[3] = {1, 1, 1};
  const CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      map,
      dtype,
      rank,
      const_cast<void*>(gaddr),
      dims,
      strides_bytes,
      box,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      promotion,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(res == CUDA_SUCCESS) << "sparse_mla_q8kv8_sm100: tensor map encoding failed for " << name << ": "
                                  << int(res);
}

/**
 * \brief Encode the TMA descriptors and launch the main kernel, plus the split merge kernel when
 *        `p.num_splits > 1`.
 *
 * \tparam HAVE_TOPK_LENGTH Whether `p.topk_length` is set.
 * \tparam HAVE_ATTN_SINK   Whether `p.attn_sink` is set.
 * \param p       Launch parameters; `p.stream` is the stream to launch on.
 * \param num_sms SM count of the device, which caps the persistent grid.
 */
template <bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
void run_sparse_mla_q8kv8_sm100(const SparseMlaQ8Kv8Sm100Params& p, int32_t num_sms) {
  CHECK_HOST(p.h_q > 0 && p.h_q % B_H == 0) << "sparse_mla_q8kv8_sm100: h_q must be a multiple of 64, got " << p.h_q;
  CHECK_HOST(p.topk > 0 && p.topk % B_TOPK == 0)
      << "sparse_mla_q8kv8_sm100: topk must be a multiple of 64, got " << p.topk;
  CHECK_HOST(p.s_q > 0 && p.s_kv > 0) << "sparse_mla_q8kv8_sm100: empty problem";
  CHECK_HOST(p.num_splits >= 1 && p.split_len % B_TOPK == 0 && p.split_len * p.num_splits == p.topk)
      << "sparse_mla_q8kv8_sm100: topk must split into num_splits chunks of a multiple of 64";
  CHECK_HOST(
      p.stride_q_h_q % 16 == 0 && p.stride_q_s_q % 16 == 0 && p.stride_indices_s_q % 4 == 0)
      << "sparse_mla_q8kv8_sm100: TMA and vector loads need 16B-aligned row strides";
  // gather4 index loads are int4 and the epilogue stores are 32 B per lane. The fp4 pool rows are fixed
  // strides (256 B of e2m1 + 64 B of e4m3), and 16U4_ALIGN16B wants a 16B-aligned (here: row-aligned)
  // base address.
  CHECK_HOST(
      reinterpret_cast<uintptr_t>(p.q) % 16 == 0 && reinterpret_cast<uintptr_t>(p.kv4) % 128 == 0 &&
      reinterpret_cast<uintptr_t>(p.kv4_rope) % 16 == 0 && reinterpret_cast<uintptr_t>(p.indices) % 16 == 0)
      << "sparse_mla_q8kv8_sm100: q, the fp4 KV pool and indices must be aligned";
  CHECK_HOST(
      reinterpret_cast<uintptr_t>(p.out) % 32 == 0 &&
      (p.num_splits == 1 || reinterpret_cast<uintptr_t>(p.merged_out) % 16 == 0))
      << "sparse_mla_q8kv8_sm100: out must be 32B-aligned";

  TmaMaps maps;
  {
    const uint64_t dims[3] = {uint64_t(D_QK), uint64_t(p.h_q), uint64_t(p.s_q)};
    const uint64_t strides[2] = {uint64_t(p.stride_q_h_q), uint64_t(p.stride_q_s_q)};
    const uint32_t box[3] = {128, B_H, 1};
    encode_tma(
        &maps.q,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        p.q,
        dims,
        strides,
        box,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        "q");
    const uint32_t box_rope[3] = {D_ROPE, B_H, 1};
    encode_tma(
        &maps.q_rope,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        p.q,
        dims,
        strides,
        box_rope,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        "q_rope");
  }
  {
    // NoPE: 512 packed e2m1 elements per row (256 B). ALIGN16B is the only TMA layout whose smem image
    // matches what E2M1 in the 8-bit container expects, and it pads every 16 B slot to 8 B data + 8 B
    // gap, i.e. 1 B per element again, so the gather4 destinations below do not move.
    const uint64_t dims[2] = {uint64_t(D_NOPE), uint64_t(p.s_kv)};
    const uint64_t strides[1] = {uint64_t(KV4_V_ROW_BYTES)};
    const uint32_t box[2] = {128, 1};
    encode_tma(
        &maps.kv,
        ABL_V_U8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        2,
        p.kv4,
        dims,
        strides,
        box,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "kv4");
    // RoPE stays e4m3: 64 dims cannot form an ALIGN16B map (globalDim[0] % 128 != 0) and quantizing it
    // would save nothing anyway. Its row is 80 B -- 64 B of rope then the 16 B window that carries this
    // row's ue8m0 exponent in its first byte -- and *two* maps read that one row, so the KV gather brings
    // the scale along and the kernel never issues a scattered per-token byte load.
    const uint64_t rdims[2] = {uint64_t(KV4_ROPE_ROW_BYTES), uint64_t(p.s_kv)};
    const uint64_t rstrides[1] = {uint64_t(KV4_ROPE_ROW_BYTES)};
    const uint32_t rbox[2] = {D_ROPE, 1};
    encode_tma(
        &maps.kv_rope,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        p.kv4_rope,
        rdims,
        rstrides,
        rbox,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        "kv4_rope");
    // 16 elements at column 64, no swizzle: TMA's minimum transfer granule is 16 B, which is exactly the
    // window the exponent lives in.
    if (!V2_SF_LDG) {
    const uint32_t sbox[2] = {16, 1};
    encode_tma(
        &maps.kv_sf,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        p.kv4_rope,
        rdims,
        rstrides,
        sbox,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        "kv4_sf");
    }
  }

  auto kernel = &sparse_mla_q8kv8_sm100_kernel<HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>;
  constexpr size_t smem_size = sizeof(SharedStorage);
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  const int32_t num_items = p.s_q * (p.h_q / B_H) * p.num_splits;
  // PDL only without splits: with the split merge kernel in the chain it measured 5-8% slower at 32-64
  // query tokens.
  host::LaunchKernel(uint32_t(std::min(num_items, num_sms)), NUM_THREADS, p.stream, smem_size)
      .enable_pdl(p.num_splits == 1)(kernel, p, maps);

  if (p.num_splits > 1) {
    const int64_t rows = int64_t(p.s_q) * p.h_q;
    const uint32_t blocks = uint32_t((rows + 7) / 8);
    if (p.num_splits == 2) {
      host::LaunchKernel(blocks, 256, p.stream)(&merge_splits_kernel<2, HAVE_ATTN_SINK>, p);
    } else if (p.num_splits == 4) {
      host::LaunchKernel(blocks, 256, p.stream)(&merge_splits_kernel<4, HAVE_ATTN_SINK>, p);
    } else if (p.num_splits == 8) {
      host::LaunchKernel(blocks, 256, p.stream)(&merge_splits_kernel<8, HAVE_ATTN_SINK>, p);
    } else {
      SGL_PANIC << "sparse_mla_q8kv8_sm100: num_splits must be 1, 2, 4 or 8, got " << p.num_splits;
    }
  }
}

}  // namespace sglang::q8kv8_sm100
