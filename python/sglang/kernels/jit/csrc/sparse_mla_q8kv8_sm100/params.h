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

#pragma once

#include <cutlass/bfloat16.h>

#include <cstdint>
#include <cuda_runtime.h>

namespace sglang::q8kv8_sm100 {

/**
 * \brief Launch parameters of the SM100 Q8KV8 sparse MLA kernel.
 *
 * All 576 dims (512 NoPE + 64 RoPE) of Q and KV are e4m3 with per-tensor scales, as in TRT-LLM-gen's
 * FP8 sparse MLA: logits = bmm1_scale * (q8 . k8), out = bmm2_scale * softmax(logits) . v8.
 */
struct SparseMlaQ8Kv8Sm100Params {
  int32_t s_q, s_kv, h_q, topk;
  // The top-k of a query is cut into num_splits chunks of split_len (a multiple of 64) that run as
  // separate work items; with num_splits > 1, out / max_logits / lse are [num_splits, s_q, h_q, ...]
  // partials without the sink, merged by a second kernel.
  int32_t num_splits, split_len;
  float scale_log2;  // bmm1_scale * log2(e)
  float bmm2_scale;
  float pt_eL;  // PT: log2 of the pool's reference nope scale

  const uint8_t* __restrict__ q;            // [s_q, h_q, 576] e4m3, strides in bytes
  // fp4 KV pool, the MMA operand itself (no dequantization anywhere): the 512 NoPE dims as e2m1
  // nibbles read by TMA 16U4_ALIGN16B (256 B/row, 1 B per element nominal) and the 64 RoPE dims kept as
  // e4m3 (64 B/row, an ALIGN16B map cannot be 64 elements wide). 336 B/row instead of 576.
  //
  // This is the PER-ROW variant: every token carries its own scale. Both halves of a row are stored
  // pre-scaled by 2^-k_row, so `value = stored * 2^k_row`, and the exponent (k_row + 127) rides the RoPE
  // row in a 16 B window at byte 64 -- the same TMA that fetches the rope fetches the scale, so there is
  // no per-token scattered load anywhere in the kernel. 2^k_row is applied once per token while the row
  // is in registers, and the running softmax max/sum are rescaled when it changes.
  //   kv4       [s_kv, 256] packed e2m1, dims 0-511
  //   kv4_rope  [s_kv,  80] e4m3 rope in bytes 0-63 (pre-scaled), ue8m0 exponent (k + 127) in byte 64,
  //                         bytes 65-79 padding. TMA global strides must be multiples of 16 B, hence
  //                         the 16 B granule for the exponent.
  const uint8_t* __restrict__ kv4;
  // Only read by the V2_SF_LDG=1 ablation (a dense per-row exponent array loaded with LDG instead of
  // riding the RoPE TMA). Off by default and not part of the shipped path; may be nullptr.
  const uint8_t* __restrict__ kv4_sf;
  const uint8_t* __restrict__ kv4_rope;
  const int32_t* __restrict__ indices;      // [s_q, topk]
  const float* __restrict__ attn_sink;      // [h_q] or nullptr
  const int32_t* __restrict__ topk_length;  // [s_q] or nullptr

  int64_t stride_q_s_q, stride_q_h_q;
  int64_t stride_indices_s_q;

  cutlass::bfloat16_t* __restrict__ out;  // [(num_splits,) s_q, h_q, 512], contiguous
  float* __restrict__ max_logits;         // [(num_splits,) s_q, h_q]
  float* __restrict__ lse;                // [(num_splits,) s_q, h_q]

  // Final outputs of the split merge (num_splits > 1 only).
  cutlass::bfloat16_t* __restrict__ merged_out;
  float* __restrict__ merged_max_logits;
  float* __restrict__ merged_lse;

  cudaStream_t stream;
};

}  // namespace sglang::q8kv8_sm100
