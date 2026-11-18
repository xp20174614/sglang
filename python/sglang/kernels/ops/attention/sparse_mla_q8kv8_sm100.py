"""JIT-compiled sparse MLA attention for SM100 (B200) with an fp4 KV pool, per-row scales.

Q is e4m3 on all 576 dims (512 NoPE + 64 RoPE). The KV cache is an **fp4 pool used directly as the MMA
operand** (no dequantization anywhere): the 512 NoPE dims are packed e2m1 nibbles (256 B/row) and the 64
RoPE dims stay e4m3 (64 B/row), followed by that row's ue8m0 exponent in a 16 B window. Both halves are
stored pre-scaled by 2^-k_row, so one per-token exponent restores all 576 dims. **336 B/row instead of
the 576 B e4m3 row (-41.7%)**. ``pack_fp4_pool`` below is a reference packer; production deployments
should produce the pool in their upstream quantization kernel instead of paying for it per call.

Measured on B200 against the e4m3 baseline of PR #35237 (same harness, min-of-40): prefill 4K 0.6533 ms
vs 0.6145, 200K 0.6676 ms vs 0.6994 (-4.55%); ``--check`` rel_l2 1.374%; the pool's own quantization cost
is 9.26% at sigma 0.3. The per-tensor variant of this kernel is faster in every regime measured
(0.6079 ms) and uses less memory (320 B/row); this one keeps a scale per token for robustness on
heterogeneous KV, but note that at 4K it is slower than the e4m3 baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.attention.utils import check_out_buffer
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

D_QK = 576
D_V = 512
_B_H = 64
# Split only long top-k lists, into chunks of at least 256 entries. Measured on B200: a split never
# paid off at topk=512, and at topk=2048 the best was the largest of 8 / 4 / 2 that keeps every work
# item on its own SM (decode batches of 1-64 query tokens).
_MIN_TOPK_TO_SPLIT = 1024
_MIN_SPLIT_LEN = 256


@cache_once
def _jit_sparse_mla_q8kv8_sm100_module() -> Module:
    # The kernel is built on tcgen05 and TMA gather4 and has only been validated on SM100.
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("sparse_mla_q8kv8_sm100 requires an SM100 (B200) GPU")
    return load_jit(
        "sparse_mla_q8kv8_sm100",
        cuda_files=["sparse_mla_q8kv8_sm100/entry.cuh"],
        cuda_wrappers=[("dispatch", "sparse_mla_q8kv8_sm100_dispatch")],
        # --use_fast_math maps the softmax exp2f to MUFU.EX2.
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_dependencies=["cutlass"],
    )


@register_custom_op(
    op_name="sparse_mla_q8kv8_sm100",
    mutates_args=[
        "out",
        "max_logits",
        "lse",
        "partial_out",
        "partial_max_logits",
        "partial_lse",
    ],
)
def _sparse_mla_q8kv8_sm100_op(
    q: torch.Tensor,
    kv4: torch.Tensor,
    kv4_rope: torch.Tensor,
    kv4_sf: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_length: torch.Tensor,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    partial_out: torch.Tensor,
    partial_max_logits: torch.Tensor,
    partial_lse: torch.Tensor,
    s_q: int,
    s_kv: int,
    h_q: int,
    topk: int,
    num_splits: int,
    stride_q_s_q: int,
    stride_q_h_q: int,
    stride_indices_s_q: int,
    bmm1_scale: float,
    bmm2_scale: float,
    pt_eL: float,
    have_attn_sink: int,
    have_topk_length: int,
) -> None:
    _jit_sparse_mla_q8kv8_sm100_module().dispatch(
        q,
        kv4,
        kv4_rope,
        kv4_sf,
        indices,
        attn_sink,
        topk_length,
        out,
        max_logits,
        lse,
        partial_out,
        partial_max_logits,
        partial_lse,
        s_q,
        s_kv,
        h_q,
        topk,
        num_splits,
        stride_q_s_q,
        stride_q_h_q,
        stride_indices_s_q,
        bmm1_scale,
        bmm2_scale,
        pt_eL,
        have_attn_sink,
        have_topk_length,
    )


@torch.no_grad()
def pack_fp4_pool(
    kv: torch.Tensor,
    *,
    k_search: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference packer: e4m3 KV rows -> the fp4 pool the kernel reads. **Correctness over speed.**

    Torch, element-wise reference implementation that makes the kernel usable end to end and documents
    the layout. NOT the production path -- a real deployment should emit the pool from its upstream KV
    quantization kernel, otherwise the packing cost lands in the serving loop.

    Per-row scales: every row gets its own exponent ``k_row``. **Raises if any row needs
    ``k_row > 0``** -- see the k_max check below for why that would silently corrupt results. Both halves of the row are stored divided
    by ``2^k_row`` so one exponent restores all 576 dims (the QK^T of the two halves lands in the same
    accumulator, so they must share the scale); scaling by a power of two is exact in e4m3 for normals.
    The exponent is written as ``k_row + 127`` into byte 64 of the 80 B RoPE row, so it rides the RoPE TMA
    and the kernel never does a scattered per-token scale load.

    ``k_row`` is chosen by measuring the row's actual round-trip error over candidates around the OCP
    round-up ``ceil(log2(amax_row/6))``. This matters: e2m1's smallest step is 0.5, so a smaller ``k``
    leaves fewer small values underflowed to zero, at the price of clipping the largest ones. On the B200
    harness the searched ``k`` beats the round-up in every scenario measured.

    Args:
        kv:       [s_kv, 576], float8_e4m3fn, NoPE then RoPE dims
        k_search: pick each row's k by measured error over {k-2, k-1, k, k+1}

    Returns:
        kv4:      [s_kv, 256], uint8, packed e2m1 (low nibble = even dim)
        kv4_rope: [s_kv, 80], uint8, pre-scaled e4m3 RoPE in bytes 0-63, exponent in byte 64
    """
    _check(kv.dim() == 2 and kv.shape[1] == D_QK, f"kv must be [s_kv, {D_QK}]")
    _check(kv.dtype == torch.float8_e4m3fn, f"kv must be float8_e4m3fn, got {kv.dtype}")

    table = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32, device=kv.device
    )
    nope = kv[:, :D_V].to(torch.float32)
    rope = kv[:, D_V:].to(torch.float32)
    amax = nope.abs().amax(dim=1)

    # OCP round-up per row: the smallest k with 2^k >= amax / 6.
    k0 = torch.ceil(torch.log2((amax / 6.0).clamp_min(1e-30)))
    k0 = torch.where(amax > 0, k0, torch.zeros_like(k0))

    def _sq_error(k: torch.Tensor) -> torch.Tensor:
        inv = torch.pow(2.0, -k).unsqueeze(1)
        rt = torch.pow(2.0, k).unsqueeze(1)
        scaled = nope * inv
        idx = (scaled.abs().unsqueeze(-1) - table).abs().argmin(dim=-1)
        q = table[idx] * scaled.sign() * rt
        return (q - nope).pow(2).sum(dim=1) / nope.pow(2).sum(dim=1).clamp_min(1e-30)

    k = k0
    if k_search:
        cand = torch.stack([k0 - 2, k0 - 1, k0, k0 + 1])  # [4, s_kv]
        errs = torch.stack([_sq_error(c) for c in cand])
        k = cand[errs.argmin(dim=0), torch.arange(cand.shape[1], device=kv.device)]
    k = k.clamp(-10, 10)
    # Hard precondition, checked here rather than only documented: the kernel's PV path multiplies P by
    # 2^k and stores it in e4m3, so it needs 2^k * 2^(P_SHIFT + RESCALE) <= 448, i.e. k <= 0 (this
    # kernel ships P_SHIFT = RESCALE = 4, so the bound is exactly 2^k <= 1.75). A pool packed with k > 0
    # does not fail loudly in the kernel -- it silently saturates P and the error grows with topk
    # (measured on the B200 harness: worst rel_l2 0.168 with k = 4..5 vs 0.0255 with k <= 0).
    # Rows that would need k > 0 are rows that were rescaled up to fill e4m3's +-448 range first; the
    # pool carries its own scale, so the rows must be packed at their NATURAL scale instead.
    k_max = int(k.max()) if k.numel() > 0 else 0
    if k_max > 0:
        raise ValueError(
            "per-row fp4 pool requires k <= 0 (P is stored as e4m3: "
            "2^k * 2^(P_SHIFT+RESCALE) must stay <= 448); got k = "
            f"{k_max}. Rescale the KV rows to their natural scale instead of "
            "e4m3's +-448 range."
        )

    inv = torch.pow(2.0, -k).unsqueeze(1)
    scaled = nope * inv
    idx = (scaled.abs().unsqueeze(-1) - table).abs().argmin(dim=-1).to(torch.uint8)
    idx = idx | ((scaled.sign() < 0).to(torch.uint8) << 3)  # e2m1 sign is the nibble's MSB
    kv4 = (idx[:, 0::2] | (idx[:, 1::2] << 4)).to(torch.uint8).contiguous()

    rows = kv.shape[0]
    kv4_rope = torch.zeros(rows, D_QK - D_V + 16, dtype=torch.uint8, device=kv.device)
    kv4_rope[:, : D_QK - D_V] = (
        (rope * inv).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    # ue8m0: the stored byte is the biased exponent, k + 127
    kv4_rope[:, D_QK - D_V] = (k.to(torch.int32) + 127).to(torch.uint8)
    return kv4, kv4_rope.contiguous()


@torch.no_grad()
def unpack_fp4_pool(kv4: torch.Tensor, kv4_rope: torch.Tensor) -> torch.Tensor:
    """Expand the fp4 pool back to e4m3 the way the kernel reads it (reference/debug only)."""
    table = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32, device=kv4.device
    )
    lo = (kv4 & 0xF).to(torch.long)
    hi = (kv4 >> 4).to(torch.long)

    def _decode(code: torch.Tensor) -> torch.Tensor:
        mag = table[(code & 0x7)]
        return torch.where((code & 0x8) != 0, -mag, mag)

    rt = torch.pow(2.0, kv4_rope[:, D_QK - D_V].to(torch.float32) - 127.0).unsqueeze(1)
    nope = torch.empty(kv4.shape[0], D_V, dtype=torch.float32, device=kv4.device)
    nope[:, 0::2] = _decode(lo)
    nope[:, 1::2] = _decode(hi)
    nope = nope * rt
    rope = kv4_rope[:, : D_QK - D_V].view(torch.float8_e4m3fn).to(torch.float32) * rt
    return torch.cat([nope, rope], dim=1).to(torch.float8_e4m3fn)


@cache_once
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@cache_once
def _placeholder(device: torch.device, dtype: torch.dtype, role: str) -> torch.Tensor:
    # One zero-size tensor per role, so the mutated arguments of the custom op never alias.
    return torch.empty(0, dtype=dtype, device=device)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _choose_num_splits(s_q: int, h_q: int, topk: int, num_sms: int) -> int:
    """Largest of 8 / 4 / 2 that keeps every work item on its own SM, else 1 (no split)."""
    if topk < _MIN_TOPK_TO_SPLIT:
        return 1
    items = s_q * (h_q // _B_H)
    for splits in (8, 4, 2):
        if (
            topk % (64 * splits) == 0
            and topk // splits >= _MIN_SPLIT_LEN
            and items * splits <= num_sms
        ):
            return splits
    return 1


@debug_kernel_api
def sparse_mla_q8kv8_sm100_fwd(
    q: torch.Tensor,
    kv4: torch.Tensor,
    kv4_rope: torch.Tensor,
    indices: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    *,
    pt_eL: float = 0.0,
    kv4_sf: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sparse MLA on SM100 with an fp4 KV pool (per-row scales).

    V is the first 512 dims of each KV row. Indices < 0 or >= s_kv and positions >= topk_length are
    masked. Masked positions with an in-range index still load their row, which must not hold NaN.

    The KV cache is the fp4 pool produced by ``pack_fp4_pool``: packed e2m1 NoPE dims plus pre-scaled
    e4m3 RoPE dims carrying a per-row ue8m0 exponent, 336 B/row instead of 576. The exponent is applied
    inside the kernel per token, so callers keep passing the same ``bmm1_scale`` / ``bmm2_scale`` they
    would for the e4m3 kernel.

    Args:
        q:           [s_q, h_q, 576], float8_e4m3fn, NoPE then RoPE dims
        kv4:         [s_kv, 256], uint8, packed e2m1 nibbles of the 512 NoPE dims
        kv4_rope:    [s_kv, 80], uint8, pre-scaled e4m3 RoPE dims with this row's exponent in byte 64
        indices:     [s_q, topk] or [s_q, 1, topk], int32 token rows of kv
        bmm1_scale:  q_scale * kv_scale * sm_scale
        bmm2_scale:  kv_scale of V
        attn_sink:   optional [h_q], float32
        topk_length: optional [s_q], int32 number of leading valid positions per row
        num_splits:  1, 2, 4 or 8 top-k splits; None chooses from the shapes
        out:         optional [s_q, h_q, 512] bfloat16 buffer for the output
        max_logits:  optional [s_q, h_q] float32 buffer
        lse:         optional [s_q, h_q] float32 buffer

    Returns:
        out:        [s_q, h_q, 512], bfloat16
        max_logits: [s_q, h_q], float32, -inf for rows without valid indices
        lse:        [s_q, h_q], float32, natural log without the sink, +inf for rows without
                    valid indices
    """
    _check(
        q.dim() == 3 and q.shape[2] == D_QK,
        f"q must be [s_q, h_q, {D_QK}], got {tuple(q.shape)}",
    )
    _check(q.dtype == torch.float8_e4m3fn, f"q must be float8_e4m3fn, got {q.dtype}")
    _check(
        kv4.dtype == torch.uint8 and kv4_rope.dtype == torch.uint8,
        f"the fp4 pool must be uint8, got {kv4.dtype} and {kv4_rope.dtype}",
    )
    # [s_kv, 256] packed e2m1 + [s_kv, 80] pre-scaled e4m3 RoPE with the row exponent in byte 64.
    _check(
        kv4.shape[-1] == D_V // 2 and kv4_rope.shape[-1] == D_QK - D_V,  # PT step 2: rope rows 64 B
        f"pool must be [s_kv, {D_V // 2}] + [s_kv, {D_QK - D_V + 16}], "
        f"got {tuple(kv4.shape)} and {tuple(kv4_rope.shape)}",
    )
    _check(kv4.dim() == 2 and kv4_rope.dim() == 2, "the fp4 pool must be 2-D [s_kv, bytes]")
    _check(kv4.is_contiguous() and kv4_rope.is_contiguous(), "the fp4 pool must be contiguous")
    if indices.dim() == 3:
        _check(indices.shape[1] == 1, "only h_kv == 1 is supported")
        indices = indices.squeeze(1)
    _check(indices.dtype == torch.int32, "indices must be int32")
    s_q, h_q_real, _ = q.shape
    s_kv, topk = kv4.shape[0], indices.shape[1]
    _check(indices.shape[0] == s_q, "indices must have s_q rows")
    _check(topk % 64 == 0, f"topk must be a multiple of 64, got {topk}")
    device = q.device
    for name, t in (("q", q), ("kv4", kv4), ("kv4_rope", kv4_rope), ("indices", indices)):
        _check(t.device == device, f"{name} must be on {device}, got {t.device}")
        _check(t.stride(-1) == 1, f"{name} must be contiguous in its last dimension")

    h_q = h_q_real
    out_user = max_logits_user = lse_user = None
    if h_q % _B_H != 0:
        _check(_B_H % h_q == 0, f"h_q must be a multiple of 64 or divide 64, got {h_q}")
        q_pad = q.new_zeros(s_q, _B_H, D_QK)
        q_pad[:, :h_q] = q
        q, h_q = q_pad, _B_H
        if attn_sink is not None:
            pad = attn_sink.new_full((_B_H - h_q_real,), float("-inf"))
            attn_sink = torch.cat([attn_sink, pad])
        out_user, max_logits_user, lse_user = out, max_logits, lse
        out = max_logits = lse = None
    if attn_sink is not None:
        _check(
            attn_sink.dtype == torch.float32
            and attn_sink.numel() == h_q
            and attn_sink.is_contiguous(),
            "attn_sink must be a contiguous [h_q] float32 tensor",
        )
    if topk_length is not None:
        _check(
            topk_length.dtype == torch.int32
            and topk_length.numel() == s_q
            and topk_length.is_contiguous(),
            "topk_length must be a contiguous [s_q] int32 tensor",
        )
    if num_splits is None:
        num_splits = _choose_num_splits(s_q, h_q, topk, _num_sms(device.index))
    _check(
        num_splits in (1, 2, 4, 8) and topk % (64 * num_splits) == 0,
        f"bad num_splits {num_splits} for topk {topk}",
    )

    if out is None:
        out = torch.empty(s_q, h_q, D_V, dtype=torch.bfloat16, device=device)
    else:
        check_out_buffer(out, "out", (s_q, h_q, D_V), torch.bfloat16, device)
    if max_logits is None:
        max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        check_out_buffer(max_logits, "max_logits", (s_q, h_q), torch.float32, device)
    if lse is None:
        lse = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        check_out_buffer(lse, "lse", (s_q, h_q), torch.float32, device)
    _check(
        len({out.data_ptr(), max_logits.data_ptr(), lse.data_ptr()}) == 3,
        "out, max_logits and lse must not alias",
    )
    if num_splits > 1:
        partial_out = torch.empty(
            num_splits, s_q, h_q, D_V, dtype=torch.bfloat16, device=device
        )
        partial_max_logits = torch.empty(
            num_splits, s_q, h_q, dtype=torch.float32, device=device
        )
        partial_lse = torch.empty(
            num_splits, s_q, h_q, dtype=torch.float32, device=device
        )
    else:
        partial_out = _placeholder(device, torch.bfloat16, "partial_out")
        partial_max_logits = _placeholder(device, torch.float32, "partial_max_logits")
        partial_lse = _placeholder(device, torch.float32, "partial_lse")

    _check(kv4_sf is not None and kv4_sf.dtype == torch.uint8 and kv4_sf.numel() == kv4.shape[0], "kv4_sf must be [s_kv] uint8")
    _sparse_mla_q8kv8_sm100_op(
        q,
        kv4,
        kv4_rope,
        kv4_sf.contiguous(),
        indices,
        (
            attn_sink
            if attn_sink is not None
            else _placeholder(device, torch.float32, "attn_sink")
        ),
        (
            topk_length
            if topk_length is not None
            else _placeholder(device, torch.int32, "topk_length")
        ),
        out,
        max_logits,
        lse,
        partial_out,
        partial_max_logits,
        partial_lse,
        s_q,
        s_kv,
        h_q,
        topk,
        num_splits,
        q.stride(0),
        q.stride(1),
        indices.stride(0),
        float(bmm1_scale),
        float(bmm2_scale),
        float(pt_eL),
        int(attn_sink is not None),
        int(topk_length is not None),
    )
    if h_q != h_q_real:
        out = out[:, :h_q_real]
        max_logits = max_logits[:, :h_q_real]
        lse = lse[:, :h_q_real]
        if out_user is not None:
            out = out_user.copy_(out)
        if max_logits_user is not None:
            max_logits = max_logits_user.copy_(max_logits)
        if lse_user is not None:
            lse = lse_user.copy_(lse)
    return out, max_logits, lse


def sparse_mla_q8kv8_sm100_trtllm(
    query: torch.Tensor,
    kv4: torch.Tensor,
    kv4_rope: torch.Tensor,
    block_tables: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    seq_lens: Optional[torch.Tensor] = None,
    *,
    kv4_sf: Optional[torch.Tensor] = None,
    pt_eL: float = 0.0,
) -> torch.Tensor:
    """Same arguments and result as FlashInfer's TRT-LLM-gen sparse MLA call.

    That is ``trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=topk)``. trtllm-gen treats
    the first ``min(seq_len, topk)`` slots of a row as valid; ``seq_lens`` applies the same rule here,
    on top of masking -1 and out-of-range slots, and lets the kernel skip the blocks past it.

    Args:
        query:        [batch, q_len, h_q, 576], float8_e4m3fn
        kv4:          [s_kv, 256], uint8, packed e2m1 of the 512 NoPE dims
        kv4_rope:     [s_kv, 80], uint8, pre-scaled e4m3 RoPE + the row exponent
        block_tables: [batch, q_len, topk], int32 token slots
        bmm1_scale:   q_scale * kv_scale * sm_scale
        bmm2_scale:   kv_scale of V
        seq_lens:     optional [batch], int32; needs q_len == 1

    Returns:
        [batch, q_len, h_q, 512], bfloat16
    """
    batch, q_len, h_q, _ = query.shape
    topk_length = None
    if seq_lens is not None:
        _check(q_len == 1, f"seq_lens needs q_len == 1, got q_len={q_len}")
        _check(
            seq_lens.dtype == torch.int32 and seq_lens.numel() == batch,
            f"seq_lens must be [{batch}] int32, got {tuple(seq_lens.shape)} {seq_lens.dtype}",
        )
        topk_length = seq_lens.reshape(batch)
    out, _, _ = sparse_mla_q8kv8_sm100_fwd(
        query.view(batch * q_len, h_q, D_QK),
        kv4,
        kv4_rope,
        block_tables.view(batch * q_len, block_tables.shape[-1]),
        bmm1_scale,
        bmm2_scale,
        topk_length=topk_length,
        pt_eL=pt_eL,
        kv4_sf=kv4_sf,
    )
    return out.view(batch, q_len, h_q, D_V)



# ---------------------------------------------------------------------------------------------------------
# PT pool (patch_pt.py): per-token e4m3 scale on nope only, rope e4m3 without any scale
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@torch.no_grad()
def pack_pt_pool(kv: torch.Tensor, rt_max: float = 1.75):
    """kv [s_kv, 576] float (natural scale) -> (kv4 [s_kv, 256] u8, kv4_rope [s_kv, 80] u8, eL int).
    r_t = smallest e4m3 >= amax_t / 6 / 2^eL; eL per pool so that max r_t <= rt_max (P r_t <= 448)."""
    import math
    _check(kv.dim() == 2 and kv.shape[1] == D_QK, f"kv must be [s_kv, {D_QK}]")
    dev = kv.device
    nope = kv[:, :D_V].float()
    rope = kv[:, D_V:].float()
    amax = nope.abs().amax(dim=1)
    want = amax / 6.0
    wmax = float(want.max())
    eL = math.ceil(math.log2(wmax / rt_max)) if wmax > 0 else 0
    while True:
        w = want / 2.0 ** eL
        rt = w.to(torch.float8_e4m3fn).float()
        nxt = (rt.to(torch.float8_e4m3fn).view(torch.uint8) + 1).view(torch.float8_e4m3fn).float()
        rt = torch.where(rt < w, nxt, rt)
        if float(rt.max()) <= rt_max:
            break
        eL += 1
    rt = torch.where(amax > 0, rt, torch.full_like(rt, 2.0 ** -9))
    table = torch.tensor(_E2M1, device=dev)
    scaled = nope / (rt * 2.0 ** eL).unsqueeze(1)
    mids = (table[1:] + table[:-1]) / 2
    mag = torch.bucketize(scaled.abs().clamp(max=6.0), mids).to(torch.uint8)
    code = mag | ((scaled < 0).to(torch.uint8) << 3)
    kv4 = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    kv4_rope = torch.zeros(kv.shape[0], D_QK - D_V + 16, dtype=torch.uint8, device=dev)
    kv4_rope[:, : D_QK - D_V] = rope.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    kv4_rope[:, D_QK - D_V] = rt.to(torch.float8_e4m3fn).view(torch.uint8)
    return kv4, kv4_rope.contiguous(), eL


@torch.no_grad()
def unpack_pt_pool(kv4: torch.Tensor, kv4_rope: torch.Tensor, eL: int) -> torch.Tensor:
    """The exact values the kernel multiplies with, float32 [s_kv, 576]."""
    table = torch.tensor(_E2M1, device=kv4.device)

    def dec(c):
        m = table[(c & 7).long()]
        return torch.where((c & 8) != 0, -m, m)

    rt = kv4_rope[:, D_QK - D_V].view(torch.float8_e4m3fn).float() * 2.0 ** eL
    nope = torch.empty(kv4.shape[0], D_V, device=kv4.device)
    nope[:, 0::2] = dec(kv4 & 0xF)
    nope[:, 1::2] = dec(kv4 >> 4)
    rope = kv4_rope[:, : D_QK - D_V].view(torch.float8_e4m3fn).float()
    return torch.cat([nope * rt.unsqueeze(1), rope], dim=1)



@torch.no_grad()
def pack_pt_pool_ldg(kv: torch.Tensor, rt_max: float = 1.75):
    """PT step 2 pool: (kv4 [s_kv, 256], kv4_rope [s_kv, 64], kv4_sf [s_kv], eL) = 321 B/row."""
    kv4, kv4_rope80, eL = pack_pt_pool(kv, rt_max)
    return kv4, kv4_rope80[:, : D_QK - D_V].contiguous(), kv4_rope80[:, D_QK - D_V].contiguous(), eL
