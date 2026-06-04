"""
TLE-only Hopper FlashAttention forward kernel for varlen / paged attention.

This module exposes separate FA3 kernel families for long and short/decode
traffic.  The long path uses ``flash_varlen_fwd_v3_tle_kernel`` with FlagTree
TLE Hopper features:

* a producer async task stages Q/K/V into shared memory
* the long-sequence specialization uses two replicated MMA consumers split along M
* the short/decode specialization uses a separate non-persistent kernel
* TMA copies are used by long dense Q/K/V/O configs
* paged K/V is gathered by the producer into dense shared-memory tiles
* WGMMA QK and PV are coordinated by explicit user ``wgmma_wait`` calls

Supported subset:

* fp16/bf16 inputs/outputs
* no dropout / no returned softmax matrix
* dense varlen and paged KV
* causal, local window, ALiBi, and softcap score transforms
"""

from dataclasses import dataclass
import os

import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

try:
    import triton.experimental.tle.language as tle

    TLE_FA3_AVAILABLE = True
except Exception:
    tle = None
    TLE_FA3_AVAILABLE = False


_FA3_TLE_FAMILY_LONG = 0
_FA3_TLE_FAMILY_SHORT = 1
_FA3_TLE_FAMILY_SPLITKV = 2
_FA3_TLE_FAMILY_MIXED = 3
_FA3_TLE_FAMILY_DECODE = 4
_FA3_TLE_FAMILY_PAGED_DECODE = 5
_FA3_TLE_FAMILY_SERVE = 6
_FA3_TLE_FAMILY_PAGED_SERVE = 7
_FA3_TLE_FAMILY_AUTO = -1

_FA3_TLE_BUCKET_LONG = 0
_FA3_TLE_BUCKET_SHORT = 1
_FA3_TLE_BUCKET_SPLITKV = 2
_FA3_TLE_BUCKET_MIXED_LONG = 3
_FA3_TLE_BUCKET_MIXED_SHORT = 4
_FA3_TLE_BUCKET_DECODE = 5
_FA3_TLE_BUCKET_PAGED_DECODE = 6
_FA3_TLE_BUCKET_SERVE_SHORT = 7
_FA3_TLE_BUCKET_PAGED_SERVE_SHORT = 8

_FA3_TLE_FORCE_PATHS = {
    "auto": _FA3_TLE_FAMILY_AUTO,
    "long": _FA3_TLE_FAMILY_LONG,
    "short": _FA3_TLE_FAMILY_SHORT,
    "splitkv": _FA3_TLE_FAMILY_SPLITKV,
    "mixed": _FA3_TLE_FAMILY_MIXED,
    "decode": _FA3_TLE_FAMILY_DECODE,
    "paged_decode": _FA3_TLE_FAMILY_PAGED_DECODE,
    "serve": _FA3_TLE_FAMILY_SERVE,
    "paged_serve": _FA3_TLE_FAMILY_PAGED_SERVE,
}


def _next_power_of_2_host(value: int) -> int:
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class FA3TlePlan:
    family: str
    shape_bucket: int
    force_family_id: int
    min_q_len: int = 0
    max_q_len: int = 2**31 - 1


def fa3_tle_force_family_id() -> int:
    value = os.getenv("FLAG_GEMS_FA3_TLE_FORCE_PATH", "auto").strip().lower()
    if value not in _FA3_TLE_FORCE_PATHS:
        allowed = ", ".join(sorted(_FA3_TLE_FORCE_PATHS))
        raise RuntimeError(
            f"invalid FLAG_GEMS_FA3_TLE_FORCE_PATH={value!r}; expected one of {allowed}"
        )
    return _FA3_TLE_FORCE_PATHS[value]


def fa3_tle_select_plan(
    *,
    total_q: int,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_paged: bool,
    force_family_id: int,
) -> FA3TlePlan:
    if force_family_id == _FA3_TLE_FAMILY_LONG:
        return FA3TlePlan("long", _FA3_TLE_BUCKET_LONG, force_family_id)
    if force_family_id == _FA3_TLE_FAMILY_SHORT:
        return FA3TlePlan("short", _FA3_TLE_BUCKET_SHORT, force_family_id)
    if force_family_id == _FA3_TLE_FAMILY_SPLITKV:
        return FA3TlePlan("splitkv", _FA3_TLE_BUCKET_SPLITKV, force_family_id)
    if force_family_id == _FA3_TLE_FAMILY_MIXED:
        return FA3TlePlan(
            "paged_serve" if is_paged else "serve",
            _FA3_TLE_BUCKET_PAGED_SERVE_SHORT
            if is_paged
            else _FA3_TLE_BUCKET_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )
    if force_family_id == _FA3_TLE_FAMILY_DECODE:
        return FA3TlePlan("decode", _FA3_TLE_BUCKET_DECODE, force_family_id)
    if force_family_id == _FA3_TLE_FAMILY_PAGED_DECODE:
        return FA3TlePlan(
            "paged_decode", _FA3_TLE_BUCKET_PAGED_DECODE, force_family_id
        )
    if force_family_id == _FA3_TLE_FAMILY_SERVE:
        return FA3TlePlan(
            "serve",
            _FA3_TLE_BUCKET_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )
    if force_family_id == _FA3_TLE_FAMILY_PAGED_SERVE:
        return FA3TlePlan(
            "paged_serve",
            _FA3_TLE_BUCKET_PAGED_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )

    avg_q = total_q / max(batch_size, 1)
    if max_seqlen_q > 512 and avg_q <= 128:
        return FA3TlePlan(
            "paged_serve" if is_paged else "serve",
            _FA3_TLE_BUCKET_PAGED_SERVE_SHORT
            if is_paged
            else _FA3_TLE_BUCKET_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )
    if avg_q <= 4 and max_seqlen_q <= 64:
        if is_paged:
            return FA3TlePlan(
                "paged_decode", _FA3_TLE_BUCKET_PAGED_DECODE, force_family_id
            )
        return FA3TlePlan("decode", _FA3_TLE_BUCKET_DECODE, force_family_id)
    if avg_q <= 64 or max_seqlen_q <= 128:
        return FA3TlePlan("short", _FA3_TLE_BUCKET_SHORT, force_family_id)
    return FA3TlePlan("long", _FA3_TLE_BUCKET_LONG, force_family_id)


def fa3_tle_mixed_long_plan(force_family_id: int) -> FA3TlePlan:
    return FA3TlePlan(
        "mixed_long",
        _FA3_TLE_BUCKET_MIXED_LONG,
        force_family_id,
        min_q_len=65,
    )


def _fa3_tle_config(
    *,
    family_id,
    block_m,
    block_n,
    num_buffers_kv,
    num_mma_groups,
    num_mma_warps,
    use_tma_qo,
    use_tma_kv,
):
    return triton.Config(
        {
            "FAMILY_ID": family_id,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": num_buffers_kv,
            "NUM_MMA_WARPS": num_mma_warps,
            "NUM_MMA_GROUPS": num_mma_groups,
            "Q_STAGE_CAPACITY": _next_power_of_2_host(num_mma_groups),
            "KV_STAGE_CAPACITY": _next_power_of_2_host(num_buffers_kv),
            "USE_TMA_QO": use_tma_qo,
            "USE_TMA_KV": use_tma_kv,
        },
        num_warps=4,
    )


def _fa3_tle_configs():
    return [
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_LONG,
            block_m=128,
            block_n=128,
            num_buffers_kv=2,
            num_mma_groups=2,
            num_mma_warps=8,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
        _fa3_tle_config(
            family_id=_FA3_TLE_FAMILY_LONG,
            block_m=128,
            block_n=64,
            num_buffers_kv=1,
            num_mma_groups=2,
            num_mma_warps=8,
            use_tma_qo=True,
            use_tma_kv=True,
        ),
    ]


def _fa3_tle_config_smem_bytes(cfg, head_dim: int) -> int:
    block_k = _next_power_of_2_host(head_dim)
    block_m = cfg.kwargs["BLOCK_M"]
    block_n = cfg.kwargs["BLOCK_N"]
    num_groups = cfg.kwargs["NUM_MMA_GROUPS"]
    bm_split = block_m // num_groups
    q_stage = cfg.kwargs["Q_STAGE_CAPACITY"]
    kv_stage = cfg.kwargs["KV_STAGE_CAPACITY"]
    elems = q_stage * bm_split * block_k
    elems += 2 * kv_stage * block_n * block_k
    return elems * 2


def _prune_fa3_tle_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    shape_bucket = kwargs.get(
        "SHAPE_BUCKET", nargs.get("SHAPE_BUCKET", _FA3_TLE_BUCKET_LONG)
    )
    force_family_id = kwargs.get(
        "FORCE_FAMILY_ID", nargs.get("FORCE_FAMILY_ID", _FA3_TLE_FAMILY_AUTO)
    )

    kept = []
    for cfg in configs:
        family_id = cfg.kwargs["FAMILY_ID"]
        block_n = cfg.kwargs["BLOCK_N"]
        block_m = cfg.kwargs["BLOCK_M"]
        num_groups = cfg.kwargs["NUM_MMA_GROUPS"]

        if force_family_id in (
            _FA3_TLE_FAMILY_LONG,
            _FA3_TLE_FAMILY_SHORT,
            _FA3_TLE_FAMILY_SPLITKV,
        ) and family_id != force_family_id:
            continue
        if force_family_id == _FA3_TLE_FAMILY_MIXED:
            if shape_bucket == _FA3_TLE_BUCKET_MIXED_LONG:
                if family_id != _FA3_TLE_FAMILY_LONG:
                    continue
            elif family_id not in (
                _FA3_TLE_FAMILY_SHORT,
                _FA3_TLE_FAMILY_SPLITKV,
            ):
                continue

        if force_family_id == _FA3_TLE_FAMILY_AUTO:
            if shape_bucket in (_FA3_TLE_BUCKET_LONG, _FA3_TLE_BUCKET_MIXED_LONG):
                if family_id != _FA3_TLE_FAMILY_LONG:
                    continue
            elif shape_bucket == _FA3_TLE_BUCKET_SHORT:
                if family_id != _FA3_TLE_FAMILY_SHORT:
                    continue
            else:
                if family_id not in (
                    _FA3_TLE_FAMILY_SHORT,
                    _FA3_TLE_FAMILY_SPLITKV,
                ):
                    continue

        if head_dim > 128 and block_n > 64:
            continue
        if head_dim >= 128 and family_id != _FA3_TLE_FAMILY_LONG and block_m > 64:
            continue
        if head_dim > 128 and cfg.kwargs["NUM_BUFFERS_KV"] > 1 and block_n >= 64:
            continue
        if head_dim > 192 and block_n > 64:
            continue
        if block_m % num_groups != 0:
            continue
        if is_paged and block_n > 128:
            continue
        if _fa3_tle_config_smem_bytes(cfg, head_dim) > 220 * 1024:
            continue
        kept.append(cfg)

    if kept:
        return kept
    return [configs[0]]


def _fa3_short_configs():
    configs = []
    for block_m in (16, 32, 64):
        for block_n in (32, 64, 128):
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n},
                    num_stages=3,
                    num_warps=4,
                )
            )
    return configs


def _prune_fa3_short_configs(configs, nargs, **kwargs):
    head_dim = kwargs.get("d", nargs.get("d"))
    is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
    shape_bucket = kwargs.get(
        "SHORT_SHAPE_BUCKET", nargs.get("SHORT_SHAPE_BUCKET", _FA3_TLE_BUCKET_SHORT)
    )

    kept = []
    for cfg in configs:
        block_m = cfg.kwargs["BLOCK_M"]
        block_n = cfg.kwargs["BLOCK_N"]
        if shape_bucket == _FA3_TLE_BUCKET_DECODE:
            if block_m > 32 or block_n < 64:
                continue
        elif shape_bucket == _FA3_TLE_BUCKET_PAGED_DECODE:
            if block_m > 32 or block_n > 64:
                continue
        elif shape_bucket == _FA3_TLE_BUCKET_SERVE_SHORT:
            if block_m > 32 or block_n < 64:
                continue
        elif shape_bucket == _FA3_TLE_BUCKET_PAGED_SERVE_SHORT:
            if block_m > 32 or block_n > 64:
                continue
        elif shape_bucket == _FA3_TLE_BUCKET_SHORT:
            if block_m < 32:
                continue
        if head_dim > 128 and block_n > 64:
            continue
        if head_dim > 192 and block_n > 64:
            continue
        if is_paged and block_n > 64 and block_m > 32:
            continue
        kept.append(cfg)

    if kept:
        return kept
    return [configs[0]]


@triton.jit
def _apply_softcap_v3(S, softcap, IS_SOFTCAP: tl.constexpr):
    if IS_SOFTCAP:
        S = tl_extra_shim.tanh(S * softcap)
    return S


@triton.jit
def _apply_alibi_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    IS_CAUSAL: tl.constexpr,
    IS_ALIBI: tl.constexpr,
    alibi_slope,
):
    if IS_ALIBI:
        if IS_CAUSAL:
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias
    return S


@triton.jit
def _apply_mask_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    IS_EVEN_MN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    if IS_CAUSAL or IS_LOCAL or (not IS_EVEN_MN):
        col_lb = tl.maximum(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = tl.minimum(
            max_seqlen_k - 1,
            row_idx + max_seqlen_k - max_seqlen_q + window_size_right,
        )
        if IS_CAUSAL:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)
        if IS_LOCAL:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )
        if (not IS_LOCAL) and (not IS_CAUSAL) and (not IS_EVEN_MN):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)
    return S


@triton.jit
def _softmax_online_deferred(
    S,
    m_prev,
    l_prev,
    softmax_scale_log2e: tl.constexpr,
    IS_BORDER: tl.constexpr,
):
    m_new = tl.maximum(m_prev, tl.max(S, 1))
    if IS_BORDER:
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    else:
        m_safe = m_new

    alpha = tl.math.exp2((m_prev - m_safe) * softmax_scale_log2e)
    m_scaled = tl.where(m_new == float("-inf"), 0.0, m_safe * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - m_scaled[:, None])
    l_new = l_prev * alpha + tl.sum(P, 1)
    return alpha, P, m_new, l_new


@triton.jit
def _virtual_to_cache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    BOUNDARY_CHECK: tl.constexpr = False,
):
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if BOUNDARY_CHECK:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int32)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int32)
    return page_block_index * block_size + page_offset


def _heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@triton.jit
def _buf_phase_tle(count, num_buffers: tl.constexpr):
    buf = count % num_buffers
    phase_idx = count // num_buffers
    return buf, phase_idx


@triton.jit
def _persistent_tile_coords(tile_idx, num_pid_m, batch_size):
    m_block = tile_idx % num_pid_m
    hb = tile_idx // num_pid_m
    bid = hb % batch_size
    hid = hb // batch_size
    return m_block, bid, hid


@triton.jit
def _fence_async_shared_cta():
    tl.inline_asm_elementwise(
        "mov.u32 $0, 0x0; fence.proxy.async.shared::cta;",
        constraints="=r",
        args=(),
        dtype=(tl.int32,),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _copy_paged_kv_tile_to_smem(
    src_base,
    row_stride,
    page_table_ptr_b,
    smem_tile,
    n_offset,
    k_len,
    d: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    for row_base in tl.static_range(0, BLOCK_N, 32):
        rows = row_base + tl.arange(0, 32)
        logical_idx = n_offset + rows
        row_valid = logical_idx < k_len
        if block_size % 32 == 0:
            first_idx = n_offset + row_base
            page_idx = first_idx // block_size
            page_block = tl.load(
                page_table_ptr_b + page_idx,
                mask=first_idx < k_len,
                other=0,
            ).to(tl.int32)
            cache_idx = page_block * block_size + (logical_idx % block_size)
        else:
            cache_idx = _virtual_to_cache(
                logical_idx,
                k_len,
                page_table_ptr_b,
                block_size,
                BOUNDARY_CHECK=True,
            )

        for col_base in tl.static_range(0, HEAD_DIM_PADDED, 32):
            cols = col_base + tl.arange(0, 32)
            src_ptrs = src_base + cache_idx[:, None] * row_stride + cols[None, :]
            load_mask = row_valid[:, None] & (cols[None, :] < d)
            # Keep paged gather loads on Triton's default cache policy. On this
            # TLE producer path, evict_last can lower to an illegal instruction.
            vals = tl.load(
                src_ptrs,
                mask=load_mask,
                other=0.0,
            )
            smem_rows = tl.broadcast_to(rows[:, None], (32, 32))
            smem_cols = tl.broadcast_to(cols[None, :], (32, 32))
            smem_ptrs = tle.gpu.local_ptr(smem_tile, (smem_rows, smem_cols))
            tl.store(smem_ptrs, vals)


@triton.jit
def _copy_dense_tile_to_smem(
    src_base,
    row_stride,
    smem_tile,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    for row_base in tl.static_range(0, BLOCK_ROWS, 32):
        rows = row_base + tl.arange(0, 32)
        logical_rows = row_offset + rows
        row_valid = logical_rows < row_count
        for col_base in tl.static_range(0, HEAD_DIM_PADDED, 32):
            cols = col_base + tl.arange(0, 32)
            vals = tl.load(
                src_base + logical_rows[:, None] * row_stride + cols[None, :],
                mask=row_valid[:, None] & (cols[None, :] < d),
                other=0.0,
            )
            smem_rows = tl.broadcast_to(rows[:, None], (32, 32))
            smem_cols = tl.broadcast_to(cols[None, :], (32, 32))
            smem_ptrs = tle.gpu.local_ptr(smem_tile, (smem_rows, smem_cols))
            tl.store(smem_ptrs, vals)


@triton.jit
def _store_dense_tile_from_regs(
    dst_base,
    row_stride,
    vals,
    row_offset,
    row_count,
    d: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
):
    rows = row_offset + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HEAD_DIM_PADDED)
    ptrs = dst_base + rows[:, None] * row_stride + cols[None, :]
    mask = (rows[:, None] < row_count) & (cols[None, :] < d)
    tl.store(ptrs, vals, mask=mask)


@libentry()
@triton.autotune(
    configs=_fa3_tle_configs(),
    prune_configs_by={"early_config_prune": _prune_fa3_tle_configs},
    key=[
        "d",
        "is_paged",
        "is_causal",
        "is_local",
        "is_alibi",
        "SHAPE_BUCKET",
        "FORCE_FAMILY_ID",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_K": _heur_block_k,
    }
)
@triton.jit(
    do_not_specialize=[
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "total_q",
    ]
)
def flash_varlen_fwd_v3_tle_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    Q_STAGE_CAPACITY: tl.constexpr,
    KV_STAGE_CAPACITY: tl.constexpr,
    USE_TMA_QO: tl.constexpr,
    USE_TMA_KV: tl.constexpr,
    FAMILY_ID: tl.constexpr,
    SHAPE_BUCKET: tl.constexpr,
    FORCE_FAMILY_ID: tl.constexpr,
    MIN_Q_LEN_TO_PROCESS: tl.constexpr,
    MAX_Q_LEN_TO_PROCESS: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K
    INPUT_DTYPE: tl.constexpr = q_ptr.dtype.element_ty
    THREADS_IN_MMA_GROUPS: tl.constexpr = NUM_MMA_WARPS * 32

    q_smem = tle.gpu.alloc(
        [Q_STAGE_CAPACITY, BM_SPLIT, HEAD_DIM_PADDED],
        dtype=INPUT_DTYPE,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_smem = tle.gpu.alloc(
        [KV_STAGE_CAPACITY, BLOCK_N, HEAD_DIM_PADDED],
        dtype=INPUT_DTYPE,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [KV_STAGE_CAPACITY, BLOCK_N, HEAD_DIM_PADDED],
        dtype=INPUT_DTYPE,
        layout=None,
        scope=tle.gpu.smem,
    )

    q_empties = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
        init=tle.gpu.READY,
    )
    q_fulls = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BM_SPLIT * HEAD_DIM_PADDED * 2,
    )
    q_fulls_manual = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
    )
    k_empties = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=NUM_MMA_GROUPS,
        init=tle.gpu.READY,
    )
    k_fulls = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BLOCK_N * HEAD_DIM_PADDED * 2,
    )
    k_fulls_manual = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=1,
    )
    v_empties = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=NUM_MMA_GROUPS,
        init=tle.gpu.READY,
    )
    v_fulls = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BLOCK_N * HEAD_DIM_PADDED * 2,
    )
    v_fulls_manual = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=1,
    )

    pingpong = tle.gpu.alloc_barriers(
        num_barriers=2,
        arrive_count=THREADS_IN_MMA_GROUPS,
    )
    ping_to_c0 = pingpong[0]
    ping_to_c1 = pingpong[1]

    with tle.gpu.async_tasks():
        with tle.gpu.async_task("producer"):
            prog_id = tl.program_id(0)
            num_progs = tl.num_programs(0)
            num_pid_m = tl.cdiv(seqlen_q, BLOCK_M)
            total_tiles = num_pid_m * b * h

            tile_idx = prog_id
            tile_count = 0
            accum_cnt_kv = 0
            while tile_idx < total_tiles:
                m_block, bid, hid = _persistent_tile_coords(tile_idx, num_pid_m, b)

                if is_cu_seqlens_q:
                    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
                    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
                    q_len = q_eos - q_bos
                    q_offset = q_bos * q_row_stride
                else:
                    q_len = seqlen_q
                    q_offset = bid * q_batch_stride

                if is_cu_seqlens_k:
                    k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
                    k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
                    k_len_cache = k_eos - k_bos
                else:
                    k_len_cache = seqlen_k
                    k_bos = 0

                if is_seqused_k:
                    k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
                else:
                    k_len = k_len_cache

                process_q_tile = (q_len >= MIN_Q_LEN_TO_PROCESS) & (
                    q_len <= MAX_Q_LEN_TO_PROCESS
                )
                valid_q_tile = (m_block * BLOCK_M < q_len) & process_q_tile
                if valid_q_tile:
                    if is_local:
                        n_block_min = tl.maximum(
                            0,
                            (
                                m_block * BLOCK_M
                                + k_len
                                - q_len
                                - window_size_left
                            )
                            // BLOCK_N,
                        )
                    else:
                        n_block_min = 0

                    n_block_max = tl.cdiv(k_len, BLOCK_N)
                    if is_causal or is_local:
                        n_block_max = tl.minimum(
                            n_block_max,
                            tl.cdiv(
                                (m_block + 1) * BLOCK_M
                                + k_len
                                - q_len
                                + window_size_right,
                                BLOCK_N,
                            ),
                        )

                    if n_block_min < n_block_max:
                        kv_head = hid // h_hk_ratio
                        q_base = q_ptr + q_offset + hid * q_head_stride
                        if is_paged:
                            page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
                            k_base = k_ptr + kv_head * k_head_stride
                            v_base = v_ptr + kv_head * v_head_stride
                        else:
                            k_base = (
                                k_ptr
                                + k_bos * k_row_stride
                                + kv_head * k_head_stride
                            )
                            v_base = (
                                v_ptr
                                + k_bos * v_row_stride
                                + kv_head * v_head_stride
                            )

                        if USE_TMA_QO:
                            q_desc = tl.make_tensor_descriptor(
                                base=q_base,
                                shape=[q_len, d],
                                strides=[q_row_stride, 1],
                                block_shape=[BM_SPLIT, HEAD_DIM_PADDED],
                            )
                        if (not is_paged) and USE_TMA_KV:
                            k_desc = tl.make_tensor_descriptor(
                                base=k_base,
                                shape=[k_len_cache, d],
                                strides=[k_row_stride, 1],
                                block_shape=[BLOCK_N, HEAD_DIM_PADDED],
                            )
                            v_desc = tl.make_tensor_descriptor(
                                base=v_base,
                                shape=[k_len_cache, d],
                                strides=[v_row_stride, 1],
                                block_shape=[BLOCK_N, HEAD_DIM_PADDED],
                            )

                        q_buf, q_phase_idx = _buf_phase_tle(
                            tile_count, NUM_BUFFERS_Q
                        )
                        q0_idx = q_buf
                        q1_idx = q_buf + NUM_BUFFERS_Q

                        tle.gpu.barrier_wait(q_empties[q0_idx], phaseIdx=q_phase_idx)
                        if USE_TMA_QO:
                            tle.gpu.copy(
                                q_desc,
                                q_smem.slot(q0_idx),
                                [BM_SPLIT, HEAD_DIM_PADDED],
                                [m_block * BLOCK_M, 0],
                                barrier=q_fulls[q0_idx],
                            )
                        else:
                            _copy_dense_tile_to_smem(
                                q_base,
                                q_row_stride,
                                q_smem.slot(q0_idx),
                                m_block * BLOCK_M,
                                q_len,
                                d,
                                BM_SPLIT,
                                HEAD_DIM_PADDED,
                            )
                            _fence_async_shared_cta()
                            tle.gpu.barrier_arrive(
                                q_fulls_manual[q0_idx], phaseIdx=q_phase_idx
                            )

                        kv_buf, kv_phase_idx = _buf_phase_tle(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        kv_offset = n_block_min * BLOCK_N
                        tle.gpu.barrier_wait(k_empties[kv_buf], phaseIdx=kv_phase_idx)
                        if is_paged:
                            _copy_paged_kv_tile_to_smem(
                                k_base,
                                k_row_stride,
                                page_table_ptr_b,
                                k_smem.slot(kv_buf),
                                kv_offset,
                                k_len,
                                d,
                                block_size,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                            _fence_async_shared_cta()
                            tle.gpu.barrier_arrive(
                                k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                            )
                        elif USE_TMA_KV:
                            tle.gpu.copy(
                                k_desc,
                                k_smem.slot(kv_buf),
                                [BLOCK_N, HEAD_DIM_PADDED],
                                [kv_offset, 0],
                                barrier=k_fulls[kv_buf],
                            )
                        else:
                            _copy_dense_tile_to_smem(
                                k_base,
                                k_row_stride,
                                k_smem.slot(kv_buf),
                                kv_offset,
                                k_len,
                                d,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                            _fence_async_shared_cta()
                            tle.gpu.barrier_arrive(
                                k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                            )

                        if NUM_MMA_GROUPS == 2:
                            tle.gpu.barrier_wait(
                                q_empties[q1_idx], phaseIdx=q_phase_idx
                            )
                            if USE_TMA_QO:
                                tle.gpu.copy(
                                    q_desc,
                                    q_smem.slot(q1_idx),
                                    [BM_SPLIT, HEAD_DIM_PADDED],
                                    [m_block * BLOCK_M + BM_SPLIT, 0],
                                    barrier=q_fulls[q1_idx],
                                )
                            else:
                                _copy_dense_tile_to_smem(
                                    q_base,
                                    q_row_stride,
                                    q_smem.slot(q1_idx),
                                    m_block * BLOCK_M + BM_SPLIT,
                                    q_len,
                                    d,
                                    BM_SPLIT,
                                    HEAD_DIM_PADDED,
                                )
                                _fence_async_shared_cta()
                                tle.gpu.barrier_arrive(
                                    q_fulls_manual[q1_idx], phaseIdx=q_phase_idx
                                )

                        tle.gpu.barrier_wait(v_empties[kv_buf], phaseIdx=kv_phase_idx)
                        if is_paged:
                            _copy_paged_kv_tile_to_smem(
                                v_base,
                                v_row_stride,
                                page_table_ptr_b,
                                v_smem.slot(kv_buf),
                                kv_offset,
                                k_len,
                                d,
                                block_size,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                            _fence_async_shared_cta()
                            tle.gpu.barrier_arrive(
                                v_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                            )
                        elif USE_TMA_KV:
                            tle.gpu.copy(
                                v_desc,
                                v_smem.slot(kv_buf),
                                [BLOCK_N, HEAD_DIM_PADDED],
                                [kv_offset, 0],
                                barrier=v_fulls[kv_buf],
                            )
                        else:
                            _copy_dense_tile_to_smem(
                                v_base,
                                v_row_stride,
                                v_smem.slot(kv_buf),
                                kv_offset,
                                k_len,
                                d,
                                BLOCK_N,
                                HEAD_DIM_PADDED,
                            )
                            _fence_async_shared_cta()
                            tle.gpu.barrier_arrive(
                                v_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                            )
                        accum_cnt_kv += 1

                        n_block = n_block_min + 1
                        while n_block < n_block_max:
                            kv_buf, kv_phase_idx = _buf_phase_tle(
                                accum_cnt_kv, NUM_BUFFERS_KV
                            )
                            kv_offset = n_block * BLOCK_N

                            tle.gpu.barrier_wait(
                                k_empties[kv_buf], phaseIdx=kv_phase_idx
                            )
                            if is_paged:
                                _copy_paged_kv_tile_to_smem(
                                    k_base,
                                    k_row_stride,
                                    page_table_ptr_b,
                                    k_smem.slot(kv_buf),
                                    kv_offset,
                                    k_len,
                                    d,
                                    block_size,
                                    BLOCK_N,
                                    HEAD_DIM_PADDED,
                                )
                                _fence_async_shared_cta()
                                tle.gpu.barrier_arrive(
                                    k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                                )
                            elif USE_TMA_KV:
                                tle.gpu.copy(
                                    k_desc,
                                    k_smem.slot(kv_buf),
                                    [BLOCK_N, HEAD_DIM_PADDED],
                                    [kv_offset, 0],
                                    barrier=k_fulls[kv_buf],
                                )
                            else:
                                _copy_dense_tile_to_smem(
                                    k_base,
                                    k_row_stride,
                                    k_smem.slot(kv_buf),
                                    kv_offset,
                                    k_len,
                                    d,
                                    BLOCK_N,
                                    HEAD_DIM_PADDED,
                                )
                                _fence_async_shared_cta()
                                tle.gpu.barrier_arrive(
                                    k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                                )

                            tle.gpu.barrier_wait(
                                v_empties[kv_buf], phaseIdx=kv_phase_idx
                            )
                            if is_paged:
                                _copy_paged_kv_tile_to_smem(
                                    v_base,
                                    v_row_stride,
                                    page_table_ptr_b,
                                    v_smem.slot(kv_buf),
                                    kv_offset,
                                    k_len,
                                    d,
                                    block_size,
                                    BLOCK_N,
                                    HEAD_DIM_PADDED,
                                )
                                _fence_async_shared_cta()
                                tle.gpu.barrier_arrive(
                                    v_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                                )
                            elif USE_TMA_KV:
                                tle.gpu.copy(
                                    v_desc,
                                    v_smem.slot(kv_buf),
                                    [BLOCK_N, HEAD_DIM_PADDED],
                                    [kv_offset, 0],
                                    barrier=v_fulls[kv_buf],
                                )
                            else:
                                _copy_dense_tile_to_smem(
                                    v_base,
                                    v_row_stride,
                                    v_smem.slot(kv_buf),
                                    kv_offset,
                                    k_len,
                                    d,
                                    BLOCK_N,
                                    HEAD_DIM_PADDED,
                                )
                                _fence_async_shared_cta()
                                tle.gpu.barrier_arrive(
                                    v_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                                )
                            accum_cnt_kv += 1
                            n_block += 1

                        tile_count += 1

                tile_idx += num_progs

        with tle.gpu.async_task(
            num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS,
            registers=232,
            replicate=NUM_MMA_GROUPS,
            name="mma",
        ):
            cid: tl.constexpr = tle.gpu.async_task_replica_id()
            prog_id = tl.program_id(0)
            num_progs = tl.num_programs(0)
            num_pid_m = tl.cdiv(seqlen_q, BLOCK_M)
            total_tiles = num_pid_m * b * h

            if NUM_MMA_GROUPS == 2 and cid == 1:
                tle.gpu.barrier_arrive(ping_to_c0)

            tile_idx = prog_id
            tile_count = 0
            accum_cnt_kv = 0
            while tile_idx < total_tiles:
                m_block, bid, hid = _persistent_tile_coords(tile_idx, num_pid_m, b)

                if is_cu_seqlens_q:
                    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
                    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
                    q_len = q_eos - q_bos
                    o_offset = q_bos * o_row_stride
                    lse_offset = q_bos
                else:
                    q_len = seqlen_q
                    o_offset = bid * o_batch_stride
                    lse_offset = bid * seqlen_q

                if is_cu_seqlens_k:
                    k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
                    k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
                    k_len_cache = k_eos - k_bos
                else:
                    k_len_cache = seqlen_k

                if is_seqused_k:
                    k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
                else:
                    k_len = k_len_cache

                if is_alibi:
                    alibi_slope = tl.load(
                        alibi_slopes_ptr + bid * alibi_slopes_batch_stride + hid
                    )
                    alibi_slope = alibi_slope / scale_softmax
                else:
                    alibi_slope = 0.0

                process_q_tile = (q_len >= MIN_Q_LEN_TO_PROCESS) & (
                    q_len <= MAX_Q_LEN_TO_PROCESS
                )
                valid_q_tile = (m_block * BLOCK_M < q_len) & process_q_tile
                if valid_q_tile:
                    if is_local:
                        n_block_min = tl.maximum(
                            0,
                            (
                                m_block * BLOCK_M
                                + k_len
                                - q_len
                                - window_size_left
                            )
                            // BLOCK_N,
                        )
                    else:
                        n_block_min = 0

                    n_block_max = tl.cdiv(k_len, BLOCK_N)
                    if is_causal or is_local:
                        n_block_max = tl.minimum(
                            n_block_max,
                            tl.cdiv(
                                (m_block + 1) * BLOCK_M
                                + k_len
                                - q_len
                                + window_size_right,
                                BLOCK_N,
                            ),
                        )

                    row_idx_q = (
                        m_block * BLOCK_M
                        + cid * BM_SPLIT
                        + tl.arange(0, BM_SPLIT)
                    )
                    o_base = o_ptr + o_offset + hid * o_head_stride
                    if USE_TMA_QO:
                        o_desc = tl.make_tensor_descriptor(
                            base=o_base,
                            shape=[q_len, d],
                            strides=[o_row_stride, 1],
                            block_shape=[BM_SPLIT, HEAD_DIM_PADDED],
                        )

                    if n_block_min < n_block_max:
                        rowmax = tl.full(
                            [BM_SPLIT], float("-inf"), dtype=tl.float32
                        )
                        rowsum = tl.zeros([BM_SPLIT], dtype=tl.float32)
                        acc = tl.zeros(
                            [BM_SPLIT, HEAD_DIM_PADDED], dtype=tl.float32
                        )

                        q_buf, q_phase_idx = _buf_phase_tle(
                            tile_count, NUM_BUFFERS_Q
                        )
                        q_idx = q_buf + cid * NUM_BUFFERS_Q
                        if USE_TMA_QO:
                            tle.gpu.barrier_wait(
                                q_fulls[q_idx], phaseIdx=q_phase_idx
                            )
                        else:
                            tle.gpu.barrier_wait(
                                q_fulls_manual[q_idx], phaseIdx=q_phase_idx
                            )

                        kv_buf, kv_phase_idx = _buf_phase_tle(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        if is_paged or (not USE_TMA_KV):
                            tle.gpu.barrier_wait(
                                k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                            )
                        else:
                            tle.gpu.barrier_wait(
                                k_fulls[kv_buf], phaseIdx=kv_phase_idx
                            )

                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_wait(ping_to_c0)
                            else:
                                tle.gpu.barrier_wait(ping_to_c1)
                        qk = tle.gpu.wgmma(
                            q_smem.slot(q_idx),
                            k_smem.slot(kv_buf),
                            out_dtype=tl.float32,
                            trans_b=True,
                        )
                        if NUM_MMA_GROUPS == 2:
                            if cid == 0:
                                tle.gpu.barrier_arrive(ping_to_c1)
                            else:
                                tle.gpu.barrier_arrive(ping_to_c0)

                        qk = tle.gpu.wgmma_wait(0, qk)
                        tle.gpu.barrier_arrive(
                            k_empties[kv_buf], phaseIdx=kv_phase_idx
                        )

                        n_block = n_block_min
                        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                        qk = _apply_softcap_v3(qk, softcap, is_softcap)
                        qk = _apply_alibi_v3(
                            qk,
                            col_idx,
                            row_idx_q,
                            q_len,
                            k_len,
                            IS_CAUSAL=is_causal,
                            IS_ALIBI=is_alibi,
                            alibi_slope=alibi_slope,
                        )
                        qk = _apply_mask_v3(
                            qk,
                            col_idx,
                            row_idx_q,
                            q_len,
                            k_len,
                            window_size_left,
                            window_size_right,
                            IS_EVEN_MN=False,
                            IS_CAUSAL=is_causal,
                            IS_LOCAL=is_local,
                        )
                        alpha, p, rowmax, rowsum = _softmax_online_deferred(
                            qk,
                            rowmax,
                            rowsum,
                            softmax_scale_log2e=scale_softmax_log2,
                            IS_BORDER=True,
                        )
                        accum_cnt_kv += 1
                        n_block += 1

                        while n_block < n_block_max:
                            kv_buf, kv_phase_idx = _buf_phase_tle(
                                accum_cnt_kv, NUM_BUFFERS_KV
                            )
                            if is_paged or (not USE_TMA_KV):
                                tle.gpu.barrier_wait(
                                    k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                                )
                            else:
                                tle.gpu.barrier_wait(
                                    k_fulls[kv_buf], phaseIdx=kv_phase_idx
                                )

                            if NUM_MMA_GROUPS == 2:
                                if cid == 0:
                                    tle.gpu.barrier_wait(ping_to_c0)
                                else:
                                    tle.gpu.barrier_wait(ping_to_c1)
                            qk = tle.gpu.wgmma(
                                q_smem.slot(q_idx),
                                k_smem.slot(kv_buf),
                                out_dtype=tl.float32,
                                trans_b=True,
                            )
                            if NUM_MMA_GROUPS == 2:
                                if cid == 0:
                                    tle.gpu.barrier_arrive(ping_to_c1)
                                else:
                                    tle.gpu.barrier_arrive(ping_to_c0)

                            v_buf, v_phase_idx = _buf_phase_tle(
                                accum_cnt_kv - 1, NUM_BUFFERS_KV
                            )
                            if is_paged or (not USE_TMA_KV):
                                tle.gpu.barrier_wait(
                                    v_fulls_manual[v_buf], phaseIdx=v_phase_idx
                                )
                            else:
                                tle.gpu.barrier_wait(
                                    v_fulls[v_buf], phaseIdx=v_phase_idx
                                )
                            acc = tle.gpu.wgmma(
                                p.to(INPUT_DTYPE),
                                v_smem.slot(v_buf),
                                acc,
                            )

                            qk = tle.gpu.wgmma_wait(1, qk)
                            tle.gpu.barrier_arrive(
                                k_empties[kv_buf], phaseIdx=kv_phase_idx
                            )

                            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                            qk = _apply_softcap_v3(qk, softcap, is_softcap)
                            qk = _apply_alibi_v3(
                                qk,
                                col_idx,
                                row_idx_q,
                                q_len,
                                k_len,
                                IS_CAUSAL=is_causal,
                                IS_ALIBI=is_alibi,
                                alibi_slope=alibi_slope,
                            )
                            qk = _apply_mask_v3(
                                qk,
                                col_idx,
                                row_idx_q,
                                q_len,
                                k_len,
                                window_size_left,
                                window_size_right,
                                IS_EVEN_MN=False,
                                IS_CAUSAL=is_causal,
                                IS_LOCAL=is_local,
                            )
                            alpha, p, rowmax, rowsum = _softmax_online_deferred(
                                qk,
                                rowmax,
                                rowsum,
                                softmax_scale_log2e=scale_softmax_log2,
                                IS_BORDER=True,
                            )

                            acc = tle.gpu.wgmma_wait(0, acc)
                            tle.gpu.barrier_arrive(
                                v_empties[v_buf], phaseIdx=v_phase_idx
                            )
                            acc = acc * alpha[:, None]

                            accum_cnt_kv += 1
                            n_block += 1

                        v_buf, v_phase_idx = _buf_phase_tle(
                            accum_cnt_kv - 1, NUM_BUFFERS_KV
                        )
                        if is_paged or (not USE_TMA_KV):
                            tle.gpu.barrier_wait(
                                v_fulls_manual[v_buf], phaseIdx=v_phase_idx
                            )
                        else:
                            tle.gpu.barrier_wait(
                                v_fulls[v_buf], phaseIdx=v_phase_idx
                            )
                        acc = tle.gpu.wgmma(p.to(INPUT_DTYPE), v_smem.slot(v_buf), acc)

                        acc = tle.gpu.wgmma_wait(1, acc)
                        tle.gpu.barrier_arrive(q_empties[q_idx], phaseIdx=q_phase_idx)

                        acc = tle.gpu.wgmma_wait(0, acc)
                        tle.gpu.barrier_arrive(
                            v_empties[v_buf], phaseIdx=v_phase_idx
                        )

                        invalid = (rowsum == 0) | (rowsum != rowsum)
                        inv_sum = tl.where(invalid, 1.0, 1.0 / rowsum)
                        acc = acc * inv_sum[:, None]
                        lse = tl.where(
                            invalid,
                            float("inf"),
                            rowmax * scale_softmax + tl.log(rowsum),
                        )
                        tile_count += 1
                    else:
                        acc = tl.zeros(
                            [BM_SPLIT, HEAD_DIM_PADDED], dtype=tl.float32
                        )
                        lse = tl.full([BM_SPLIT], float("inf"), dtype=tl.float32)

                    if USE_TMA_QO:
                        o_desc.store(
                            [m_block * BLOCK_M + cid * BM_SPLIT, 0],
                            acc.to(o_ptr.dtype.element_ty),
                        )
                    else:
                        _store_dense_tile_from_regs(
                            o_base,
                            o_row_stride,
                            acc.to(o_ptr.dtype.element_ty),
                            m_block * BLOCK_M + cid * BM_SPLIT,
                            q_len,
                            d,
                            BM_SPLIT,
                            HEAD_DIM_PADDED,
                        )
                    lse_ptr = softmax_lse_ptr + hid * total_q + lse_offset + row_idx_q
                    tl.store(lse_ptr, lse, mask=row_idx_q < q_len)

                tile_idx += num_progs

@libentry()
@triton.autotune(
    configs=_fa3_short_configs(),
    prune_configs_by={"early_config_prune": _prune_fa3_short_configs},
    key=[
        "d",
        "is_paged",
        "is_causal",
        "is_local",
        "is_alibi",
        "SHORT_SHAPE_BUCKET",
        "MIN_Q_LEN_TO_PROCESS",
        "MAX_Q_LEN_TO_PROCESS",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_K": _heur_block_k,
    }
)
@triton.jit(
    do_not_specialize=[
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "total_q",
    ]
)
def flash_varlen_fwd_v3_tle_short_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHORT_SHAPE_BUCKET: tl.constexpr,
    MIN_Q_LEN_TO_PROCESS: tl.constexpr,
    MAX_Q_LEN_TO_PROCESS: tl.constexpr,
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
    else:
        k_len_cache = seqlen_k
        k_bos = 0

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    process_q = (q_len >= MIN_Q_LEN_TO_PROCESS) & (q_len <= MAX_Q_LEN_TO_PROCESS)
    process_q = process_q & (m_block * BLOCK_M < q_len)
    if process_q:
        if is_local:
            n_block_min = tl.maximum(
                0,
                (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N,
            )
        else:
            n_block_min = 0

        n_block_max = tl.cdiv(k_len, BLOCK_N)
        if is_causal or is_local:
            n_block_max = tl.minimum(
                n_block_max,
                tl.cdiv(
                    (m_block + 1) * BLOCK_M
                    + k_len
                    - q_len
                    + window_size_right,
                    BLOCK_N,
                ),
            )

        if (not is_causal) and (not is_local):
            n_masking_steps = 1
        else:
            n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1
        n_masking_steps = tl.maximum(
            0, tl.minimum(n_block_max - n_block_min, n_masking_steps)
        )

        if is_alibi:
            alibi_slope = tl.load(
                alibi_slopes_ptr + bid * alibi_slopes_batch_stride + hid
            )
            alibi_slope = alibi_slope / scale_softmax
        else:
            alibi_slope = 0.0

        q_base = q_ptr + q_offset + hid * q_head_stride
        q_desc = tl.make_tensor_descriptor(
            base=q_base,
            shape=[q_len, d],
            strides=[q_row_stride, 1],
            block_shape=[BLOCK_M, HEAD_DIM_PADDED],
        )
        o_base = o_ptr + o_offset + hid * o_head_stride
        o_desc = tl.make_tensor_descriptor(
            base=o_base,
            shape=[q_len, d],
            strides=[o_row_stride, 1],
            block_shape=[BLOCK_M, HEAD_DIM_PADDED],
        )

        kv_head = hid // h_hk_ratio
        if is_paged:
            page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
            k_base = k_ptr + kv_head * k_head_stride
            v_base = v_ptr + kv_head * v_head_stride
        else:
            k_base = k_ptr + k_bos * k_row_stride + kv_head * k_head_stride
            v_base = v_ptr + k_bos * v_row_stride + kv_head * v_head_stride
            k_desc = tl.make_tensor_descriptor(
                base=k_base,
                shape=[k_len_cache, d],
                strides=[k_row_stride, 1],
                block_shape=[BLOCK_N, HEAD_DIM_PADDED],
            )
            v_desc = tl.make_tensor_descriptor(
                base=v_base,
                shape=[k_len_cache, d],
                strides=[v_row_stride, 1],
                block_shape=[BLOCK_N, HEAD_DIM_PADDED],
            )

        q_tile = q_desc.load([m_block * BLOCK_M, 0])
        row_idx_q = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        rowmax = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        rowsum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

        n_block_start_mask = n_block_max - 1
        for step in tl.range(0, n_masking_steps):
            n_block = n_block_start_mask - step
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

            if is_paged:
                cache_idx = _virtual_to_cache(
                    col_idx,
                    k_len,
                    page_table_ptr_b,
                    block_size,
                    BOUNDARY_CHECK=True,
                )
                d_idx = tl.arange(0, HEAD_DIM_PADDED)
                d_mask = d_idx < d
                kv_mask = col_idx < k_len
                bK = tl.load(
                    k_base + cache_idx[None, :] * k_row_stride + d_idx[:, None],
                    mask=d_mask[:, None] & kv_mask[None, :],
                    other=0.0,
                )
                bV = tl.load(
                    v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
            else:
                bK = tl.trans(k_desc.load([n_block * BLOCK_N, 0]))
                bV = v_desc.load([n_block * BLOCK_N, 0])

            S = tl.dot(q_tile, bK, out_dtype=tl.float32)
            S = _apply_softcap_v3(S, softcap, is_softcap)
            S = _apply_alibi_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                IS_CAUSAL=is_causal,
                IS_ALIBI=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = _apply_mask_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                IS_EVEN_MN=False,
                IS_CAUSAL=is_causal,
                IS_LOCAL=is_local,
            )
            alpha, P, rowmax, rowsum = _softmax_online_deferred(
                S,
                rowmax,
                rowsum,
                softmax_scale_log2e=scale_softmax_log2,
                IS_BORDER=True,
            )
            acc = acc * alpha[:, None]
            acc = tl.dot(P.to(v_ptr.dtype.element_ty), bV, acc, out_dtype=tl.float32)

        n_dense_end = n_block_max - n_masking_steps
        for n_block in tl.range(
            n_dense_end - 1,
            n_block_min - 1,
            step=-1,
            num_stages=3,
        ):
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            if is_paged:
                cache_idx = _virtual_to_cache(
                    col_idx,
                    k_len,
                    page_table_ptr_b,
                    block_size,
                    BOUNDARY_CHECK=True,
                )
                d_idx = tl.arange(0, HEAD_DIM_PADDED)
                d_mask = d_idx < d
                kv_mask = col_idx < k_len
                bK = tl.load(
                    k_base + cache_idx[None, :] * k_row_stride + d_idx[:, None],
                    mask=d_mask[:, None] & kv_mask[None, :],
                    other=0.0,
                )
                bV = tl.load(
                    v_base + cache_idx[:, None] * v_row_stride + d_idx[None, :],
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
            else:
                bK = tl.trans(k_desc.load([n_block * BLOCK_N, 0]))
                bV = v_desc.load([n_block * BLOCK_N, 0])

            S = tl.dot(q_tile, bK, out_dtype=tl.float32)
            S = _apply_softcap_v3(S, softcap, is_softcap)
            S = _apply_alibi_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                IS_CAUSAL=is_causal,
                IS_ALIBI=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = _apply_mask_v3(
                S,
                col_idx,
                row_idx_q,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                IS_EVEN_MN=True,
                IS_CAUSAL=False,
                IS_LOCAL=is_local,
            )
            alpha, P, rowmax, rowsum = _softmax_online_deferred(
                S,
                rowmax,
                rowsum,
                softmax_scale_log2e=scale_softmax_log2,
                IS_BORDER=is_local,
            )
            acc = acc * alpha[:, None]
            acc = tl.dot(P.to(v_ptr.dtype.element_ty), bV, acc, out_dtype=tl.float32)

        invalid = (rowsum == 0) | (rowsum != rowsum)
        inv_sum = tl.where(invalid, 1.0, 1.0 / rowsum)
        acc = acc * inv_sum[:, None]
        lse = tl.where(
            invalid,
            float("inf"),
            rowmax * scale_softmax + tl.log(rowsum),
        )
        o_desc.store([m_block * BLOCK_M, 0], acc.to(o_ptr.dtype.element_ty))
        lse_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        lse_ptr = softmax_lse_ptr + hid * total_q + lse_offset + lse_row
        tl.store(lse_ptr, lse, mask=lse_row < q_len)


flash_varlen_fwd_v3_tle_splitkv_kernel = flash_varlen_fwd_v3_tle_short_kernel
