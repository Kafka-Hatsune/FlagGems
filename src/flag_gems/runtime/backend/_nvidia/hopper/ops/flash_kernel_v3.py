"""
TLE-only Hopper FlashAttention forward kernel for varlen / paged attention.

This module intentionally exposes a single FA3 kernel:
``flash_varlen_fwd_v3_tle_kernel``.  It uses FlagTree TLE Hopper features:

* a producer async task stages Q/K/V into shared memory
* two replicated MMA consumer tasks split BLOCK_M along M
* TMA copies are used for dense Q/K/V and O
* paged K/V is gathered by the producer into dense shared-memory tiles
* WGMMA QK and PV are coordinated by explicit user ``wgmma_wait`` calls

Supported subset:

* fp16/bf16 inputs/outputs
* no dropout / no returned softmax matrix
* dense varlen and paged KV
* causal, local window, ALiBi, and softcap score transforms
"""

import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

try:
    import triton.experimental.tle.language as tle

    TLE_FA3_AVAILABLE = True
except Exception:
    tle = None
    TLE_FA3_AVAILABLE = False


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


@libentry()
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

                valid_q_tile = m_block * BLOCK_M < q_len
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

                        q_desc = tl.make_tensor_descriptor(
                            base=q_base,
                            shape=[q_len, d],
                            strides=[q_row_stride, 1],
                            block_shape=[BM_SPLIT, HEAD_DIM_PADDED],
                        )
                        if not is_paged:
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
                        tle.gpu.copy(
                            q_desc,
                            q_smem.slot(q0_idx),
                            [BM_SPLIT, HEAD_DIM_PADDED],
                            [m_block * BLOCK_M, 0],
                            barrier=q_fulls[q0_idx],
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
                        else:
                            tle.gpu.copy(
                                k_desc,
                                k_smem.slot(kv_buf),
                                [BLOCK_N, HEAD_DIM_PADDED],
                                [kv_offset, 0],
                                barrier=k_fulls[kv_buf],
                            )

                        tle.gpu.barrier_wait(q_empties[q1_idx], phaseIdx=q_phase_idx)
                        tle.gpu.copy(
                            q_desc,
                            q_smem.slot(q1_idx),
                            [BM_SPLIT, HEAD_DIM_PADDED],
                            [m_block * BLOCK_M + BM_SPLIT, 0],
                            barrier=q_fulls[q1_idx],
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
                        else:
                            tle.gpu.copy(
                                v_desc,
                                v_smem.slot(kv_buf),
                                [BLOCK_N, HEAD_DIM_PADDED],
                                [kv_offset, 0],
                                barrier=v_fulls[kv_buf],
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
                            else:
                                tle.gpu.copy(
                                    k_desc,
                                    k_smem.slot(kv_buf),
                                    [BLOCK_N, HEAD_DIM_PADDED],
                                    [kv_offset, 0],
                                    barrier=k_fulls[kv_buf],
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
                            else:
                                tle.gpu.copy(
                                    v_desc,
                                    v_smem.slot(kv_buf),
                                    [BLOCK_N, HEAD_DIM_PADDED],
                                    [kv_offset, 0],
                                    barrier=v_fulls[kv_buf],
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

            if cid == 1:
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

                valid_q_tile = m_block * BLOCK_M < q_len
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
                        tle.gpu.barrier_wait(q_fulls[q_idx], phaseIdx=q_phase_idx)

                        kv_buf, kv_phase_idx = _buf_phase_tle(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        if is_paged:
                            tle.gpu.barrier_wait(
                                k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                            )
                        else:
                            tle.gpu.barrier_wait(
                                k_fulls[kv_buf], phaseIdx=kv_phase_idx
                            )

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
                            if is_paged:
                                tle.gpu.barrier_wait(
                                    k_fulls_manual[kv_buf], phaseIdx=kv_phase_idx
                                )
                            else:
                                tle.gpu.barrier_wait(
                                    k_fulls[kv_buf], phaseIdx=kv_phase_idx
                                )

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
                            if cid == 0:
                                tle.gpu.barrier_arrive(ping_to_c1)
                            else:
                                tle.gpu.barrier_arrive(ping_to_c0)

                            v_buf, v_phase_idx = _buf_phase_tle(
                                accum_cnt_kv - 1, NUM_BUFFERS_KV
                            )
                            if is_paged:
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
                        if is_paged:
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

                    o_desc.store(
                        [m_block * BLOCK_M + cid * BM_SPLIT, 0],
                        acc.to(o_ptr.dtype.element_ty),
                    )
                    lse_ptr = softmax_lse_ptr + hid * total_q + lse_offset + row_idx_q
                    tl.store(lse_ptr, lse, mask=row_idx_q < q_len)

                tile_idx += num_progs
