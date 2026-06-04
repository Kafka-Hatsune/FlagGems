"""
Host-side launcher for the TLE-only Hopper FA3 varlen forward kernel.

The public signature and return tuple intentionally match
``flag_gems.ops.flash_api.mha_varlan_fwd`` so the Hopper attention override can
dispatch to this file for ``fa_version=3``.  Unsupported FA3 inputs fail fast
with a clear RuntimeError; the older non-TLE FA3 kernel is intentionally absent.
"""

import logging

import torch
import triton

import flag_gems
from flag_gems.ops.flash_api import fwd_params
from flag_gems.runtime import torch_device_fn

from .flash_kernel_v3 import (
    TLE_FA3_AVAILABLE,
    fa3_tle_force_family_id,
    fa3_tle_mixed_long_plan,
    fa3_tle_select_plan,
    flash_varlen_fwd_v3_tle_kernel,
)

logger = logging.getLogger(__name__)

_TMA_ALLOCATOR_REGISTERED = False


def _check_device(x):
    if x.device.type != flag_gems.device:
        raise RuntimeError(f"expected {flag_gems.device} tensor, got {x.device}")


def _ensure_tma_allocator():
    global _TMA_ALLOCATOR_REGISTERED
    if _TMA_ALLOCATOR_REGISTERED:
        return
    if not hasattr(triton, "set_allocator"):
        raise RuntimeError(
            "TLE FA3 requires Triton with on-device TMA descriptors "
            "(missing triton.set_allocator)."
        )

    def _alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc_fn)
    _TMA_ALLOCATOR_REGISTERED = True


def is_fa3_supported() -> bool:
    if not TLE_FA3_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        import triton.language as tl

        return hasattr(tl, "make_tensor_descriptor") and hasattr(
            triton, "set_allocator"
        )
    except Exception:
        return False


def _round_multiple(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _tma_strides_are_aligned(tensor: torch.Tensor) -> bool:
    elem_bytes = tensor.element_size()
    return all((stride * elem_bytes) % 16 == 0 for stride in tensor.stride()[:-1])


def _require_tle_supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    return_softmax,
    leftpad_k,
):
    if not is_fa3_supported():
        raise RuntimeError(
            "TLE FA3 requires CUDA Hopper, Triton TMA descriptors, and "
            "triton.experimental.tle."
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "TLE FA3 currently supports torch.float16 and torch.bfloat16 inputs only."
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("TLE FA3 requires q, k, and v to have the same dtype.")
    if p_dropout != 0:
        raise RuntimeError("TLE FA3 does not support dropout.")
    if return_softmax:
        raise RuntimeError("TLE FA3 does not support returning the softmax matrix.")
    if leftpad_k is not None:
        raise RuntimeError("TLE FA3 does not support leftpad_k.")
    if q.ndim != 3:
        raise RuntimeError(f"TLE FA3 expects q with shape (total_q, h, d), got {q.shape}.")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise RuntimeError("TLE FA3 requires q/k/v to be contiguous in the head dimension.")
    if cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_contiguous():
        raise RuntimeError("TLE FA3 requires contiguous int32 cu_seqlens_q.")
    if cu_seqlens_k.dtype != torch.int32 or not cu_seqlens_k.is_contiguous():
        raise RuntimeError("TLE FA3 requires contiguous int32 cu_seqlens_k placeholder.")
    if seqused_k is not None and (
        seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous()
    ):
        raise RuntimeError("TLE FA3 requires contiguous int32 seqused_k when provided.")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise RuntimeError("TLE FA3 requires positive max_seqlen_q and max_seqlen_k.")

    head_size = q.shape[-1]
    if head_size < 32 or head_size > 256 or head_size % 8 != 0:
        raise RuntimeError("TLE FA3 requires 32 <= head_dim <= 256 and head_dim % 8 == 0.")

    is_paged = page_table is not None
    if is_paged:
        if page_table.dtype != torch.int32 or page_table.ndim != 2:
            raise RuntimeError("TLE FA3 paged mode requires an int32 2D block table.")
        if page_table.stride(-1) != 1:
            raise RuntimeError("TLE FA3 paged mode requires contiguous block-table rows.")
        if seqused_k is None:
            raise RuntimeError("TLE FA3 paged mode requires seqused_k.")
        if k.ndim != 4 or v.ndim != 4:
            raise RuntimeError("TLE FA3 paged mode expects k/v cache shape (pages, block, hk, d).")
    else:
        if k.ndim != 3 or v.ndim != 3:
            raise RuntimeError("TLE FA3 dense mode expects k/v shape (total_k, hk, d).")

    if out is not None and out.dtype != q.dtype:
        raise RuntimeError("TLE FA3 requires output dtype to match q.")
    if alibi_slopes is not None:
        if alibi_slopes.dtype != torch.float32 or alibi_slopes.stride(-1) != 1:
            raise RuntimeError("TLE FA3 requires fp32 ALiBi slopes with last stride 1.")


def mha_varlan_fwd_v3(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    _check_device(q)
    _check_device(k)
    _check_device(v)
    q_device = q.device
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    _require_tle_supported(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        page_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        return_softmax,
        leftpad_k,
    )

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    page_table_batch_stride = page_table.stride(0)

    if k.size() != v.size():
        raise RuntimeError("TLE FA3 requires k and v to have the same shape.")
    if cu_seqlens_q.size() != (batch_size + 1,):
        raise RuntimeError("cu_seqlens_q must have shape (batch_size + 1,).")
    if cu_seqlens_k.size() != (batch_size + 1,):
        raise RuntimeError("cu_seqlens_k must have shape (batch_size + 1,).")
    if seqused_k is not None and seqused_k.size() != (batch_size,):
        raise RuntimeError("seqused_k must have shape (batch_size,).")
    if num_heads % num_heads_k != 0:
        raise RuntimeError("TLE FA3 requires num_heads % num_heads_k == 0.")

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1
    is_local = window_size_left >= 0

    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)
    if q.shape != (total_q, num_heads, head_size):
        raise RuntimeError("internal TLE FA3 q shape mismatch after optional swap.")
    if is_paged:
        expected = (num_pages, block_size, num_heads_k, head_size)
        if k.shape != expected or v.shape != expected:
            raise RuntimeError(f"TLE FA3 expected paged k/v shape {expected}.")
    if k.stride() != v.stride():
        raise RuntimeError("TLE FA3 requires k and v to have matching strides.")

    if alibi_slopes is not None:
        if alibi_slopes.device != q_device:
            raise RuntimeError("ALiBi slopes must be on the same device as q.")
        if alibi_slopes.shape == (num_heads,):
            alibi_slopes_batch_stride = 0
        elif alibi_slopes.shape == (batch_size, num_heads):
            alibi_slopes_batch_stride = alibi_slopes.stride(0)
        else:
            raise RuntimeError(
                "ALiBi slopes must have shape (num_heads,) or (batch_size, num_heads)."
            )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    if softcap > 0.0:
        is_softcap = True
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax = softcap
        adjusted_scale_softmax_log2e = softcap * 1.4426950408889634
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * 1.4426950408889634

    head_size_rounded = _round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = _round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = _round_multiple(max_seqlen_k, 32)

    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q)
        else:
            out_ = None
            out = torch.empty_like(q)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q_device)
        p = torch.empty((), device=q_device)
        philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        params = fwd_params(
            q,
            k,
            v,
            out,
            p,
            lse,
            q.stride(-3),
            k.stride(-3),
            v.stride(-3),
            q.stride(-2),
            k.stride(-2),
            v.stride(-2),
            out.stride(-3),
            out.stride(-2),
            q_batch_stride,
            k_batch_stride,
            v_batch_stride,
            o_batch_stride,
            cu_seqlens_q is not None,
            cu_seqlens_q,
            seqused_k is None,
            cu_seqlens_k,
            seqused_k is not None,
            seqused_k,
            batch_size,
            k_batch_size,
            num_heads,
            num_heads_k,
            num_heads // num_heads_k,
            max_seqlen_q,
            max_seqlen_k,
            seqlen_q_rounded,
            seqlen_k_rounded,
            head_size,
            head_size_rounded,
            is_softcap,
            adjusted_softcap,
            adjusted_scale_softmax,
            adjusted_scale_softmax_log2e,
            False,
            0.0,
            1.0,
            255,
            philox_args,
            False,
            is_causal,
            is_local,
            window_size_left,
            window_size_right,
            seqlenq_ngroups_swapped,
            is_paged,
            is_alibi,
            alibi_slopes,
            alibi_slopes_batch_stride,
            total_q,
            page_table,
            page_table_batch_stride,
            block_size,
        )

        _ensure_tma_allocator()
        num_sms = torch.cuda.get_device_properties(q_device).multi_processor_count
        args = tuple(getattr(params, k) for k in params.__slots__)
        force_family_id = fa3_tle_force_family_id()
        plan = fa3_tle_select_plan(
            total_q=total_q,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_paged=is_paged,
            force_family_id=force_family_id,
        )

        def _run_plan(run_plan):
            if run_plan.family in ("long", "mixed_long"):
                if (
                    not _tma_strides_are_aligned(q)
                    or not _tma_strides_are_aligned(out)
                    or (not is_paged and not _tma_strides_are_aligned(k))
                    or (not is_paged and not _tma_strides_are_aligned(v))
                ):
                    raise RuntimeError(
                        "TLE FA3 long/TMA path requires 16-byte aligned Q/K/V/O strides."
                    )

            grid = lambda meta: (
                min(
                    num_sms,
                    triton.cdiv(max_seqlen_q, meta["BLOCK_M"])
                    * batch_size
                    * num_heads,
                ),
            )
            logger.debug(
                "kernel: flash_varlen_fwd_v3_tle family=%s bucket=%s q_len=[%s,%s]",
                run_plan.family,
                run_plan.shape_bucket,
                run_plan.min_q_len,
                run_plan.max_q_len,
            )
            flash_varlen_fwd_v3_tle_kernel[grid](
                *args,
                SHAPE_BUCKET=run_plan.shape_bucket,
                FORCE_FAMILY_ID=run_plan.force_family_id,
                MIN_Q_LEN_TO_PROCESS=run_plan.min_q_len,
                MAX_Q_LEN_TO_PROCESS=run_plan.max_q_len,
                tle_wgmma_pipeline_mode="user_promise",
            )

        if plan.family == "mixed":
            _run_plan(fa3_tle_mixed_long_plan(force_family_id))
            _run_plan(plan)
        else:
            _run_plan(plan)

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        unused = torch.empty((), dtype=torch.int64, device=q_device)

    return out, q, k, v, lse, philox_args, unused, p
