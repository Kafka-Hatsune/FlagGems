# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX MV: select a load orientation, then launch a tuned FP32 reduction.

Dispatch selects a load orientation and optional split reduction from strides
and parallelism. Standard libtuner selects tiles within each kernel.
Tensor strides and storage offsets are consumed directly. Only compiled code
and layout choices are cached; partial results are always call-local.
"""

import copy
import hashlib
import logging
from functools import cached_property, partial
from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl
from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.device_info import get_sm_count
from flag_gems.utils.libentry import LibTuner

logger = logging.getLogger(__name__)
_graph_bench = partial(triton.testing.do_bench_cudagraph, rep=20)


def _prune_tiles(configs, named_args, **kwargs):
    # AABS may shrink a candidate in-place; keep the search space for later shapes.
    args = {**named_args, **kwargs}
    m, k = args["M"], args["K"]
    return [
        copy.deepcopy(config)
        for config in configs
        if config.kwargs["BM"] <= max(16, triton.next_power_of_2(m))
        and config.kwargs["BK"] <= max(128, triton.next_power_of_2(k))
    ]


_KEY = ["M", "K", "BATCH", "SAB", "SAM", "SAK", "SXB", "SXK", "SYB", "SYM", "SPLIT_K"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_row"),
    key=["M", "K", "SAM", "SAK", "SXK", "SYM"],
    prune_configs_by={"early_config_prune": _prune_tiles},
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _mv_row_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SXK: tl.constexpr,
    SYM: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    # BM=1 is one output per CTA; larger BM shares x across several rows.
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    offsets = tl.arange(0, BK).to(tl.int64)
    acc = tl.zeros((BM, BK), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + offsets
        x = tl.load(X + ks * SXK, ks < K, other=0).to(tl.float32)
        a = tl.load(
            A + rows[:, None] * SAM + ks[None, :] * SAK,
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc = tl.fma(x[None, :], a, acc)
    tl.store(Y + rows * SYM, tl.sum(acc, axis=1), rows < M)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_column"),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_tiles},
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _mv_column_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SXB: tl.constexpr,
    SXK: tl.constexpr,
    SYB: tl.constexpr,
    SYM: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    part = tl.program_id(1)
    batch = tl.program_id(2)
    offsets = tl.arange(0, BK)
    # Rows are the inner dimension: contiguous column-major loads need no
    # materialized transpose, and each x element is reused across these rows.
    acc = tl.zeros((BK, BM), tl.float32)
    for start in range(part * BK, K, SPLIT_K * BK):
        ks = start + offsets
        a = tl.load(
            A + batch * SAB + ks[:, None] * SAK + rows[None, :] * SAM,
            (ks[:, None] < K) & (rows[None, :] < M),
            other=0,
        ).to(tl.float32)
        x = tl.load(X + batch * SXB + ks * SXK, ks < K, other=0).to(tl.float32)
        acc = tl.fma(a, x[:, None], acc)
    values = tl.sum(acc, axis=0)
    tl.store(Y + batch * SYB + part * M + rows * SYM, values, rows < M)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_reduce"),
    key=["M", "BATCH", "SPLIT_K", "SYB", "SYM"],
    prune_configs_by={
        "early_config_prune": lambda configs, named_args, **kw: copy.deepcopy(configs)
    },
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _mv_reduce_kernel(
    P,
    Y,
    M: tl.constexpr,
    BATCH: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SYB: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    parts = tl.arange(0, SPLIT_K)
    values = tl.load(
        P + batch * SPLIT_K * M + parts[:, None] * M + rows[None, :],
        rows[None, :] < M,
        other=0,
    )
    tl.store(Y + batch * SYB + rows * SYM, tl.sum(values, axis=0), rows < M)


@triton.jit
def _zero_kernel(Y, M: tl.constexpr, SYB: tl.constexpr, SYM: tl.constexpr):
    rows = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y + tl.program_id(1) * SYB + rows * SYM, 0.0, rows < M)


class _MvPlan(NamedTuple):
    """An immutable execution choice; never owns tensors or workspace."""

    launch: Callable
    split_k: int = 1


class _MvCall(NamedTuple):
    """Validated tensors and metadata for one invocation."""

    a: torch.Tensor
    x: torch.Tensor
    out: torch.Tensor
    m: int
    k: int
    a_strides: tuple
    x_stride: int
    out_stride: int
    key: tuple


# Capture readiness only; tile selection and its cache belong to libtuner.
_MV_WARMED = set()


def _plan_key(a, x, out):
    return (
        a.device,
        *a.shape,
        a.dtype,
        x.dtype,
        out.dtype,
        *a.stride(),
        x.stride(0),
        out.stride(0),
        a.data_ptr() % 16,
        x.data_ptr() % 16,
        out.data_ptr() % 16,
    )


def _byte_span(tensor):
    start = tensor.data_ptr()
    if tensor.numel() == 0:
        return start, start
    last = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return start, start + (last + 1) * tensor.element_size()


def _validate_mv(input, vec, out, *, allow_fp32_output=False):
    """Establish the complete call contract before dispatch or GPU execution.

    A missing output is allocated here so its layout/alignment are also known.
    This allocates storage only; no input is copied or materialized.
    """
    if input.layout != torch.strided or vec.layout != torch.strided:
        raise NotImplementedError("MetaX mv supports strided dense tensors")
    if input.ndim != 2 or vec.ndim != 1 or input.shape[1] != vec.shape[0]:
        raise RuntimeError("mv expects a matrix [M, K] and a vector [K]")
    if input.device != vec.device or input.dtype != vec.dtype:
        raise RuntimeError("mv inputs must have the same device and dtype")
    if input.device.type != "cuda":
        raise NotImplementedError("MetaX mv requires a MetaX GPU")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("MetaX mv supports float16, bfloat16 and float32")

    m, k = input.shape
    tensors = (input, vec) if out is None else (input, vec, out)
    if out is not None:
        if out.layout != torch.strided:
            raise NotImplementedError("mv out must be a strided dense tensor")
        valid_dtype = out.dtype == input.dtype or (
            allow_fp32_output and out.dtype == torch.float32
        )
        if out.shape != (m,) or not valid_dtype or out.device != input.device:
            raise RuntimeError(
                "mv out must have shape [M] and a compatible dtype/device"
            )
        if m > 1 and out.stride(0) == 0:
            raise NotImplementedError("mv out must not have overlapping elements")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise NotImplementedError("MetaX mv is inference-only; use torch.no_grad()")
    # Only real dtypes are accepted above, so a separate conjugate check is redundant.
    if any(t.is_neg() or any(s < 0 for s in t.stride()) for t in tensors):
        raise NotImplementedError(
            "MetaX mv does not support lazy negative views or negative strides"
        )
    if out is not None:
        lo, hi = _byte_span(out)
        for tensor in (input, vec):
            other_lo, other_hi = _byte_span(tensor)
            if max(lo, other_lo) < min(hi, other_hi):
                raise NotImplementedError(
                    "mv out must not overlap an input storage span"
                )
    else:
        out = torch.empty((m,), device=input.device, dtype=input.dtype)

    key = _plan_key(input, vec, out)
    # A successful prior launch warmed every participating tuner outside capture.
    # Empty/zero products do not autotune and can be captured on their first call.
    if m and k and key not in _MV_WARMED:
        with torch_device_fn.device(input.device):
            if torch_device_fn.is_current_stream_capturing():
                raise RuntimeError("warm MetaX mv before CUDA Graph capture")
    return _MvCall(
        input,
        vec,
        out,
        m,
        k,
        input.stride(),
        vec.stride(0),
        out.stride(0),
        key,
    )


def _split_count(m, k):
    """Estimate parallelism using a 128-row tile and about four CTAs per SM."""
    if k < 1024:
        return 1
    # Plain host arithmetic avoids constexpr_function wrappers on every call.
    output_tiles = (m + 127) // 128
    occupancy_target = (4 * get_sm_count() + output_tiles - 1) // output_tiles
    work_limit = (k + 63) // 64
    occupancy = 1 << (occupancy_target - 1).bit_length()
    reduction_work = 1 << (work_limit - 1).bit_length()
    return min(128, occupancy, reduction_work)


def _dispatch_mv(call):
    """Select the algorithm from metadata; never launch or benchmark candidates."""
    if not call.m:
        return _MvPlan(_launch_empty)
    if not call.k:
        return _MvPlan(_launch_zero)
    if call.a_strides[0] >= call.a_strides[1]:
        return _MvPlan(_launch_row)
    # Column loads expose fewer output CTAs; split long reductions for parallelism.
    return _MvPlan(_launch_column, _split_count(call.m, call.k))


def _launch_empty(call, split_k):
    pass


def _launch_zero(call, split_k):
    _zero_kernel[(triton.cdiv(call.m, 256), 1)](
        call.out, call.m, 0, call.out_stride, num_warps=4
    )


def _launch_row(call, split_k):
    _mv_row_kernel[lambda cfg: (triton.cdiv(call.m, cfg["BM"]),)](
        call.a,
        call.x,
        call.out,
        call.m,
        call.k,
        *call.a_strides,
        call.x_stride,
        call.out_stride,
    )


def _launch_column(call, split_k):
    if split_k > 1:
        target = torch.empty(
            (split_k, call.m), device=call.a.device, dtype=torch.float32
        )
        target_strides = (split_k * call.m, 1)
    else:
        target, target_strides = call.out, (0, call.out_stride)
    _mv_column_kernel[lambda cfg: (triton.cdiv(call.m, cfg["BM"]), split_k, 1)](
        call.a,
        call.x,
        target,
        call.m,
        call.k,
        1,
        0,
        *call.a_strides,
        0,
        call.x_stride,
        *target_strides,
        SPLIT_K=split_k,
    )
    if split_k > 1:
        _mv_reduce_kernel[lambda cfg: (triton.cdiv(call.m, cfg["BLOCK"]), 1)](
            target,
            call.out,
            call.m,
            1,
            split_k,
            0,
            call.out_stride,
        )


def _launch_mv(plan, call):
    """Execute exactly the selected plan; no validation or workload classification."""
    plan.launch(call, plan.split_k)
    _MV_WARMED.add(call.key)
    return call.out


def mv(input, vec, *, out=None):
    """Validate the call, dispatch its workload, then launch the selected plan."""
    logger.debug("GEMS METAX MV")
    call = _validate_mv(input, vec, out)
    with torch_device_fn.device(input.device):
        plan = _dispatch_mv(call)
        return _launch_mv(plan, call)


def _mv_for_bmm(input, vec, out):
    """MV entry for BMM's additional half-input / FP32-output contract."""
    call = _validate_mv(input, vec, out, allow_fp32_output=True)
    with torch_device_fn.device(input.device):
        plan = _dispatch_mv(call)
        return _launch_mv(plan, call)


def _selected_config(a, x, out):
    """Read the exact libtuner cache entry after a successful warmup."""
    m, k = a.shape
    if not m or not k:
        return dict(algorithm="empty" if not m else "zero")
    if a.stride(0) >= a.stride(1):
        split, family, kernel = 1, "row", _mv_row_kernel
        key = (m, k, *a.stride(), x.stride(0), out.stride(0))
    else:
        split, family, kernel = _split_count(m, k), "column", _mv_column_kernel
        strides = (split * m, 1) if split > 1 else (0, out.stride(0))
        key = (m, k, 1, 0, *a.stride(), 0, x.stride(0), *strides, split)
    key += (str(a.dtype), str(x.dtype), str(torch.float32 if split > 1 else out.dtype))
    return dict(algorithm="mv_" + family, split_k=split, best=str(kernel.fn.cache[key]))


# Compatibility primitives used by BMM. Public MV uses only the standard
# libtuner kernels above; these existing batched candidates are unchanged.


class _MvTuner(LibTuner):
    """Legacy explicit-tile policy for the shared batched GEMV primitive only."""

    @cached_property
    def cache_key(self):
        return hashlib.sha256(
            (super().cache_key + "mv_explicit_v1").encode()
        ).hexdigest()

    def _bench(self, *args, config, **meta):
        options = {**meta, **config.all_kwargs(), "warmup": False}
        try:
            return self.do_bench(
                lambda: self.fn.run(*args, **options), quantiles=(0.5, 0.2, 0.8)
            )
        except triton.OutOfResources:
            return [float("inf")] * 3

    def policy(self, bench, configs, args, kwargs):
        timings = {config: bench(config) for config in configs}
        if not timings or min(timings.values())[0] == float("inf"):
            raise RuntimeError("no MetaX mv candidate fits this workload")
        return min(timings, key=timings.get), timings


@triton.jit
def _gemv_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAB: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBB: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCB: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    BATCH: tl.constexpr = 1,
):
    """One output row per CTA group; accumulate K locally before a lane reduction."""
    nn = tl.cdiv(N, BN)
    batch = (tl.program_id(0) // (M * nn * SPLIT_K)).to(tl.int64)
    split = tl.program_id(0) // (M * nn) % SPLIT_K
    m = (tl.program_id(0) // nn % M).to(tl.int64)
    n = (tl.program_id(0) % nn * BN + tl.arange(0, BN)).to(tl.int64)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    k = tl.arange(0, BK) + split.to(tl.int64) * iterations * BK
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(iterations):
        kk = k + start * BK
        a = tl.load(A + batch * SAB + m * SAM + kk * SAK, kk < K, other=0).to(
            tl.float32
        )
        b = tl.load(
            B + batch * SBB + n[:, None] * SBN + kk[None, :] * SBK,
            (n[:, None] < N) & (kk[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc = tl.fma(a[None, :], b, acc)
    total = tl.sum(acc, axis=1)
    tl.store(
        C + split.to(tl.int64) * BATCH * M * N + batch * SCB + m * SCM + n * SCN,
        total,
        n < N,
    )


def _prune_gemv(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    n, k = args["N"], args["K"]
    result = []
    for config in configs:
        bn, bk = config.kwargs["BN"], config.kwargs["BK"]
        if bn > triton.next_power_of_2(n) or bk > max(64, triton.next_power_of_2(k)):
            continue
        if args.get("SPLIT_K", 1) > 1 and bk > triton.cdiv(k, args["SPLIT_K"]):
            continue
        result.append(config)
    return result


_NARROW_KEY = [
    "BATCH",
    "M",
    "N",
    "K",
    "SAB",
    "SAM",
    "SAK",
    "SBB",
    "SBK",
    "SBN",
    "SCB",
    "SCM",
    "SCN",
    "SPLIT_K",
]

# The same SIMT primitive also accepts narrow batched products from BMM.
_gemv_tuned = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("mv_simt"),
        key=_NARROW_KEY,
        prune_configs_by={"early_config_prune": _prune_gemv},
        do_bench=_graph_bench,
        rep=20,
        policy=_MvTuner,
    )(_gemv_kernel)
)
