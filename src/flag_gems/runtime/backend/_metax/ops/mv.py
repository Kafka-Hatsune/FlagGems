# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX MV: select a load orientation, then launch a tuned FP32 reduction.

K-contiguous inputs compare three equivalent reduction layouts. Inputs whose
output rows are contiguous use column loads and, for long K, split reduction.
Tensor strides and storage offsets are consumed directly. Only compiled code
and layout choices are cached; partial results are always call-local.
"""

import hashlib
import logging
import threading
from functools import cached_property, partial

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


class _MvTuner(LibTuner):
    """Measure the configured tile without FlagTree AABS rewriting its dimensions."""

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


def _prune_tiles(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, k = args["M"], args["K"]
    return [
        config
        for config in configs
        if config.kwargs["BM"] <= max(16, triton.next_power_of_2(m))
        and config.kwargs["BK"] <= max(128, triton.next_power_of_2(k))
    ]


_KEY = ["M", "K", "BATCH", "SAB", "SAM", "SAK", "SXB", "SXK", "SYB", "SYM", "SPLIT_K"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_row"),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_tiles},
    do_bench=_graph_bench,
    policy=_MvTuner,
)
@triton.jit
def _row_kernel(
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
    acc = tl.zeros((BM, BK), tl.float32)
    for start in range(part * BK, K, SPLIT_K * BK):
        ks = start + offsets
        a = tl.load(
            A + batch * SAB + rows[:, None] * SAM + ks[None, :] * SAK,
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        x = tl.load(X + batch * SXB + ks * SXK, ks < K, other=0).to(tl.float32)
        acc = tl.fma(a, x[None, :], acc)
    values = tl.sum(acc, axis=1)
    tl.store(Y + batch * SYB + part * M + rows * SYM, values, rows < M)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_column"),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_tiles},
    do_bench=_graph_bench,
    policy=_MvTuner,
)
@triton.jit
def _column_kernel(
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
    do_bench=_graph_bench,
    policy=_MvTuner,
)
@triton.jit
def _reduce_kernel(
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


# Layout tuning caches metadata, never tensor data or temporary buffers.
_ROW_PLANS = {}
_ROW_PLAN_LOCK = threading.RLock()


def _row_plan_key(a, x, out):
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


def _row_args(family, a, x, out):
    """Map the same MV onto either orientation of the shared SIMT primitive."""
    m, k = a.shape
    am, ak = a.stride()
    xk, ym = x.stride(0), out.stride(0)
    if family == "matrix_vector":
        return (a, x, out, m, 1, k, 0, am, ak, 0, xk, 0, 0, ym, 0)
    return (x, a, out, 1, m, k, 0, 0, xk, 0, ak, am, 0, 0, ym)


def _launch_row(family, a, x, out):
    m, k = a.shape
    if family == "row":
        _row_kernel[lambda cfg: (triton.cdiv(m, cfg["BM"]), 1, 1)](
            a,
            x,
            out,
            m,
            k,
            1,
            0,
            *a.stride(),
            0,
            x.stride(0),
            0,
            out.stride(0),
            SPLIT_K=1,
        )
    else:
        args = _row_args(family, a, x, out)
        rows, columns = args[3:5]
        _gemv_tuned[lambda cfg: (rows * triton.cdiv(columns, cfg["BN"]),)](
            *args,
            BATCH=1,
            SPLIT_K=1,
        )


def _split_count(m, k):
    """Fill four CTA slots per SM without creating excessive partial storage."""
    if k < 1024:
        return 1
    output_tiles = triton.cdiv(m, 128)
    occupancy = triton.next_power_of_2(triton.cdiv(4 * get_sm_count(), output_tiles))
    reduction_work = triton.next_power_of_2(triton.cdiv(k, 64))
    return min(128, occupancy, reduction_work)


def _launch_gemv(a, x, out):
    """Execute a validated MV; BMM may also supply an FP32 output tensor.

    Call under the input's device context. The public MV API below owns its
    same-dtype checks; this shared executor owns layout selection and launches.
    """
    m, k = a.shape
    if not m:
        return
    if not k:
        _zero_kernel[(triton.cdiv(m, 256), 1)](out, m, 0, out.stride(0), num_warps=4)
        return

    if a.stride(0) >= a.stride(1):
        key = _row_plan_key(a, x, out)
        family = _ROW_PLANS.get(key)
        if family is None:
            with _ROW_PLAN_LOCK:
                family = _ROW_PLANS.get(key)
                if family is None:
                    if torch_device_fn.is_current_stream_capturing():
                        raise RuntimeError("warm MetaX mv before CUDA Graph capture")
                    candidates = ("row", "matrix_vector", "vector_matrix")
                    for candidate in candidates:
                        _launch_row(candidate, a, x, out)
                    timings = {
                        candidate: _graph_bench(
                            lambda candidate=candidate: _launch_row(
                                candidate, a, x, out
                            ),
                            quantiles=(0.5, 0.2, 0.8),
                        )[0]
                        for candidate in candidates
                    }
                    family = _ROW_PLANS[key] = min(timings, key=timings.get)
        _launch_row(family, a, x, out)
        return

    split = _split_count(m, k)
    if split > 1:
        target = torch.empty((split, m), device=a.device, dtype=torch.float32)
        target_strides = (split * m, 1)
    else:
        target, target_strides = out, (0, out.stride(0))
    _column_kernel[lambda cfg: (triton.cdiv(m, cfg["BM"]), split, 1)](
        a,
        x,
        target,
        m,
        k,
        1,
        0,
        *a.stride(),
        0,
        x.stride(0),
        *target_strides,
        SPLIT_K=split,
    )
    if split > 1:
        _reduce_kernel[lambda cfg: (triton.cdiv(m, cfg["BLOCK"]), 1)](
            target,
            out,
            m,
            1,
            split,
            0,
            out.stride(0),
        )


def _byte_span(tensor):
    start = tensor.data_ptr()
    if tensor.numel() == 0:
        return start, start
    last = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return start, start + (last + 1) * tensor.element_size()


def mv(input, vec, *, out=None):
    """Compute a strided matrix-vector product in FP32 and round once to output."""
    logger.debug("GEMS METAX MV")
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
    tensors = (input, vec) if out is None else (input, vec, out)
    if out is not None and out.layout != torch.strided:
        raise NotImplementedError("mv out must be a strided dense tensor")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise NotImplementedError("MetaX mv is inference-only; use torch.no_grad()")
    if any(
        t.is_conj() or t.is_neg() or any(s < 0 for s in t.stride()) for t in tensors
    ):
        raise NotImplementedError(
            "MetaX mv does not support lazy conjugate/negative views"
        )

    m, k = input.shape
    if out is None:
        out = torch.empty((m,), device=input.device, dtype=input.dtype)
    else:
        if out.shape != (m,) or out.dtype != input.dtype or out.device != input.device:
            raise RuntimeError(
                "mv out must have shape [M] and match the input dtype/device"
            )
        if m > 1 and out.stride(0) == 0:
            raise NotImplementedError("mv out must not have overlapping elements")
        lo, hi = _byte_span(out)
        for tensor in (input, vec):
            other_lo, other_hi = _byte_span(tensor)
            if max(lo, other_lo) < min(hi, other_hi):
                raise NotImplementedError(
                    "mv out must not overlap an input storage span"
                )

    with torch_device_fn.device(input.device):
        _launch_gemv(input, vec, out)
    return out


def _selected_config(a, x, out):
    """Inspect the chosen layout and tile after a warmup call."""
    m, k = a.shape
    if not m or not k:
        return dict(algorithm="empty" if not m else "zero")
    column_major = a.stride(0) < a.stride(1)
    if not column_major:
        family = _ROW_PLANS[_row_plan_key(a, x, out)]
        if family != "row":
            args = _row_args(family, a, x, out)
            key = (1, *args[3:], 1, str(a.dtype), str(x.dtype), str(out.dtype))
            return dict(
                algorithm="mv_" + family, split_k=1, best=str(_gemv_tuned.fn.cache[key])
            )
    split = _split_count(m, k) if column_major else 1
    strides = (split * m, 1) if split > 1 else (0, out.stride(0))
    key = (m, k, 1, 0, *a.stride(), 0, x.stride(0), *strides, split)
    key += (str(a.dtype), str(x.dtype), str(torch.float32 if split > 1 else out.dtype))
    kernel = _column_kernel if column_major else _row_kernel
    return dict(
        algorithm="mv_column" if column_major else "mv_row",
        split_k=split,
        best=str(kernel.fn.cache[key]),
    )
