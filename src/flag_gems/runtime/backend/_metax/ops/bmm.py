# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stride-aware MetaX BMM: dispatch tensor metadata, then launch a tuned plan."""

import copy
import logging
import os
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTuner

from ..device_info import MetaXDeviceInfo, device_info
from .mm import mm

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "bmm_metax_expand.yaml")
)


@triton.jit
def _bmm_kernel(
    A,
    B,
    C,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SCB: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr = 1,
    SPLIT_K: tl.constexpr = 1,
    TRANSPOSE: tl.constexpr = False,
    STATIC_K: tl.constexpr = False,
    INTERLEAVE: tl.constexpr = False,
):
    """One output tile per CTA, optionally with a deterministic K partition."""
    tiles_m = tl.cdiv(N if TRANSPOSE else M, BM)
    tiles_n = tl.cdiv(M if TRANSPOSE else N, BN)
    tiles = tiles_m * tiles_n * SPLIT_K
    if INTERLEAVE:
        batch = tl.program_id(0) % BATCH
        tile_id = tl.program_id(0) // BATCH
    else:
        batch = tl.program_id(0) // tiles
        tile_id = tl.program_id(0) % tiles
    A += batch.to(tl.int64) * SAB
    B += batch.to(tl.int64) * SBB
    C += batch.to(tl.int64) * SCB
    if TRANSPOSE:
        # Compute C^T = B^T A^T. Swapping the dot operands changes which
        # operand feeds which MMA port without materializing a transpose.
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tile_id // (nm * nn) % SPLIT_K
    pid = tile_id % (nm * nn)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % group_m
    pn = local // group_m
    mi = pm * BM + tl.arange(0, BM)
    ni = pn * BN + tl.arange(0, BN)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    ki = tl.arange(0, BK) + split * iterations * BK
    # Tell the vectorizer each axis is a dense power-of-two tile. The values
    # do not change; masked tails still compare against M/N/K below.
    mi = tl.max_contiguous(tl.multiple_of(mi, BM), BM)
    ni = tl.max_contiguous(tl.multiple_of(ni, BN), BN)
    ki = tl.max_contiguous(tl.multiple_of(ki, BK), BK)
    ap = A + mi[:, None].to(tl.int64) * SAM + ki[None, :].to(tl.int64) * SAK
    bp = B + ki[:, None].to(tl.int64) * SBK + ni[None, :].to(tl.int64) * SBN
    acc = tl.zeros((BM, BN), tl.float32)
    if STATIC_K:
        # Short reductions cannot amortize a pipelined loop's prologue and
        # epilogue. Keep addresses in int64 here too, including sliced views.
        for k in tl.static_range((K + BK * SPLIT_K - 1) // (BK * SPLIT_K)):
            if K % (BK * SPLIT_K) == 0:
                if M % BM == 0:
                    ak = tl.load(ap)
                else:
                    ak = tl.load(ap, mi[:, None] < M, other=0)
                if N % BN == 0:
                    bk = tl.load(bp)
                else:
                    bk = tl.load(bp, ni[None, :] < N, other=0)
            else:
                ak = tl.load(
                    ap, (mi[:, None] < M) & (ki[None, :] + k * BK < K), other=0
                )
                bk = tl.load(
                    bp, (ki[:, None] + k * BK < K) & (ni[None, :] < N), other=0
                )
            acc = tl.dot(ak, bk, acc, out_dtype=tl.float32, allow_tf32=False)
            ap += BK * SAK
            bp += BK * SBK
    else:
        for k in range(iterations):
            if K % (BK * SPLIT_K) == 0:
                if M % BM == 0:
                    ak = tl.load(ap)
                else:
                    ak = tl.load(ap, mi[:, None] < M, other=0)
                if N % BN == 0:
                    bk = tl.load(bp)
                else:
                    bk = tl.load(bp, ni[None, :] < N, other=0)
            else:
                ak = tl.load(
                    ap, (mi[:, None] < M) & (ki[None, :] + k * BK < K), other=0
                )
                bk = tl.load(
                    bp, (ki[:, None] + k * BK < K) & (ni[None, :] < N), other=0
                )
            acc = tl.dot(ak, bk, acc, out_dtype=tl.float32, allow_tf32=False)
            ap += BK * SAK
            bp += BK * SBK
    cp = (
        C
        + split.to(tl.int64) * M * N
        + mi[:, None].to(tl.int64) * SCM
        + ni[None, :].to(tl.int64) * SCN
    )
    tl.store(cp, acc, (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _bmm_long_k_tf32(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SAM: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
):
    # The public dispatch enables this approximate mode only when TF32 is
    # allowed and the long reduction passed the repository accuracy checks.
    tiles = tl.cdiv(M, 128) * tl.cdiv(N, 128)
    batch = (tl.program_id(0) // tiles).to(tl.int64)
    pid = tl.program_id(0) % tiles
    mi = (pid // tl.cdiv(N, 128) * 128 + tl.arange(0, 128)).to(tl.int64)
    ni = (pid % tl.cdiv(N, 128) * 128 + tl.arange(0, 128)).to(tl.int64)
    ki = tl.arange(0, 32).to(tl.int64)
    ap = A + batch * SAB + mi[:, None] * SAM + ki[None, :]
    bp = B + batch * SBB + ki[:, None] * SBK + ni[None, :] * SBN
    acc = tl.zeros((128, 128), tl.float32)
    for _ in range(K // 32):
        ak = tl.load(ap, mi[:, None] < M, 0)
        bk = tl.load(bp, ni[None, :] < N, 0)
        acc = tl.dot(ak, bk, acc, input_precision="tf32")
        ap += 32
        bp += 32 * SBK
    tl.store(
        C + batch * M * N + mi[:, None] * N + ni[None, :],
        acc,
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@triton.jit
def _bmm_dual_n(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    # Compose a 192-column tile from two legal power-of-two dot outputs.
    # Both dots share A; aligned column-major B avoids the spill-heavy case.
    pid = tl.program_id(0)
    mi = (pid // tl.cdiv(N, 192) * 256 + tl.arange(0, 256)).to(tl.int64)
    n0 = (pid % tl.cdiv(N, 192) * 192 + tl.arange(0, 128)).to(tl.int64)
    n1 = (pid % tl.cdiv(N, 192) * 192 + 128 + tl.arange(0, 64)).to(tl.int64)
    ki = tl.arange(0, 32).to(tl.int64)
    c0 = tl.zeros((256, 128), tl.float32)
    c1 = tl.zeros((256, 64), tl.float32)
    for i in range(K // 32):
        kk = i * 32 + ki
        ak = tl.load(A + mi[:, None] * K + kk[None, :], mi[:, None] < M, 0)
        b0 = tl.load(B + kk[:, None] + n0[None, :] * K)
        b1 = tl.load(B + kk[:, None] + n1[None, :] * K)
        c0 = tl.dot(ak, b0, c0)
        c1 = tl.dot(ak, b1, c1)
    tl.store(C + mi[:, None] * N + n0[None, :], c0, mi[:, None] < M)
    tl.store(C + mi[:, None] * N + n1[None, :], c1, mi[:, None] < M)


def _prune_bmm(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    size = args["A"].element_size()
    shared_bytes = device_info.for_device(args["A"].device).shared_bytes
    result = []
    for cfg in configs:
        q = cfg.kwargs
        bm, bn, bk = q["BM"], q["BN"], q["BK"]
        mt, nt = (n, m) if q["TRANSPOSE"] else (m, n)
        # Dense outputs already expose many CTAs. Tiny output tiles only
        # repeat operand traffic, especially for the long-K model workloads.
        if min(m, n) >= 1024 and bm * bn < 4096:
            continue
        if size == 2 and min(m, n) >= 1024 and k >= 8192 and bm * bn < 16384:
            continue
        if q["STATIC_K"] and (k > 128 or min(m, n) < 64):
            continue
        if bm > max(32, triton.next_power_of_2(mt)) or bn > max(
            32, triton.next_power_of_2(nt)
        ):
            continue
        if (bm + bn) * bk * size * (1 if cfg.num_stages == 1 else 2) > shared_bytes:
            continue
        if q["scenario"] == "unprefetch" and (mt % bm or nt % bn or k % bk):
            continue
        if bm >= 256 and bn >= 256 and args["SBK"] == 1:
            if not (
                q["scenario"] == "reduceSmemUsage"
                and size == 2
                and args["SAK"] == 1
                and args["SCN"] == 1
                and nt % bn == 0
                and k % bk == 0
                and not q["TRANSPOSE"]
                and args["C"].dtype == args["A"].dtype
            ):
                continue
        if q["scenario"] == "reduceSmemUsage" and not (size == 2 and args["SBK"] == 1):
            continue
        if q["TRANSPOSE"] and not (
            args.get("wide", False)
            or args["SAM"] == 1
            or (
                args["SBN"] == 1
                and (k <= 256 or (size == 2 and args["SAK"] == 1 and min(m, n) >= 64))
            )
        ):
            continue
        if (
            bm == bn == 128
            and cfg.num_warps == 8
            and q["scenario"] == "unprefetch"
            and cfg.num_stages > 1
            and args["SBK"] != 1
        ):
            continue
        result.append(copy.deepcopy(cfg))
    return result


class _BmmMmaTuner(LibTuner.get("default")):
    """Keep the MMA tile floor without changing process-wide AABS settings.

    This compiler can shrink masked M/N below its supported 16-element MMA
    tile. The ordinary LibTuner cache/search remains in charge; only this
    kernel's fixed-config measurement bypasses that adjustment.
    """

    def _make_config_table_name(self):
        # The previous AABS policy could exclude the intended transposed tile.
        # Re-select winners while keeping per-config timings: those are keyed
        # by the actual (possibly AABS-adjusted) configuration that was timed.
        return f"{super()._make_config_table_name()}_logical_transpose_v1"

    def _bench(self, *args, config, **meta):
        # The source-level AABS analysis does not follow M/N's constexpr swap.
        # Pruning already bounded BM/BN against the correct logical axes.
        if (
            not config.kwargs["TRANSPOSE"]
            and min(self.nargs[name] for name in ("M", "N", "K")) >= 16
        ):
            return super()._bench(*args, config=config, **meta)
        current = {**meta, **config.all_kwargs()}

        def launch():
            if config.pre_hook:
                config.pre_hook({**self.nargs, **current})
            self.fn.run(*args, **current)

        try:
            return self.do_bench(launch, quantiles=(0.5, 0.2, 0.8))
        except triton.runtime.errors.OutOfResources:
            return [float("inf")] * 3


def _tune(kernel, name, key, prune, *, policy="default"):
    return libentry()(
        libtuner(
            configs=runtime.ops_get_configs(name, yaml_path=EXPAND_CONFIG_FILENAME),
            key=key,
            prune_configs_by={"early_config_prune": prune},
            policy=policy,
            rep=20,
            flagtune_op_name="bmm",
            flagtune_expand_op_name=name,
            flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        )(kernel)
    )


_KEY = [
    "BATCH",
    "M",
    "N",
    "K",
    "SAB",
    "SBB",
    "SCB",
    "SAM",
    "SAK",
    "SBK",
    "SBN",
    "SCM",
    "SCN",
    "SPLIT_K",
]
_bmm_tuned = _tune(_bmm_kernel, "bmm_gemm", _KEY, _prune_bmm, policy=_BmmMmaTuner)


def _prune_wide(configs, named_args, **kwargs):
    return _prune_bmm(configs, named_args, wide=True, **kwargs)


_wide_tuned = _tune(_bmm_kernel, "bmm_wide", _KEY, _prune_wide, policy=_BmmMmaTuner)


@triton.jit
def _bmm_small_kernel(
    A,
    B,
    C,
    BATCH: tl.constexpr,
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
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mi, ni = ids // N, ids % N
    ki = tl.arange(0, BK)
    acc = tl.zeros((BLOCK, BK), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = ki + step * BK
        a = tl.load(
            A
            + batch * SAB
            + mi[:, None].to(tl.int64) * SAM
            + kk[None, :].to(tl.int64) * SAK,
            (mi[:, None] < M) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        b = tl.load(
            B
            + batch * SBB
            + ni[:, None].to(tl.int64) * SBN
            + kk[None, :].to(tl.int64) * SBK,
            (ids[:, None] < M * N) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        acc = tl.fma(a, b, acc)
    tl.store(
        C + batch * SCB + mi.to(tl.int64) * SCM + ni.to(tl.int64) * SCN,
        tl.sum(acc, 1),
        ids < M * N,
    )


@dataclass(frozen=True)
class _Features:
    """Validated metadata used to choose a plan without retaining tensors."""

    batch: int
    m: int
    n: int
    k: int
    a_strides: tuple
    b_strides: tuple
    c_strides: tuple
    dtype: torch.dtype
    out_dtype: torch.dtype
    a_contiguous: bool
    b_contiguous: bool
    c_contiguous: bool
    allow_tf32: bool

    @classmethod
    def from_tensors(cls, a, b, c):
        return cls(
            batch=a.shape[0],
            m=a.shape[1],
            n=b.shape[2],
            k=a.shape[2],
            a_strides=tuple(a.stride()),
            b_strides=tuple(b.stride()),
            c_strides=tuple(c.stride()),
            dtype=a.dtype,
            out_dtype=c.dtype,
            a_contiguous=a.is_contiguous(),
            b_contiguous=b.is_contiguous(),
            c_contiguous=c.is_contiguous(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        )

    @property
    def half(self):
        return self.dtype in (torch.float16, torch.bfloat16)


class _BmmCall(NamedTuple):
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    f: _Features


class _BmmPlan(NamedTuple):
    launch: Callable
    split_k: int = 1
    transpose: bool = False
    column: bool = False
    chunk_n: int = 0
    block_k: int = 0


@triton.jit
def _bmm_vector_kernel(
    A,
    X,
    Y,
    R: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SAR: tl.constexpr,
    SAK: tl.constexpr,
    SXB: tl.constexpr,
    SXK: tl.constexpr,
    SYB: tl.constexpr,
    SYR: tl.constexpr,
    SPLIT_K: tl.constexpr,
    COLUMN: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    part = tl.program_id(1)
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    offsets = tl.arange(0, BK).to(tl.int64)
    if COLUMN:
        acc = tl.zeros((BK, BR), tl.float32)
    else:
        acc = tl.zeros((BR, BK), tl.float32)
    for start in range(part * BK, K, SPLIT_K * BK):
        ks = start + offsets
        x = tl.load(X + batch * SXB + ks * SXK, ks < K, 0).to(tl.float32)
        if COLUMN:
            a = tl.load(
                A + batch * SAB + ks[:, None] * SAK + r[None, :] * SAR,
                (ks[:, None] < K) & (r[None, :] < R),
                0,
            ).to(tl.float32)
            acc = tl.fma(a, x[:, None], acc)
        else:
            a = tl.load(
                A + batch * SAB + r[:, None] * SAR + ks[None, :] * SAK,
                (r[:, None] < R) & (ks[None, :] < K),
                0,
            ).to(tl.float32)
            acc = tl.fma(a, x[None, :], acc)
    value = tl.sum(acc, 0 if COLUMN else 1)
    tl.store(Y + batch * SYB + part.to(tl.int64) * R + r * SYR, value, r < R)


@triton.jit
def _bmm_vector_reduce(
    P,
    Y,
    R: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SYB: tl.constexpr,
    SYR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT_K):
        value += tl.load(P + batch * SPLIT_K * R + s * R + r, r < R, 0)
    tl.store(Y + batch * SYB + r.to(tl.int64) * SYR, value, r < R)


@triton.jit
def _bmm_pack_rhs(B, P, K: tl.constexpr, N: tl.constexpr):
    """Pack a dense batched RHS into K-contiguous panels for the MMA kernel."""
    batch = tl.program_id(1).to(tl.int64)
    ki = tl.program_id(0) // tl.cdiv(N, 128) * 64 + tl.arange(0, 64)
    ni = tl.program_id(0) % tl.cdiv(N, 128) * 128 + tl.arange(0, 128)
    mask = (ki[:, None] < K) & (ni[None, :] < N)
    value = tl.load(
        B + batch * K * N + ki[:, None].to(tl.int64) * N + ni[None, :],
        mask,
        other=0,
    )
    tl.store(
        P + batch * K * N + ni[None, :].to(tl.int64) * K + ki[:, None],
        value,
        mask,
    )


@triton.jit
def _bmm_pack_fp32(
    X,
    H,
    L,
    R: tl.constexpr,
    C: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SC: tl.constexpr,
    COLUMN: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    r = (tl.program_id(0) // tl.cdiv(C, 32) * 32 + tl.arange(0, 32)).to(tl.int64)
    c = (tl.program_id(0) % tl.cdiv(C, 32) * 32 + tl.arange(0, 32)).to(tl.int64)
    mask = (r[:, None] < R) & (c[None, :] < C)
    x = tl.load(X + batch * SB + r[:, None] * SR + c[None, :] * SC, mask, 0)
    hi = x.to(tl.bfloat16)
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
    offsets = batch * R * C
    if COLUMN:
        offsets += r[:, None] + c[None, :] * R
    else:
        offsets += r[:, None] * C + c[None, :]
    tl.store(H + offsets, hi, mask)
    tl.store(L + offsets, lo, mask)


@triton.jit
def _bmm_combine_fp32(P, Q, R, Y, SIZE: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    p = tl.load(P + x, x < SIZE, 0)
    q = tl.load(Q + x, x < SIZE, 0)
    r = tl.load(R + x, x < SIZE, 0)
    # Cross terms of an infinite high component can contain inf*0. Preserve
    # the main product's nonfinite result instead of contaminating it.
    value = tl.where(tl.abs(r) < float("inf"), (p + q) + r, r)
    tl.store(Y + x, value, x < SIZE)


@triton.jit
def _bmm_combine_chunked_fp32(
    P,
    Q,
    R,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SCB: tl.constexpr,
    SCM: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    i = tl.program_id(0).to(tl.int64) * 1024 + tl.arange(0, 1024)
    offset = batch * M * N + i
    p = tl.load(P + offset, i < M * N, 0)
    q = tl.load(Q + offset, i < M * N, 0)
    r = tl.load(R + offset, i < M * N, 0)
    value = tl.where(tl.abs(r) < float("inf"), (p + q) + r, r)
    tl.store(Y + batch * SCB + (i // N) * SCM + i % N, value, i < M * N)


def _prune_vector(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    return [
        copy.deepcopy(cfg)
        for cfg in configs
        if cfg.kwargs["BR"] <= max(16, triton.next_power_of_2(args["R"]))
        and cfg.kwargs["BK"] <= max(128, triton.next_power_of_2(args["K"]))
        # Bound private memory for the batched FP32 reduction on C550.
        and cfg.kwargs["BR"] * cfg.kwargs["BK"] <= 32768
        and cfg.kwargs["BR"] * cfg.kwargs["BK"] <= 8192 * cfg.num_warps
    ]


_vector_tuned = _tune(
    _bmm_vector_kernel,
    "bmm_vector",
    [
        "R",
        "K",
        "BATCH",
        "SAB",
        "SAR",
        "SAK",
        "SXB",
        "SXK",
        "SYB",
        "SYR",
        "SPLIT_K",
        "COLUMN",
    ],
    _prune_vector,
)


def _dispatch_bmm(f: _Features, info: MetaXDeviceInfo) -> _BmmPlan:
    if not f.batch or not f.m or not f.n:
        return _BmmPlan(_launch_empty)
    if not f.k:
        return _BmmPlan(_launch_zero)
    if (
        f.dtype == torch.float32
        and f.allow_tf32
        and min(f.m, f.n) >= 1024
        and f.k >= 4096
        and f.k % 32 == 0
        and f.a_strides[2] == 1
        and 1 in f.b_strides[1:]
        and f.c_contiguous
    ):
        return _BmmPlan(_launch_long_k_tf32)
    if (
        f.batch == 1
        and f.half
        and f.out_dtype == f.dtype
        and f.m >= 8192
        and 1536 <= f.n <= 4096
        and f.n % 192 == 0
        and f.k >= 4096
        and f.k % 32 == 0
        and f.a_contiguous
        and f.b_strides[1:] == (1, f.k)
        and f.c_contiguous
    ):
        return _BmmPlan(_launch_dual_n)
    # Packing and partials have a fixed per-call budget, independent of VRAM.
    workspace = 4 * f.batch * (f.m * f.k + f.k * f.n + 3 * f.m * f.n)
    if (
        f.dtype == torch.float32
        and f.allow_tf32
        and min(f.m, f.n) >= 1024
        and f.k >= 512
        and f.c_contiguous
    ):
        if workspace <= 8 * 1024**3:
            return _BmmPlan(_launch_compensated)
        available = 4 * 1024**3 - 4 * f.batch * f.m * f.k
        max_n = available // (4 * f.batch * (f.k + 3 * f.m))
        if max_n >= 1024:
            chunk_n = 1 << (max_n.bit_length() - 1)
            return _BmmPlan(_launch_compensated_chunked, chunk_n=chunk_n)
    if (
        f.batch == 1
        and f.dtype == f.out_dtype
        and 1 in f.a_strides[1:]
        and 1 in f.b_strides[1:]
        and f.c_contiguous
    ):
        return _BmmPlan(_launch_mm)
    if f.m * f.n <= 32:
        return _BmmPlan(_launch_small, block_k=triton.next_power_of_2(min(f.k, 256)))
    if f.m == 1 or f.n == 1:
        rows = max(f.m, f.n)
        column = (
            f.a_strides[1] < f.a_strides[2]
            if f.n == 1
            else f.b_strides[2] < f.b_strides[1]
        )
        split = 1
        if column and f.k >= 1024:
            programs = f.batch * triton.cdiv(rows, 128)
            wanted = triton.cdiv(4 * info.sm_count, programs)
            split = min(32, 1 << (wanted - 1).bit_length())
        return _BmmPlan(_launch_vector, split, f.m == 1, column)
    if f.half and 2 <= f.m <= 32 and f.n >= 512 and f.k >= 512:
        return _BmmPlan(_launch_wide)
    if (
        f.half
        and f.out_dtype == f.dtype
        and f.m >= 4096
        and f.n >= 4096
        and 2048 <= f.k <= 4096
        and f.m % 256 == 0
        and f.n % 256 == 0
        and f.k % 32 == 0
        and f.a_contiguous
        and f.b_contiguous
        and f.c_contiguous
        and f.batch * f.k * f.n * 2 <= 8 * 1024**3
    ):
        return _BmmPlan(_launch_packed_gemm)
    return _BmmPlan(_launch_gemm)


def _launch_empty(call, plan):
    return


def _launch_zero(call, plan):
    call.c.zero_()


def _launch_mm(call, plan):
    mm(call.a[0], call.b[0], out=call.c[0])


def _launch_long_k_tf32(call, plan):
    _bmm_long_k_tf32[
        (call.f.batch * triton.cdiv(call.f.m, 128) * triton.cdiv(call.f.n, 128),)
    ](
        call.a,
        call.b,
        call.c,
        call.f.m,
        call.f.n,
        call.f.k,
        call.a.stride(0),
        call.b.stride(0),
        call.a.stride(1),
        call.b.stride(1),
        call.b.stride(2),
        num_warps=8,
        num_stages=2,
        pipeline="basic",
    )


def _launch_dual_n(call, plan):
    _bmm_dual_n[(triton.cdiv(call.f.m, 256) * triton.cdiv(call.f.n, 192),)](
        call.a,
        call.b,
        call.c,
        call.f.m,
        call.f.n,
        call.f.k,
        num_warps=8,
        num_stages=2,
        pipeline="basic",
    )


def _launch_compensated(call, plan):
    b, m, n, k = call.f.batch, call.f.m, call.f.n, call.f.k
    ah = torch.empty((b, m, k), device=call.a.device, dtype=torch.bfloat16)
    al = torch.empty_like(ah)
    bh = torch.empty((b, n, k), device=call.a.device, dtype=torch.bfloat16).transpose(
        1, 2
    )
    bl = torch.empty_like(bh)
    parts = [torch.empty_like(call.c) for _ in range(3)]
    _bmm_pack_fp32[(triton.cdiv(m, 32) * triton.cdiv(k, 32), b)](
        call.a, ah, al, m, k, *call.a.stride(), False, num_warps=4
    )
    _bmm_pack_fp32[(triton.cdiv(k, 32) * triton.cdiv(n, 32), b)](
        call.b, bh, bl, k, n, *call.b.stride(), True, num_warps=4
    )
    _launch_fp32_components(ah, al, bh, bl, parts, b, m, n, k)
    _bmm_combine_fp32[(triton.cdiv(call.c.numel(), 1024),)](
        *parts, call.c, call.c.numel(), num_warps=4
    )


def _launch_fp32_components(ah, al, bh, bl, parts, b, m, n, k):
    for x, y, out in ((ah, bl, parts[0]), (al, bh, parts[1]), (ah, bh, parts[2])):
        _bmm_kernel[(b * triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
            x,
            y,
            out,
            b,
            x.stride(0),
            y.stride(0),
            out.stride(0),
            m,
            n,
            k,
            *x.stride()[1:],
            *y.stride()[1:],
            *out.stride()[1:],
            BM=256,
            BN=256,
            BK=32,
            GROUP_M=1,
            num_warps=8,
            num_stages=2,
            pipeline="basic",
            scenario="reduceSmemUsage",
        )


def _launch_compensated_chunked(call, plan):
    b, m, n, k = call.f.batch, call.f.m, call.f.n, call.f.k
    ah = torch.empty((b, m, k), device=call.a.device, dtype=torch.bfloat16)
    al = torch.empty_like(ah)
    _bmm_pack_fp32[(triton.cdiv(m, 32) * triton.cdiv(k, 32), b)](
        call.a, ah, al, m, k, *call.a.stride(), False, num_warps=4
    )
    for start in range(0, n, plan.chunk_n):
        width = min(plan.chunk_n, n - start)
        bv = call.b[:, :, start : start + width]
        cv = call.c[:, :, start : start + width]
        bh = torch.empty(
            (b, width, k), device=call.a.device, dtype=torch.bfloat16
        ).transpose(1, 2)
        bl = torch.empty_like(bh)
        parts = [
            torch.empty((b, m, width), device=call.a.device, dtype=torch.float32)
            for _ in range(3)
        ]
        _bmm_pack_fp32[(triton.cdiv(k, 32) * triton.cdiv(width, 32), b)](
            bv, bh, bl, k, width, *bv.stride(), True, num_warps=4
        )
        _launch_fp32_components(ah, al, bh, bl, parts, b, m, width, k)
        _bmm_combine_chunked_fp32[(triton.cdiv(m * width, 1024), b)](
            *parts, cv, m, width, cv.stride(0), cv.stride(1), num_warps=4
        )
        # Release the previous chunk before allocating the next one, also
        # allowing CUDA Graph's private allocator to reuse the same storage.
        del bh, bl, parts


def _launch_small(call, plan):
    _bmm_small_kernel[(triton.cdiv(call.f.m * call.f.n, 8), call.f.batch)](
        call.a,
        call.b,
        call.c,
        call.f.batch,
        call.f.m,
        call.f.n,
        call.f.k,
        *call.a.stride(),
        *call.b.stride(),
        *call.c.stride(),
        BLOCK=8,
        BK=plan.block_k,
        num_warps=4,
    )


def _launch_vector(call, plan):
    if not plan.transpose:
        matrix, vector = call.a, call.b
        rows = call.f.m
        sar, sak = call.a.stride()[1:]
        sxk = call.b.stride(1)
        syr = call.c.stride(1)
    else:
        matrix, vector = call.b, call.a
        rows = call.f.n
        sak, sar = call.b.stride()[1:]
        sxk = call.a.stride(2)
        syr = call.c.stride(2)
    split = plan.split_k
    target = call.c
    syb = call.c.stride(0)
    target_syr = syr
    if split > 1:
        target = torch.empty(
            (call.f.batch, split, rows), device=call.a.device, dtype=torch.float32
        )
        syb = split * rows
        target_syr = 1
    _vector_tuned[lambda cfg: (triton.cdiv(rows, cfg["BR"]), split, call.f.batch)](
        matrix,
        vector,
        target,
        rows,
        call.f.k,
        call.f.batch,
        matrix.stride(0),
        sar,
        sak,
        vector.stride(0),
        sxk,
        syb,
        target_syr,
        SPLIT_K=split,
        COLUMN=plan.column,
    )
    if split > 1:
        _bmm_vector_reduce[(triton.cdiv(rows, 512), call.f.batch)](
            target, call.c, rows, split, call.c.stride(0), syr, BLOCK=512, num_warps=4
        )


def _launch_packed_gemm(call, plan):
    packed = torch.empty(
        (call.f.batch, call.f.n, call.f.k), device=call.b.device, dtype=call.b.dtype
    ).transpose(1, 2)
    _bmm_pack_rhs[
        (triton.cdiv(call.f.k, 64) * triton.cdiv(call.f.n, 128), call.f.batch)
    ](call.b, packed, call.f.k, call.f.n, num_warps=4)
    _launch_tiled(call._replace(b=packed), plan, _bmm_tuned)


def _launch_gemm(call, plan):
    _launch_tiled(call, plan, _bmm_tuned)


def _launch_wide(call, plan):
    _launch_tiled(call, plan, _wide_tuned)


def _launch_tiled(call, plan, tuned):
    def grid(cfg):
        m, n = (call.f.n, call.f.m) if cfg["TRANSPOSE"] else (call.f.m, call.f.n)
        return (call.f.batch * triton.cdiv(m, cfg["BM"]) * triton.cdiv(n, cfg["BN"]),)

    tuned[grid](
        call.a,
        call.b,
        call.c,
        call.f.batch,
        call.a.stride(0),
        call.b.stride(0),
        call.c.stride(0),
        call.f.m,
        call.f.n,
        call.f.k,
        *call.a.stride()[1:],
        *call.b.stride()[1:],
        *call.c.stride()[1:],
        SPLIT_K=1,
    )


def _validate_bmm(a, b, out):
    if a.ndim != 3 or b.ndim != 3 or a.shape[2] != b.shape[1]:
        raise RuntimeError("incompatible matrix dimensions")
    if a.shape[0] != b.shape[0]:
        raise RuntimeError("incompatible batch dimensions")
    if (
        a.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or b.dtype != a.dtype
    ):
        raise RuntimeError("matrices must have the same floating dtype")
    if a.device != b.device or a.device.type != "cuda":
        raise RuntimeError("matrices must be on the same MetaX device")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise RuntimeError("matrices must have strided layouts")
    shape = (a.shape[0], a.shape[1], b.shape[2])
    if out is not None and (
        out.shape != shape
        or out.device != a.device
        or out.dtype not in (a.dtype, torch.float32)
        or out.layout != torch.strided
    ):
        raise RuntimeError("incompatible output shape, dtype, device or layout")
    if out is None:
        out = torch.empty(shape, device=a.device, dtype=a.dtype)
    return _BmmCall(a, b, out, _Features.from_tensors(a, b, out))


def bmm(a, b, *, out=None):
    """Validate once, select a metadata-only plan, and bind this call's tensors."""
    logger.debug("GEMS METAX BMM")
    call = _validate_bmm(a, b, out)
    info = device_info.for_device(a.device)
    with device_info.use_device(a.device):
        plan = _dispatch_bmm(call.f, info)
        plan.launch(call, plan)
        return call.c


def bmm_out(a, b, out):
    return bmm(a, b, out=out)
