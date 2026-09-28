# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mul import mul
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils.libentry import LibTuner

from ..device_info import device_info

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "baddbmm_metax_expand.yaml")
)
# Measured native-tile allocation; device_info supplies the available capacity.
_NATIVE128_SHARED_BYTES = 64 * 1024


@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_gemm_kernel(
    A,
    B,
    C,
    P,
    Bias,
    alpha,
    beta,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SCB: tl.constexpr,
    SIB: tl.constexpr,
    SIM: tl.constexpr,
    SIN: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ONE: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
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
    FLAT_EPILOGUE: tl.constexpr = False,
):
    """One output tile per CTA, optionally with a deterministic K partition."""
    if SPLIT_K > 1:
        C = P
        SCM, SCN = N, 1
        SCB = SPLIT_K * M * N
    batch = tl.program_id(1).to(tl.int64)
    A += batch * SAB
    B += batch * SBB
    C += batch * SCB
    Bias += batch * SIB
    if TRANSPOSE:
        # Compute C^T = B^T A^T. Swapping the dot operands changes which
        # operand feeds which MMA port without materializing a transpose.
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
        SIM, SIN = SIN, SIM
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tl.program_id(0) // (nm * nn) % SPLIT_K
    pid = tl.program_id(0) % (nm * nn)
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
    if FLAT_EPILOGUE:
        # Materialize the accumulator's output order before the bias arithmetic.
        i = tl.arange(0, BM * BN)
        rows = (pm * BM + i // BN).to(tl.int64)
        cols = (pn * BN + i % BN).to(tl.int64)
        value = tl.reshape(acc, (BM * BN,))
        mask = (rows < M) & (cols < N)
        if SPLIT_K == 1:
            if not ALPHA_ONE:
                value *= alpha
            if not BETA_ZERO:
                bias = tl.load(Bias + rows * SIM + cols * SIN, mask, 0).to(tl.float32)
                value += bias if BETA_ONE else beta * bias
        tl.store(C + split.to(tl.int64) * M * N + rows * SCM + cols * SCN, value, mask)
    else:
        cp = (
            C
            + split.to(tl.int64) * M * N
            + mi[:, None].to(tl.int64) * SCM
            + ni[None, :].to(tl.int64) * SCN
        )
        if SPLIT_K == 1:
            if not ALPHA_ONE:
                acc *= alpha
            if not BETA_ZERO:
                if SIM == 0 and SIN == 0:
                    bias = tl.full((BM, BN), tl.load(Bias).to(tl.float32), tl.float32)
                elif SIM == 0:
                    bias = tl.broadcast_to(
                        tl.load(
                            Bias + ni[None, :].to(tl.int64) * SIN, ni[None, :] < N, 0
                        ).to(tl.float32),
                        (BM, BN),
                    )
                elif SIN == 0:
                    bias = tl.broadcast_to(
                        tl.load(
                            Bias + mi[:, None].to(tl.int64) * SIM, mi[:, None] < M, 0
                        ).to(tl.float32),
                        (BM, BN),
                    )
                else:
                    bias = tl.load(
                        Bias
                        + mi[:, None].to(tl.int64) * SIM
                        + ni[None, :].to(tl.int64) * SIN,
                        (mi[:, None] < M) & (ni[None, :] < N),
                        other=0,
                    ).to(tl.float32)
                acc += bias if BETA_ONE else beta * bias
        tl.store(cp, acc, (mi[:, None] < M) & (ni[None, :] < N))


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_finish_kernel(
    P,
    C,
    Bias,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    SCB: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    SIB: tl.constexpr,
    SIM: tl.constexpr,
    SIN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ONE: tl.constexpr,
    ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m, n = i // N, i % N
    acc = tl.zeros((BLOCK,), tl.float32)
    if not ZERO:
        for split in tl.static_range(SPLIT_K):
            acc += tl.load(P + (batch * SPLIT_K + split) * M * N + i, i < M * N, 0)
        if not ALPHA_ONE:
            acc *= alpha
    if not BETA_ZERO:
        bias = tl.load(Bias + batch * SIB + m * SIM + n * SIN, i < M * N, 0)
        bias = bias.to(tl.float32)
        acc += bias if BETA_ONE else beta * bias
    tl.store(C + batch * SCB + m * SCM + n * SCN, acc, i < M * N)


@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_vector_kernel(
    A,
    B,
    C,
    P,
    Bias,
    alpha,
    beta,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SCB: tl.constexpr,
    SIB: tl.constexpr,
    SIM: tl.constexpr,
    SIN: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ONE: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    if SPLIT_K > 1:
        C = P
        SCM, SCN = N, 1
        SCB = SPLIT_K * M * N
    batch = tl.program_id(1).to(tl.int64)
    A += batch * SAB
    B += batch * SBB
    C += batch * SCB
    Bias += batch * SIB
    if TRANSPOSE:
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
        SIM, SIN = SIN, SIM
    nn = tl.cdiv(N, BN)
    pid = tl.program_id(0)
    split = pid // (M * nn)
    m = (pid // nn % M).to(tl.int64)
    n = (pid % nn * BN + tl.arange(0, BN)).to(tl.int64)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    k = tl.arange(0, BK).to(tl.int64) + split * iterations * BK
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(iterations):
        ks = k + start * BK
        a = tl.load(A + m * SAM + ks * SAK, ks < K, 0).to(tl.float32)
        b = tl.load(
            B + n[:, None] * SBN + ks[None, :] * SBK,
            (n[:, None] < N) & (ks[None, :] < K),
            0,
        ).to(tl.float32)
        acc = tl.fma(a[None, :], b, acc)
    result = tl.sum(acc, 1)
    if SPLIT_K == 1:
        if not ALPHA_ONE:
            result *= alpha
        if not BETA_ZERO:
            bias = tl.load(Bias + m * SIM + n * SIN, n < N, 0).to(tl.float32)
            result += bias if BETA_ONE else bias * beta
    tl.store(C + split.to(tl.int64) * M * N + m * SCM + n * SCN, result, n < N)


class _BaddbmmTuner(LibTuner.get("default")):
    """Measure the MMA kernel and its selected FP32 reduction together."""

    def get_key(self, args):
        policy_key = ("confirm16x3",) if args["K"] >= 1024 else ()
        return (
            super().get_key(args)
            + policy_key
            + tuple(args[name].data_ptr() % 16 for name in ("A", "B", "C", "Bias"))
        )

    def get_benchmark_key(self, args):
        return super().get_benchmark_key(args) + tuple(
            args[name].data_ptr() % 16 for name in ("A", "B", "C", "Bias")
        )

    def _bench(self, *args, config, **meta):
        options = {**meta, **config.all_kwargs()}
        values = {**dict(zip(self.arg_names, args)), **options}

        def launch():
            self.fn.run(*args, **options)
            if values["SPLIT_K"] > 1:
                _baddbmm_finish_kernel[
                    (triton.cdiv(values["M"] * values["N"], 256), values["BATCH"])
                ](
                    values["P"],
                    values["C"],
                    values["Bias"],
                    values["alpha"],
                    values["beta"],
                    values["M"],
                    values["N"],
                    values["SCB"],
                    values["SCM"],
                    values["SCN"],
                    values["SIB"],
                    values["SIM"],
                    values["SIN"],
                    values["SPLIT_K"],
                    values["BETA_ZERO"],
                    values["ALPHA_ONE"],
                    values["BETA_ONE"],
                    False,
                    BLOCK=256,
                    num_warps=4,
                )

        try:
            return self.do_bench(launch, quantiles=(0.5, 0.2, 0.8))
        except triton.runtime.errors.OutOfResources:
            return [float("inf")] * 3

    def policy(self, bench_fn, configs, args, kwargs):
        best, timings = super().policy(bench_fn, configs, args, kwargs)
        values = {**dict(zip(self.arg_names, args)), **kwargs}
        if values["K"] < 1024:
            return best, timings
        finalists = [
            config
            for config in sorted(timings, key=timings.get)[:16]
            if timings[config][0] < float("inf")
        ]
        if len(finalists) < 2:
            return best, timings
        # Recheck long reductions in alternating full-call rounds.
        # Coarse benchmark data stays reusable; get_key versions the selection.
        samples = {config: [] for config in finalists}
        with self.use_benchmark_protocol("replay", warmup=0, rep=100):
            for iteration in range(3):
                order = finalists if iteration % 2 == 0 else reversed(finalists)
                for config in order:
                    value = self._bench(*args, config=config, **kwargs)
                    if isinstance(value, (int, float)):
                        value = [value] * 3
                    samples[config].append(value)
        for config, values in samples.items():
            timings[config] = [sorted(v[i] for v in values)[1] for i in range(3)]
        return min(finalists, key=timings.get), timings


def _prune_baddbmm_gemm(configs, named_args, **kwargs):
    a = {**named_args, **kwargs}
    m, n, k = a["M"], a["N"], a["K"]
    info = device_info.for_device(a["A"].device)
    sm, shared_bytes = info.sm_count, info.shared_bytes
    result = []
    for cfg in configs:
        q = cfg.kwargs
        bm, bn, bk = q["BM"], q["BN"], q["BK"]
        native_tile = (
            bm == bn == bk == 128
            and cfg.num_warps == 4
            and cfg.num_stages == 4
            and q["pipeline"] == "cpasync"
            and not q["scenario"]
            and not q["TRANSPOSE"]
            and not q["STATIC_K"]
        )
        dense_output = a["SCM"] == n and a["SCN"] == 1
        direct = (
            native_tile
            and a["A"].dtype in (torch.float16, torch.bfloat16)
            and a["C"].dtype == a["A"].dtype
            and m >= 1024
            and n >= 128
            and k >= 512
            and a["SAM"] == k
            and a["SAK"] == 1
            and dense_output
            and all(a[name].data_ptr() % 16 == 0 for name in ("A", "B", "C"))
            and k % 8 == n % 8 == 0
            and a["BATCH"] * triton.cdiv(m, bm) * triton.cdiv(n, bn) >= sm // 2
        )
        direct_nn = direct and a["SBK"] == n and a["SBN"] == 1
        direct_nt = direct and a["SBK"] == 1 and a["SBN"] == k
        # Flatten before bias arithmetic so the NT half store does not
        # require the faulty MMA-layout bias/store conversion.
        if q["FLAT_EPILOGUE"] and not direct_nt:
            continue
        mt, nt = (n, m) if q["TRANSPOSE"] else (m, n)
        if bm > max(32, triton.next_power_of_2(mt)) or bn > max(
            32, triton.next_power_of_2(nt)
        ):
            continue
        if min(m, n) >= 1024 and bm * bn < 4096:
            continue
        shared = (
            _NATIVE128_SHARED_BYTES
            if direct_nn or (direct_nt and q["FLAT_EPILOGUE"])
            else (bm + bn)
            * bk
            * a["A"].element_size()
            * (1 if cfg.num_stages == 1 else 2)
        )
        if shared > shared_bytes:
            continue
        if (
            bk > max(32, triton.next_power_of_2(k))
            or q["scenario"] == "reduceSmemUsage"
        ):
            continue
        if q["scenario"] == "unprefetch" and (mt % bm or nt % bn or k % bk):
            continue
        if q["STATIC_K"] and (k > 128 or min(m, n) < 64):
            continue
        if bm >= 256 and bn >= 256 and a["SBK"] == 1:
            continue
        if q["TRANSPOSE"] and not (a["SBN"] == 1 or a["SAM"] == 1):
            continue
        # Roll changes only the installed compiler's loop-unroll policy.
        if q["scenario"] == "roll" and not (64 <= min(m, n) <= 256 and k >= 1024):
            continue
        tiles = a["BATCH"] * triton.cdiv(mt, bm) * triton.cdiv(nt, bn)
        for split in (1, 2, 4, 8, 16, 32):
            if split > a["MAX_SPLIT"]:
                break
            if (q["FLAT_EPILOGUE"] or direct_nn) and split != 1:
                continue
            if split > 1 and (
                q["STATIC_K"] or k < split * 256 or tiles * split > 4 * sm
            ):
                continue
            if (
                a["A"].dtype == torch.float32
                and split > 1
                and k % (bk * split)
                and q["pipeline"].startswith("cpasync")
            ):
                continue
            candidate = copy.deepcopy(cfg)
            candidate.kwargs["SPLIT_K"] = split
            result.append(candidate)
    return result


def _prune_baddbmm_vector(configs, named_args, **kwargs):
    a = {**named_args, **kwargs}
    n = a["M"] if a["TRANSPOSE"] else a["N"]
    sk = a["SAK"] if a["TRANSPOSE"] else a["SBK"]
    sn = a["SAM"] if a["TRANSPOSE"] else a["SBN"]
    along_k = sk == 1 or sk <= sn or n < 32
    rows = a["N"] if a["TRANSPOSE"] else a["M"]
    sm = device_info.for_device(a["A"].device).sm_count
    result = []
    for c in configs:
        if c.kwargs["BN"] > max(1, triton.next_power_of_2(n)):
            continue
        if not (
            (along_k and c.kwargs["BN"] <= 8) or (not along_k and c.kwargs["BN"] >= 32)
        ):
            continue
        tiles = a["BATCH"] * rows * triton.cdiv(n, c.kwargs["BN"])
        for split in (1, 2, 4, 8, 16, 32):
            if split > a["MAX_SPLIT"]:
                break
            # Include the extra SIMT wave needed by rounded-up K partitions.
            if split > 1 and (a["K"] < split * 256 or tiles * split > 8 * sm):
                continue
            candidate = copy.deepcopy(c)
            candidate.kwargs["SPLIT_K"] = split
            result.append(candidate)
    return result


_KEY = [
    "BATCH",
    "SAB",
    "SBB",
    "SCB",
    "SIB",
    "M",
    "N",
    "K",
    "SAM",
    "SAK",
    "SBK",
    "SBN",
    "SCM",
    "SCN",
    "SIM",
    "SIN",
    "BETA_ZERO",
    "ALPHA_ONE",
    "BETA_ONE",
    "MAX_SPLIT",
]

_baddbmm_gemm = libentry()(
    libtuner(
        configs=runtime.ops_get_configs(
            "baddbmm_gemm", yaml_path=EXPAND_CONFIG_FILENAME
        ),
        key=_KEY,
        policy=_BaddbmmTuner,
        prune_configs_by={"early_config_prune": _prune_baddbmm_gemm},
        flagtune_op_name="baddbmm",
        flagtune_expand_op_name="baddbmm_gemm",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        rep=30,
    )(_baddbmm_gemm_kernel)
)

_baddbmm_vector = libentry()(
    libtuner(
        configs=runtime.ops_get_configs(
            "baddbmm_vector", yaml_path=EXPAND_CONFIG_FILENAME
        ),
        key=_KEY + ["TRANSPOSE"],
        policy=_BaddbmmTuner,
        prune_configs_by={"early_config_prune": _prune_baddbmm_vector},
        flagtune_op_name="baddbmm",
        flagtune_expand_op_name="baddbmm_vector",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        rep=30,
    )(_baddbmm_vector_kernel)
)


@dataclass(frozen=True)
class _Features:
    """Validated input metadata; never owns tensors or workspace."""

    batch: int
    m: int
    n: int
    k: int
    a_strides: tuple
    b_strides: tuple
    out_strides: tuple
    bias_strides: tuple
    dtype: torch.dtype
    out_dtype: torch.dtype
    alignment: tuple
    alpha_zero: bool
    alpha_one: bool
    beta_zero: bool
    beta_one: bool

    @classmethod
    def from_tensors(cls, a, b, bias, out, bias_strides, alpha, beta):
        return cls(
            batch=a.shape[0],
            m=a.shape[-2],
            n=b.shape[-1],
            k=a.shape[-1],
            a_strides=tuple(a.stride()),
            b_strides=tuple(b.stride()),
            out_strides=tuple(out.stride()),
            bias_strides=tuple(bias_strides),
            dtype=a.dtype,
            out_dtype=out.dtype,
            alignment=tuple(t.data_ptr() % 16 for t in (a, b, bias, out)),
            alpha_zero=alpha == 0,
            alpha_one=alpha == 1,
            beta_zero=beta == 0,
            beta_one=beta == 1,
        )


class _BaddbmmCall(NamedTuple):
    a: torch.Tensor
    b: torch.Tensor
    bias: torch.Tensor
    out: torch.Tensor
    alpha: float
    beta: float
    f: _Features


class _BaddbmmPlan(NamedTuple):
    launch: Callable
    max_split: int = 1
    transpose: bool = False


def _validate_baddbmm(
    bias,
    a,
    b,
    out,
    beta,
    alpha,
):
    if a.ndim != 3 or b.ndim != 3 or a.shape[-1] != b.shape[-2]:
        raise RuntimeError("incompatible matrix dimensions")
    if a.shape[0] != b.shape[0]:
        raise RuntimeError("incompatible batch dimensions")
    if (
        a.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or b.dtype != a.dtype
    ):
        raise RuntimeError("matrices must have the same floating dtype")
    if a.device != b.device or a.device.type == "cpu":
        raise RuntimeError("matrices must be on the same accelerator")
    dtype = a.dtype
    batch, m, n = a.shape[0], a.shape[-2], b.shape[-1]
    shape = (batch, m, n)
    if bias.device != a.device or bias.dtype not in (a.dtype, dtype):
        raise RuntimeError("bias dtype and device must match the input or output")
    if not broadcastable_to(bias.shape, shape):
        raise RuntimeError("incompatible bias shape")
    si = bias.broadcast_to(shape).stride()
    if out is not None and (
        out.shape != shape or out.dtype != dtype or out.device != a.device
    ):
        raise RuntimeError("incompatible output shape, dtype or device")
    if out is None:
        out = torch.empty(shape, device=a.device, dtype=dtype)
    f = _Features.from_tensors(a, b, bias, out, si, alpha, beta)
    return _BaddbmmCall(a, b, bias, out, alpha, beta, f)


@lru_cache(maxsize=4096)
def _dispatch_baddbmm(f: _Features) -> _BaddbmmPlan:
    if not f.batch or not f.m or not f.n:
        return _BaddbmmPlan(_launch_empty)
    if not f.k or f.alpha_zero:
        return _BaddbmmPlan(_launch_zero)
    if (
        max(f.m, f.n) <= 32
        and f.k <= 256
        and (1 not in f.a_strides[-2:] or 1 not in f.b_strides[-2:])
    ):
        return _BaddbmmPlan(_launch_vector)
    transpose = f.n == 1 or (f.n <= 8 and f.m > f.n)
    rows, cols = (f.n, f.m) if transpose else (f.m, f.n)
    limit = min(
        32, max(1, f.k // 256), max(1, 32 * 1024**2 // (4 * f.batch * f.m * f.n))
    )
    if rows == 1 or (rows <= 8 and cols <= 32):
        return _BaddbmmPlan(_launch_vector, 1 << (limit.bit_length() - 1), transpose)
    return _BaddbmmPlan(_launch_gemm, 1 << (limit.bit_length() - 1))


def _arguments(c, partial, max_split):
    return (
        c.a,
        c.b,
        c.out,
        partial,
        c.bias,
        c.alpha,
        c.beta,
        c.f.batch,
        c.f.a_strides[0],
        c.f.b_strides[0],
        c.f.out_strides[0],
        *c.f.bias_strides,
        c.f.beta_zero,
        c.f.alpha_one,
        c.f.beta_one,
        max_split,
        c.f.m,
        c.f.n,
        c.f.k,
        *c.f.a_strides[-2:],
        *c.f.b_strides[-2:],
        *c.f.out_strides[-2:],
    )


def _finish(c, partial, split, zero=False):
    _baddbmm_finish_kernel[(triton.cdiv(c.f.m * c.f.n, 256), c.f.batch)](
        partial,
        c.out,
        c.bias,
        c.alpha,
        c.beta,
        c.f.m,
        c.f.n,
        *c.f.out_strides,
        *c.f.bias_strides,
        split,
        c.f.beta_zero,
        c.f.alpha_one,
        c.f.beta_one,
        zero,
        BLOCK=256,
        num_warps=4,
    )


def _launch_empty(c, plan):
    return


def _launch_zero(c, plan):
    _finish(c, c.out, 1, zero=True)


def _workspace(c, plan):
    elements = c.f.batch * plan.max_split * c.f.m * c.f.n if plan.max_split > 1 else 0
    return torch.empty((elements,), dtype=torch.float32, device=c.a.device)


def _launch_vector(c, plan):
    partial = _workspace(c, plan)
    rows, cols = (c.f.n, c.f.m) if plan.transpose else (c.f.m, c.f.n)
    _, meta = _baddbmm_vector[
        lambda q: (rows * triton.cdiv(cols, q["BN"]) * q["SPLIT_K"], c.f.batch)
    ](*_arguments(c, partial, plan.max_split), TRANSPOSE=plan.transpose)
    if meta["SPLIT_K"] > 1:
        _finish(c, partial, meta["SPLIT_K"])


def _launch_gemm(c, plan):
    partial = _workspace(c, plan)

    def grid(q):
        m, n = (c.f.n, c.f.m) if q["TRANSPOSE"] else (c.f.m, c.f.n)
        return (
            triton.cdiv(m, q["BM"]) * triton.cdiv(n, q["BN"]) * q["SPLIT_K"],
            c.f.batch,
        )

    _, meta = _baddbmm_gemm[grid](*_arguments(c, partial, plan.max_split))
    if meta["SPLIT_K"] > 1:
        _finish(c, partial, meta["SPLIT_K"])


def _baddbmm_launch(bias, A, B, beta, alpha, out):
    call = _validate_baddbmm(bias, A, B, out, beta, alpha)
    with device_info.use_device(A.device):
        plan = _dispatch_baddbmm(call.f)
        plan.launch(call, plan)
    return call.out


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS METAX BADDBMM FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        return _baddbmm_launch(bias, A, B, beta, alpha, None)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS METAX BADDBMM BACKWARD")
        A, B, bias = ctx.saved_tensors

        grad_A = None
        grad_B = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_bias = compute_bias_grad(grad_output, ctx.beta, bias)
        if ctx.needs_input_grad[1]:
            grad_A = compute_A_grad(grad_output, B, ctx.alpha)
        if ctx.needs_input_grad[2]:
            grad_B = compute_B_grad(A, grad_output, ctx.alpha)

        return grad_bias, grad_A, grad_B, None, None


def compute_bias_grad(d_output, beta, bias):
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape)


def _bmm_no_tf32(lhs, rhs):
    # The metax Triton bmm kernel is inaccurate for large K and cannot
    # compile when the contraction dim is < 16, so fall back to torch.bmm.
    # TF32 must be disabled to match the fp32 precision expected by the tests
    # (the Triton kernel uses allow_tf32=False).
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.bmm(lhs, rhs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def compute_A_grad(d_output, B, alpha):
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        mul1 = _bmm_no_tf32(d_output.to(torch.float32), B_T.to(torch.float32))
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = _bmm_no_tf32(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        mul2 = _bmm_no_tf32(A_T.to(torch.float32), d_output.to(torch.float32))
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = _bmm_no_tf32(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    return _baddbmm_launch(bias, A, B, beta, alpha, out)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(bias, A, B, beta, alpha)
