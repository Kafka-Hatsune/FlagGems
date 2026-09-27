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
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTuner

from ..device_info import device_info

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "router_gemm_metax_expand.yaml")
)


@triton.jit
def _router_simt(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SXM: tl.constexpr,
    SXK: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Coalesced K reads use the C550's 64-thread warp. One token is reused
    # across BN expert dot products without materializing input copies.
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    pid = tl.program_id(0)
    split = pid // (nm * nn) % SPLIT_K
    iterations = tl.cdiv(K, BK * SPLIT_K)
    ki = tl.arange(0, BK).to(tl.int64) + split * iterations * BK
    mi = (pid // nn % nm).to(tl.int64)
    ni = (pid % nn * BN + tl.arange(0, BN)).to(tl.int64)
    xp = X + mi * SXM
    wp = W + ni[:, None] * SWN
    acc = tl.zeros((BN, BK), tl.float32)
    for i in range(iterations):
        ks = ki + i * BK
        x = tl.load(xp + ks * SXK, ks < K, 0).to(tl.float32)
        w = tl.load(
            wp + ks[None, :] * SWK, (ni[:, None] < N) & (ks[None, :] < K), 0
        ).to(tl.float32)
        acc = tl.fma(x[None, :], w, acc)
    result = tl.sum(acc, 1)
    tl.store(Y + split.to(tl.int64) * M * N + mi * N + ni, result, ni < N)


@triton.jit
def _router_mma(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SXM: tl.constexpr,
    SXK: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tl.program_id(0) // (nm * nn) % SPLIT_K
    pid = tl.program_id(0) % (nm * nn)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    mi = (first_m + local % group_m) * BM + tl.arange(0, BM)
    ni = local // group_m * BN + tl.arange(0, BN)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    ki = tl.arange(0, BK) + split * iterations * BK
    mi = tl.max_contiguous(tl.multiple_of(mi, BM), BM)
    ni = tl.max_contiguous(tl.multiple_of(ni, BN), BN)
    ki = tl.max_contiguous(tl.multiple_of(ki, BK), BK)
    xp = X + mi[:, None].to(tl.int64) * SXM + ki[None, :].to(tl.int64) * SXK
    wp = W + ni[None, :].to(tl.int64) * SWN + ki[:, None].to(tl.int64) * SWK
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(iterations):
        if K % (BK * SPLIT_K) == 0:
            if M % BM == 0:
                x = tl.load(xp)
            else:
                x = tl.load(xp, mi[:, None] < M, 0)
            if N % BN == 0:
                w = tl.load(wp)
            else:
                w = tl.load(wp, ni[None, :] < N, 0)
        else:
            x = tl.load(xp, (mi[:, None] < M) & (ki[None, :] + i * BK < K), 0)
            w = tl.load(wp, (ni[None, :] < N) & (ki[:, None] + i * BK < K), 0)
        acc = tl.dot(x, w, acc)
        xp += BK * SXK
        wp += BK * SWK
    tl.store(
        Y + split.to(tl.int64) * M * N + mi[:, None].to(tl.int64) * N + ni[None, :],
        acc,
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@triton.jit
def _router_kernel(
    X,
    W,
    Y,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SXM: tl.constexpr,
    SXK: tl.constexpr,
    SWN: tl.constexpr,
    SWK: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    ALLOW_SIMT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SIMT: tl.constexpr,
):
    if SPLIT_K > 1:
        Y = P
    if SIMT:
        _router_simt(X, W, Y, M, N, K, SXM, SXK, SWN, SWK, BM, BN, BK, SPLIT_K)
    else:
        _router_mma(X, W, Y, M, N, K, SXM, SXK, SWN, SWK, BM, BN, BK, SPLIT_K, GROUP_M)


@libentry()
@triton.jit
def _router_finish(
    P,
    Y,
    SIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    if not ZERO:
        for split in tl.static_range(SPLIT_K):
            acc += tl.load(P + split * SIZE + i, i < SIZE, 0)
    tl.store(Y + i, acc, i < SIZE)


# Measured compiler allocation for this kernel, not a device capacity fallback.
_NT128_SHARED_BYTES = 64 * 1024


class _RouterTuner(LibTuner.get("default")):
    def get_key(self, args):
        return super().get_key(args) + (
            args["X"].data_ptr() % 16,
            args["W"].data_ptr() % 16,
        )

    def get_benchmark_key(self, args):
        return super().get_benchmark_key(args) + (
            args["X"].data_ptr() % 16,
            args["W"].data_ptr() % 16,
        )

    def _bench(self, *args, config, **meta):
        options = {**meta, **config.all_kwargs()}
        values = {**dict(zip(self.arg_names, args)), **options}

        def launch():
            self.fn.run(*args, **options)
            if values["SPLIT_K"] > 1:
                size = values["M"] * values["N"]
                _router_finish[(triton.cdiv(size, 256),)](
                    values["P"],
                    values["Y"],
                    size,
                    values["SPLIT_K"],
                    False,
                    256,
                    num_warps=4,
                )

        try:
            return self.do_bench(launch, quantiles=(0.5, 0.2, 0.8))
        except triton.runtime.errors.OutOfResources:
            return [float("inf")] * 3


def _prune_router(configs, named_args, **kwargs):
    a = {**named_args, **kwargs}
    m, n, k = a["M"], a["N"], a["K"]
    info = device_info.for_device(a["X"].device)
    sm, shared_bytes = info.sm_count, info.shared_bytes
    result = []
    for cfg in configs:
        q = cfg.kwargs
        bm, bn, bk = q["BM"], q["BN"], q["BK"]
        nt_128 = (
            not q["SIMT"]
            and bm == bn == bk == 128
            and cfg.num_warps == 4
            and q["pipeline"] == "cpasync"
            and a["SXK"] == a["SWK"] == 1
        )
        if q["SIMT"]:
            if (
                not a["ALLOW_SIMT"]
                or bm != 1
                or bm > triton.next_power_of_2(m)
                or bn > triton.next_power_of_2(n)
            ):
                continue
            if bm * bn * bk > 8192:
                continue
        elif bm > max(16, triton.next_power_of_2(m)) or bn > max(
            16, triton.next_power_of_2(n)
        ):
            continue
        elif (_NT128_SHARED_BYTES if nt_128 else (bm + bn) * bk * 4) > shared_bytes:
            continue
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        for split in (1, 2, 4, 8, 16, 32):
            if split > a["MAX_SPLIT"]:
                break
            if nt_128 and k % (bk * split):
                continue
            if split > 1 and (k < split * 256 or tiles * split > 4 * sm):
                continue
            candidate = copy.deepcopy(cfg)
            candidate.kwargs["SPLIT_K"] = split
            result.append(candidate)
    return result


_router_tuned = libentry()(
    libtuner(
        configs=runtime.ops_get_configs(
            "router_gemm", yaml_path=EXPAND_CONFIG_FILENAME
        ),
        key=["M", "N", "K", "SXM", "SXK", "SWN", "SWK", "MAX_SPLIT", "ALLOW_SIMT"],
        policy=_RouterTuner,
        prune_configs_by={"early_config_prune": _prune_router},
        flagtune_op_name="router_gemm",
        flagtune_expand_op_name="router_gemm",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
        rep=30,
    )(_router_kernel)
)


@dataclass(frozen=True)
class _Features:
    """Validated BF16 router metadata, independent of the current tensors."""

    m: int
    n: int
    k: int
    x_strides: tuple
    w_strides: tuple
    alignment: tuple

    @classmethod
    def from_tensors(cls, x, w):
        return cls(
            x.shape[0],
            w.shape[0],
            x.shape[1],
            tuple(x.stride()),
            tuple(w.stride()),
            (x.data_ptr() % 16, w.data_ptr() % 16),
        )


class _RouterCall(NamedTuple):
    x: torch.Tensor
    w: torch.Tensor
    y: torch.Tensor
    f: _Features


class _RouterPlan(NamedTuple):
    launch: Callable
    max_split: int = 1
    allow_simt: bool = False


def _validate_router(input, weight):
    if input.ndim != 2 or weight.ndim != 2 or input.shape[1] != weight.shape[1]:
        raise RuntimeError(
            "router input and weight must have compatible matrix dimensions"
        )
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError("router input and weight must be bfloat16")
    if input.device != weight.device or input.device.type == "cpu":
        raise RuntimeError("router input and weight must be on the same accelerator")
    m, n = input.shape[0], weight.shape[0]
    y = torch.empty((m, n), dtype=torch.float32, device=input.device)
    return _RouterCall(input, weight, y, _Features.from_tensors(input, weight))


@lru_cache(maxsize=4096)
def _dispatch_router(f: _Features) -> _RouterPlan:
    if not f.m or not f.n:
        return _RouterPlan(_launch_router_empty)
    if not f.k:
        return _RouterPlan(_launch_router_zero)
    limit = min(32, max(1, f.k // 256), max(1, 32 * 1024**2 // max(4 * f.m * f.n, 1)))
    return _RouterPlan(
        _launch_router, 1 << (limit.bit_length() - 1), f.m <= 32 or f.n <= 8
    )


def _launch_router_empty(c, p):
    return


def _launch_router_zero(c, p):
    _router_finish[(triton.cdiv(c.f.m * c.f.n, 256),)](
        c.y, c.y, c.f.m * c.f.n, 1, True, 256, num_warps=4
    )


def _launch_router(c, p):
    partial = torch.empty(
        (p.max_split * c.f.m * c.f.n if p.max_split > 1 else 0,),
        device=c.x.device,
        dtype=torch.float32,
    )
    _, meta = _router_tuned[
        lambda q: (
            triton.cdiv(c.f.m, q["BM"]) * triton.cdiv(c.f.n, q["BN"]) * q["SPLIT_K"],
        )
    ](
        c.x,
        c.w,
        c.y,
        partial,
        c.f.m,
        c.f.n,
        c.f.k,
        *c.f.x_strides,
        *c.f.w_strides,
        p.max_split,
        p.allow_simt
    )
    if meta["SPLIT_K"] > 1:
        _router_finish[(triton.cdiv(c.f.m * c.f.n, 256),)](
            partial,
            c.y,
            c.f.m * c.f.n,
            meta["SPLIT_K"],
            False,
            256,
            num_warps=4,
        )


def router_gemm_bf16_fp32(input, weight):
    logger.debug("GEMS METAX ROUTER GEMM")
    c = _validate_router(input, weight)
    with device_info.use_device(input.device):
        p = _dispatch_router(c.f)
        p.launch(c, p)
    return c.y


def router_gemm(input, weight):
    return router_gemm_bf16_fp32(input, weight)
