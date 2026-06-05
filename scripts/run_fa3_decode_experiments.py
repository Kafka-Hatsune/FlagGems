#!/usr/bin/env python3
"""Run decode-first Hopper FA3 experimental kernels.

The script launches each variant in a separate subprocess so compiler hangs,
CUDA faults, or experimental deadlocks become CSV rows instead of stopping the
whole sweep.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

RESULT_PREFIX = "FA3_DECODE_RESULT_JSON="
SHAPE_CHOICES = (
    "smoke",
    "dense_decode",
    "paged_decode",
    "bench_decode",
    "bench_paged_decode",
)


def _load_registry_module():
    registry_path = (
        SRC_ROOT
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_nvidia"
        / "hopper"
        / "ops"
        / "fa3_ws"
        / "experimental_registry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_fa3_decode_experimental_registry", registry_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {registry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REGISTRY = _load_registry_module()


@dataclass
class LaunchContext:
    args: tuple[Any, ...]
    out: Any
    lse: Any
    finalize: Callable[[Any], Any]
    batch_size: int
    num_heads: int
    num_kv_heads: int
    max_seqlen_q: int
    max_seqlen_k: int
    total_q: int
    head_dim: int
    is_cu_seqlens_q: bool
    cu_seqlens_q: Any
    o_row_stride: int
    o_head_stride: int
    o_batch_stride: int
    scale_softmax: float
    scale_softmax_log2: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--shape", default="smoke", choices=SHAPE_CHOICES)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--dump-ir", default=None)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-shape", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _round_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _shape_key_for_variant(requested: str, exp) -> str | None:
    if requested == "smoke":
        return "paged_decode" if exp.paged else "dense_decode"
    if requested in ("paged_decode", "bench_paged_decode"):
        return requested if exp.paged else None
    if requested in ("dense_decode", "bench_decode"):
        return requested if not exp.paged else None
    return None


def _worker_env(base_env: dict[str, str], exp, dump_dir: Path | None) -> dict[str, str]:
    env = dict(base_env)
    env["TLE_WGMMA_PIPELINE_MODE"] = "user_promise"
    if exp.paged:
        env.setdefault("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_D128", "1")
        env.setdefault("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL", "1")
    if dump_dir is not None:
        env["TRITON_ALWAYS_COMPILE"] = "1"
        env["TRITON_KERNEL_DUMP"] = "1"
        env["TRITON_DUMP_DIR"] = str(dump_dir)
    return env


def _worker_command(args: argparse.Namespace, exp, shape_key: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_worker-variant",
        exp.name,
        "--_worker-shape",
        shape_key,
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--seed",
        str(args.seed),
    ]
    if args.check:
        cmd.append("--check")
    if args.perf:
        cmd.append("--perf")
    return cmd


def _parse_worker_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return None


def _find_first_ir_files(dump_dir: Path | None) -> tuple[str, str]:
    if dump_dir is None or not dump_dir.exists():
        return "", ""
    ttgir = next((str(path) for path in dump_dir.rglob("*.ttgir")), "")
    ptx = next((str(path) for path in dump_dir.rglob("*.ptx")), "")
    return ttgir, ptx


def _run_parent(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for name in REGISTRY.resolve_experiment_names(args.variant):
        exp = REGISTRY.get_experiment(name)
        shape_key = _shape_key_for_variant(args.shape, exp)
        if shape_key is None:
            rows.append(
                {
                    "variant": exp.name,
                    "family": exp.family,
                    "module": exp.module,
                    "shape": args.shape,
                    "paged": exp.paged,
                    "status": "skipped",
                    "error": "shape is not compatible with variant",
                }
            )
            continue

        dump_dir = None
        if args.dump_ir:
            dump_dir = Path(args.dump_ir) / exp.name / shape_key
            dump_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        try:
            completed = subprocess.run(
                _worker_command(args, exp, shape_key),
                cwd=REPO_ROOT,
                env=_worker_env(os.environ, exp, dump_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_sec,
                check=False,
            )
            elapsed = time.time() - start
            row = _parse_worker_result(completed.stdout)
            if row is None:
                row = {
                    "variant": exp.name,
                    "shape": shape_key,
                    "status": "error",
                    "error_type": "MissingResult",
                    "error": "worker did not emit result JSON",
                }
            if completed.returncode != 0 and row.get("status") == "ok":
                row["status"] = "error"
                row["error_type"] = "SubprocessReturnCode"
                row["error"] = str(completed.returncode)
            if completed.stderr and args.verbose:
                row["stderr_tail"] = completed.stderr[-4000:]
            row["elapsed_sec"] = f"{elapsed:.3f}"
        except subprocess.TimeoutExpired as exc:
            row = {
                "variant": exp.name,
                "shape": shape_key,
                "status": "timeout",
                "error_type": "TimeoutExpired",
                "error": str(exc),
                "elapsed_sec": f"{args.timeout_sec:.3f}",
            }

        row.setdefault("family", exp.family)
        row.setdefault("module", exp.module)
        row.setdefault("paged", exp.paged)
        ttgir, ptx = _find_first_ir_files(dump_dir)
        row["ttgir_path"] = ttgir
        row["ptx_path"] = ptx
        rows.append(row)
        print(row)

    _write_rows(rows, args.csv)
    return 0


def _write_rows(rows: list[dict[str, Any]], out: str | None) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        return
    if out:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def _make_shape(shape_key: str):
    from tests.hopper_fa3_utils import Shape

    if shape_key == "dense_decode":
        return Shape("fa3_exp_dense_decode_b4_k512_d128_gqa4", [(1, 512)] * 4, 8, 2, 128, True)
    if shape_key == "paged_decode":
        return Shape(
            "fa3_exp_paged_decode_b4_kmix_d128_gqa4",
            [(1, 128), (1, 192), (1, 256), (1, 320)],
            8,
            2,
            128,
            True,
            paged=True,
            block_size=16,
        )
    if shape_key == "bench_decode":
        return Shape("fa3_exp_dense_decode_b16_k2k_d128_gqa4", [(1, 2048)] * 16, 32, 8, 128, True)
    if shape_key == "bench_paged_decode":
        return Shape(
            "fa3_exp_paged_decode_b8_kmix_d192_gqa4",
            [(1, 1024 + 128 * i) for i in range(8)],
            32,
            8,
            192,
            True,
            paged=True,
            block_size=16,
        )
    raise ValueError(f"unknown shape key {shape_key!r}")


def _ensure_tma_allocator(torch, triton):
    def _alloc_fn(size: int, alignment: int, stream):
        del alignment, stream
        return torch.empty(size, device="cuda", dtype=torch.int8)

    if hasattr(triton, "set_allocator"):
        triton.set_allocator(_alloc_fn)


def _prepare_launch(torch, tensors, shape) -> LaunchContext:
    from flag_gems.ops.flash_api import fwd_params

    q = tensors.q
    k = tensors.k
    v = tensors.v
    q_device = q.device
    is_paged = shape.paged
    page_table = tensors.block_table
    if page_table is None:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    max_seqlen_q = int(tensors.max_seqlen_q)
    max_seqlen_k = int(tensors.max_seqlen_k)
    batch_size = tensors.cu_seqlens_q.numel() - 1
    total_q, num_heads, head_dim = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    page_table_batch_stride = page_table.stride(0)

    softmax_scale = 1.0 / math.sqrt(head_dim)
    is_causal = shape.causal
    if max_seqlen_q == 1:
        is_causal = False
    window_size_left = -1
    window_size_right = 0 if is_causal else -1
    is_local = False

    q_groups = num_heads // num_heads_k
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
    )
    if seqlenq_ngroups_swapped:
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_dim))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_dim)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
    else:
        cu_seqlens_q = tensors.cu_seqlens_q
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0

    total_q = q.size(0)
    out = torch.empty_like(q)
    o_batch_stride = out.stride(0) * max_seqlen_q if seqlenq_ngroups_swapped else 0
    lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q_device)
    p = torch.empty((), device=q_device)
    philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)
    cu_seqlens_k = (
        torch.empty((batch_size + 1,), dtype=torch.int32, device=q_device)
        if is_paged
        else tensors.cu_seqlens_k
    )

    head_size_rounded = _round_multiple(head_dim, 32) if head_dim <= 192 else 256
    seqlen_q_rounded = _round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = _round_multiple(max_seqlen_k, 32)
    scale_softmax_log2 = softmax_scale * 1.4426950408889634

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
        not is_paged,
        cu_seqlens_k,
        is_paged,
        tensors.seqused_k if is_paged else None,
        batch_size,
        k_batch_size,
        num_heads,
        num_heads_k,
        num_heads // num_heads_k,
        max_seqlen_q,
        max_seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        head_dim,
        head_size_rounded,
        False,
        0.0,
        softmax_scale,
        scale_softmax_log2,
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
        False,
        None,
        0,
        total_q,
        page_table,
        page_table_batch_stride,
        block_size,
    )
    args = tuple(getattr(params, key) for key in params.__slots__)

    def _finalize(result):
        if not seqlenq_ngroups_swapped:
            return result
        result = result.reshape(batch_size, max_seqlen_q, num_heads_k, head_dim)
        result = result.transpose(1, 2)
        return result.reshape(batch_size, num_heads_k * max_seqlen_q, head_dim)

    return LaunchContext(
        args=args,
        out=out,
        lse=lse,
        finalize=_finalize,
        batch_size=batch_size,
        num_heads=num_heads,
        num_kv_heads=num_heads_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        total_q=total_q,
        head_dim=head_dim,
        is_cu_seqlens_q=cu_seqlens_q is not None,
        cu_seqlens_q=cu_seqlens_q,
        o_row_stride=out.stride(-3),
        o_head_stride=out.stride(-2),
        o_batch_stride=o_batch_stride,
        scale_softmax=softmax_scale,
        scale_softmax_log2=scale_softmax_log2,
    )


def _decode_splits(max_seqlen_k: int, head_dim: int) -> int:
    del head_dim
    if max_seqlen_k >= 4096:
        return 8
    if max_seqlen_k >= 2048:
        return 4
    return 2


def _run_experiment(torch, triton, exp, ctx: LaunchContext):
    if exp.family == "onepass":
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.fa_hopper_decode_onepass import (
            flash_varlen_fwd_v3_tle_decode_onepass_kernel,
        )
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.utils import (
            _FA3_TLE_BUCKET_DECODE,
            _FA3_TLE_BUCKET_PAGED_DECODE,
        )

        grid = (ctx.max_seqlen_q, ctx.batch_size, ctx.num_heads)
        flash_varlen_fwd_v3_tle_decode_onepass_kernel[grid](
            *ctx.args,
            DECODE_SHAPE_BUCKET=_FA3_TLE_BUCKET_PAGED_DECODE
            if exp.paged
            else _FA3_TLE_BUCKET_DECODE,
        )
        return ctx.finalize(ctx.out)

    if exp.family == "splitkv":
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.fa_hopper_decode_splitkv import (
            flash_varlen_fwd_v3_tle_decode_splitkv_combine_kernel,
            flash_varlen_fwd_v3_tle_decode_splitkv_kernel,
        )

        num_splits = _decode_splits(ctx.max_seqlen_k, ctx.head_dim)
        partial_out = torch.empty(
            (num_splits, ctx.num_heads, ctx.total_q, ctx.head_dim),
            dtype=torch.float32,
            device=ctx.out.device,
        )
        partial_m = torch.empty(
            (num_splits, ctx.num_heads, ctx.total_q),
            dtype=torch.float32,
            device=ctx.out.device,
        )
        partial_l = torch.empty_like(partial_m)
        split_grid = (ctx.max_seqlen_q, ctx.batch_size, ctx.num_heads * num_splits)
        flash_varlen_fwd_v3_tle_decode_splitkv_kernel[split_grid](
            *ctx.args,
            partial_out,
            partial_m,
            partial_l,
            NUM_SPLITS=num_splits,
        )
        combine_block_m = 8
        combine_grid = (
            triton.cdiv(ctx.max_seqlen_q, combine_block_m),
            ctx.batch_size,
            ctx.num_heads,
        )
        flash_varlen_fwd_v3_tle_decode_splitkv_combine_kernel[combine_grid](
            ctx.out,
            ctx.lse,
            partial_out,
            partial_m,
            partial_l,
            ctx.o_row_stride,
            ctx.o_head_stride,
            ctx.o_batch_stride,
            ctx.is_cu_seqlens_q,
            ctx.cu_seqlens_q,
            ctx.batch_size,
            ctx.num_heads,
            ctx.max_seqlen_q,
            ctx.head_dim,
            ctx.total_q,
            ctx.scale_softmax,
            ctx.scale_softmax_log2,
            BLOCK_M=combine_block_m,
            BLOCK_K=1 << (ctx.head_dim - 1).bit_length(),
            NUM_SPLITS=num_splits,
        )
        return ctx.finalize(ctx.out)

    if exp.family == "seesaw":
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.fa_hopper_decode_seesaw import (
            flash_varlen_fwd_v3_tle_decode_seesaw_kernel,
        )
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.utils import (
            _FA3_TLE_BUCKET_WS_DECODE,
            _FA3_TLE_BUCKET_WS_PAGED_DECODE,
            _FA3_TLE_FAMILY_WS_SIMPLE,
        )

        num_sms = torch.cuda.get_device_properties(ctx.out.device).multi_processor_count
        grid = lambda meta: (
            min(
                num_sms,
                triton.cdiv(ctx.max_seqlen_q, meta["BLOCK_M"])
                * ctx.batch_size
                * ctx.num_heads,
            ),
        )
        flash_varlen_fwd_v3_tle_decode_seesaw_kernel[grid](
            *ctx.args,
            SHAPE_BUCKET=_FA3_TLE_BUCKET_WS_PAGED_DECODE
            if exp.paged
            else _FA3_TLE_BUCKET_WS_DECODE,
            FORCE_FAMILY_ID=_FA3_TLE_FAMILY_WS_SIMPLE,
            MIN_Q_LEN_TO_PROCESS=0,
            MAX_Q_LEN_TO_PROCESS=2**31 - 1,
            tle_wgmma_pipeline_mode="user_promise",
        )
        return ctx.finalize(ctx.out)

    raise ValueError(f"unknown experiment family {exp.family!r}")


def _run_worker(args: argparse.Namespace) -> int:
    try:
        import torch
        import triton

        import flag_gems
        from tests.hopper_fa3_utils import (
            attn_flops,
            build_reference,
            is_fa3_supported,
            make_varlen,
            max_mean_abs,
            tolerances,
        )

        exp = REGISTRY.get_experiment(args._worker_variant)
        shape = _make_shape(args._worker_shape)
        row: dict[str, Any] = {
            "variant": exp.name,
            "family": exp.family,
            "module": exp.module,
            "shape": shape.name,
            "paged": shape.paged,
            "batch": len(shape.seq_lens),
            "q_heads": shape.nh_q,
            "kv_heads": shape.nh_k,
            "head_dim": shape.head_dim,
            "max_seqlen_q": max(q for q, _ in shape.seq_lens),
            "max_seqlen_k": max(k for _, k in shape.seq_lens),
            "check_status": "",
            "max_abs": "",
            "mean_abs": "",
            "ms": "",
            "p20_ms": "",
            "p80_ms": "",
        }

        if not is_fa3_supported():
            row.update(
                {
                    "status": "skipped",
                    "error": "requires CUDA Hopper with TLE FA3 support",
                }
            )
            print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
            return 0

        _ensure_tma_allocator(torch, triton)
        tensors = make_varlen(shape, torch.float16, flag_gems.device, seed=args.seed)

        def _run_once():
            ctx = _prepare_launch(torch, tensors, shape)
            return _run_experiment(torch, triton, exp, ctx)

        out = _run_once()
        torch.cuda.synchronize()

        if args.check:
            ref, ref_kind = build_reference(tensors, shape, fa_version=3)
            atol, rtol = tolerances(torch.float16, tensors.max_seqlen_k, ref_kind)
            max_abs, mean_abs = max_mean_abs(out, ref)
            torch.testing.assert_close(
                out.float(),
                ref.float(),
                atol=atol,
                rtol=rtol,
                msg=(
                    f"variant={exp.name}, shape={shape.name}, ref={ref_kind}, "
                    f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
                ),
            )
            row.update(
                {
                    "check_status": "ok",
                    "ref_kind": ref_kind,
                    "max_abs": f"{max_abs:.6e}",
                    "mean_abs": f"{mean_abs:.6e}",
                }
            )

        if args.perf:
            bench = triton.testing.do_bench(
                _run_once,
                warmup=args.warmup,
                rep=args.rep,
                quantiles=(0.5, 0.2, 0.8),
            )
            if isinstance(bench, (tuple, list)):
                ms, p20, p80 = float(bench[0]), float(bench[1]), float(bench[2])
            else:
                ms = p20 = p80 = float(bench)
            flops = attn_flops(shape)
            row.update(
                {
                    "ms": f"{ms:.6f}",
                    "p20_ms": f"{p20:.6f}",
                    "p80_ms": f"{p80:.6f}",
                    "tflops_approx": f"{flops / (ms * 1e-3) / 1e12:.3f}",
                }
            )

        row["status"] = "ok"
    except ModuleNotFoundError as exc:
        row = {
            "variant": args._worker_variant,
            "shape": args._worker_shape,
            "status": "skipped" if exc.name in ("torch", "triton") else "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - report worker failures as CSV rows.
        row = {
            "variant": args._worker_variant,
            "shape": args._worker_shape,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    if args.list:
        for exp in REGISTRY.iter_experiments():
            print(
                f"{exp.name}: family={exp.family}, module={exp.module}, "
                f"paged={exp.paged}, description={exp.description}"
            )
        return 0
    if args._worker:
        return _run_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
