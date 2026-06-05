#!/usr/bin/env python3
"""Run split Hopper FA3 TLE kernel families.

The parent process launches every variant in a separate subprocess, so compiler
hangs or CUDA faults are recorded as CSV rows instead of stopping the matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


RESULT_PREFIX = "FA3_RESULT_JSON="
SHAPE_CHOICES = ("smoke", "prefill", "small", "decode", "paged_decode", "splitkv_decode", "bench")


@dataclass(frozen=True)
class FA3Variant:
    name: str
    force_path: str
    module: str
    shape_kind: str
    paged: bool = False
    description: str = ""


VARIANTS: dict[str, FA3Variant] = {
    "long_prefill": FA3Variant(
        "long_prefill",
        "long",
        "fa_hopper_persistent_pingpong.py",
        "prefill",
        description="Persistent long/prefill TLE path.",
    ),
    "short_small": FA3Variant(
        "short_small",
        "short",
        "fa_hopper_short.py",
        "small",
        description="Short / serve TLE path.",
    ),
    "direct_decode": FA3Variant(
        "direct_decode",
        "direct",
        "fa_hopper_direct.py",
        "decode",
        description="Direct one-pass dense decode path.",
    ),
    "direct_paged_decode": FA3Variant(
        "direct_paged_decode",
        "direct",
        "fa_hopper_direct.py",
        "paged_decode",
        paged=True,
        description="Direct one-pass paged decode path.",
    ),
    "direct_small": FA3Variant(
        "direct_small",
        "direct",
        "fa_hopper_direct.py",
        "small",
        description="Direct one-pass small dense path.",
    ),
    "splitkv_decode": FA3Variant(
        "splitkv_decode",
        "splitkv",
        "fa_hopper_splitkv.py",
        "splitkv_decode",
        description="Split-KV dense decode path.",
    ),
    "ws_simple_dense_decode": FA3Variant(
        "ws_simple_dense_decode",
        "ws_simple",
        "fa_hopper_persistent_pingpong.py",
        "decode",
        description="Persistent ping-pong WS dense decode path.",
    ),
    "ws_simple_paged_decode": FA3Variant(
        "ws_simple_paged_decode",
        "ws_simple",
        "fa_hopper_persistent_pingpong.py",
        "paged_decode",
        paged=True,
        description="Persistent ping-pong WS paged decode path.",
    ),
    "ws_simple_small_dense": FA3Variant(
        "ws_simple_small_dense",
        "ws_simple",
        "fa_hopper_persistent_pingpong.py",
        "small",
        description="Persistent ping-pong WS small dense path.",
    ),
    "ws_short_dense_decode": FA3Variant(
        "ws_short_dense_decode",
        "ws_short",
        "fa_hopper_nonpersistent_tlx_style.py",
        "decode",
        description="Nonpersistent TLX-style WS dense decode path.",
    ),
    "ws_short_paged_decode": FA3Variant(
        "ws_short_paged_decode",
        "ws_short",
        "fa_hopper_nonpersistent_tlx_style.py",
        "paged_decode",
        paged=True,
        description="Nonpersistent TLX-style WS paged decode path.",
    ),
    "ws_short_small_dense": FA3Variant(
        "ws_short_small_dense",
        "ws_short",
        "fa_hopper_nonpersistent_tlx_style.py",
        "small",
        description="Nonpersistent TLX-style WS small dense path.",
    ),
}

ALIASES: dict[str, tuple[str, ...]] = {
    "all": tuple(VARIANTS),
    "legacy": (
        "long_prefill",
        "short_small",
        "direct_decode",
        "direct_paged_decode",
        "direct_small",
        "splitkv_decode",
    ),
    "ws": tuple(name for name in VARIANTS if name.startswith("ws_")),
    "dense": tuple(name for name, spec in VARIANTS.items() if not spec.paged),
    "paged": tuple(name for name, spec in VARIANTS.items() if spec.paged),
}


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


def _resolve_variants(selection: list[str] | None) -> tuple[str, ...]:
    requested = selection or ["all"]
    resolved: list[str] = []
    for item in requested:
        names = ALIASES.get(item, (item,))
        for name in names:
            if name not in VARIANTS:
                allowed = ", ".join(sorted((*VARIANTS, *ALIASES)))
                raise KeyError(f"unknown FA3 variant {name!r}; expected one of {allowed}")
            if name not in resolved:
                resolved.append(name)
    return tuple(resolved)


def _shape_key_for_variant(requested: str, spec: FA3Variant) -> str | None:
    if requested == "smoke":
        return spec.shape_kind
    if requested == "bench":
        return {
            "prefill": "bench_prefill",
            "small": "bench_small",
            "decode": "bench_decode",
            "paged_decode": "bench_paged_decode",
            "splitkv_decode": "bench_splitkv_decode",
        }.get(spec.shape_kind)
    if requested == spec.shape_kind:
        return requested
    return None


def _worker_env(base_env: dict[str, str], spec: FA3Variant, dump_dir: Path | None) -> dict[str, str]:
    env = dict(base_env)
    env["FLAG_GEMS_FA3_TLE_FORCE_PATH"] = spec.force_path
    env["TLE_WGMMA_PIPELINE_MODE"] = "user_promise"
    if spec.paged:
        env.setdefault("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_D128", "1")
        env.setdefault("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL", "1")
    if dump_dir is not None:
        env["TRITON_ALWAYS_COMPILE"] = "1"
        env["TRITON_KERNEL_DUMP"] = "1"
        env["TRITON_DUMP_DIR"] = str(dump_dir)
    return env


def _worker_command(args: argparse.Namespace, spec: FA3Variant, shape_key: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_worker-variant",
        spec.name,
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
    for name in _resolve_variants(args.variant):
        spec = VARIANTS[name]
        shape_key = _shape_key_for_variant(args.shape, spec)
        if shape_key is None:
            rows.append(
                {
                    "variant": spec.name,
                    "force_path": spec.force_path,
                    "module": spec.module,
                    "shape_name": args.shape,
                    "status": "skipped",
                    "error": "shape is not compatible with variant",
                }
            )
            continue

        dump_dir = None
        if args.dump_ir:
            dump_dir = Path(args.dump_ir) / spec.name / shape_key
            dump_dir.mkdir(parents=True, exist_ok=True)

        env = _worker_env(os.environ, spec, dump_dir)
        cmd = _worker_command(args, spec, shape_key)
        start = time_now()
        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_sec,
                check=False,
            )
            elapsed = time_now() - start
            row = _parse_worker_result(completed.stdout)
            if row is None:
                row = {
                    "variant": spec.name,
                    "shape_name": shape_key,
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
                "variant": spec.name,
                "shape_name": shape_key,
                "status": "timeout",
                "error_type": "TimeoutExpired",
                "error": str(exc),
                "elapsed_sec": f"{args.timeout_sec:.3f}",
            }

        row.setdefault("force_path", spec.force_path)
        row.setdefault("module", spec.module)
        ttgir, ptx = _find_first_ir_files(dump_dir)
        row["ttgir_path"] = ttgir
        row["ptx_path"] = ptx
        rows.append(row)
        print(row)

    _write_rows(rows, args.csv)
    return 0


def time_now() -> float:
    import time

    return time.time()


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


@contextmanager
def _temporary_env(updates: dict[str, str]):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_shape(shape_key: str, spec: FA3Variant):
    from tests.hopper_fa3_utils import Shape

    if shape_key == "prefill":
        return Shape("fa3_long_prefill_b2_s512_d128_mha", [(512, 512)] * 2, 8, 8, 128, True)
    if shape_key == "bench_prefill":
        return Shape("fa3_long_prefill_b4_s2k_d128_mha", [(2048, 2048)] * 4, 32, 32, 128, True)
    if shape_key == "small":
        return Shape("fa3_small_dense_d128_gqa4", [(64, 64), (32, 96), (1, 128)], 8, 2, 128, True)
    if shape_key == "bench_small":
        return Shape("fa3_small_dense_b8_d128_gqa4", [(128, 128)] * 8, 32, 8, 128, True)
    if shape_key == "decode":
        return Shape("fa3_dense_decode_b4_k512_d128_gqa4", [(1, 512)] * 4, 8, 2, 128, True)
    if shape_key == "bench_decode":
        return Shape("fa3_dense_decode_b16_k2k_d128_gqa4", [(1, 2048)] * 16, 32, 8, 128, True)
    if shape_key == "paged_decode":
        return Shape(
            "fa3_paged_decode_b4_kmix_d128_gqa4",
            [(1, 128), (1, 192), (1, 256), (1, 320)],
            8,
            2,
            128,
            True,
            paged=True,
            block_size=16,
        )
    if shape_key == "bench_paged_decode":
        return Shape(
            "fa3_paged_decode_b8_kmix_d192_gqa4",
            [(1, 1024 + 128 * i) for i in range(8)],
            32,
            8,
            192,
            True,
            paged=True,
            block_size=16,
        )
    if shape_key == "splitkv_decode":
        return Shape("fa3_splitkv_decode_b4_k2k_d128_gqa4", [(1, 2048)] * 4, 8, 2, 128, True)
    if shape_key == "bench_splitkv_decode":
        return Shape("fa3_splitkv_decode_b16_k4k_d128_gqa4", [(1, 4096)] * 16, 32, 8, 128, True)
    raise ValueError(f"unknown shape key {shape_key!r} for {spec.name}")


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
            output_tensor,
            tolerances,
        )
        from flag_gems.runtime.backend._nvidia.hopper.ops.flash_api_v3 import (
            mha_varlan_fwd_v3,
        )

        spec = VARIANTS[args._worker_variant]
        shape = _make_shape(args._worker_shape, spec)
        row: dict[str, Any] = {
            "variant": spec.name,
            "force_path": spec.force_path,
            "module": spec.module,
            "shape_name": shape.name,
            "paged": shape.paged,
            "head_dim": shape.head_dim,
            "max_seqlen_q": max(q for q, _ in shape.seq_lens),
            "max_seqlen_k": max(k for _, k in shape.seq_lens),
            "check_status": "",
            "max_abs": "",
            "mean_abs": "",
            "ms": "",
            "tflops_approx": "",
        }

        if not is_fa3_supported():
            row.update({"status": "skipped", "error": "requires CUDA Hopper with TLE FA3 support"})
            print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
            return 0

        env_updates = {
            "FLAG_GEMS_FA3_TLE_FORCE_PATH": spec.force_path,
            "TLE_WGMMA_PIPELINE_MODE": "user_promise",
        }
        if spec.paged:
            env_updates["FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_D128"] = "1"
            env_updates["FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL"] = "1"

        with _temporary_env(env_updates):
            tensors = make_varlen(shape, torch.float16, flag_gems.device, seed=args.seed)

            def _run_direct():
                cu_seqlens_k = tensors.cu_seqlens_k
                if cu_seqlens_k is None:
                    cu_seqlens_k = torch.empty(
                        (len(shape.seq_lens) + 1,),
                        dtype=torch.int32,
                        device=tensors.q.device,
                    )
                return mha_varlan_fwd_v3(
                    tensors.q,
                    tensors.k,
                    tensors.v,
                    None,
                    tensors.cu_seqlens_q,
                    cu_seqlens_k,
                    tensors.seqused_k if shape.paged else None,
                    None,
                    tensors.block_table,
                    None,
                    tensors.max_seqlen_q,
                    tensors.max_seqlen_k,
                    0.0,
                    1.0 / math.sqrt(shape.head_dim),
                    False,
                    shape.causal,
                    -1,
                    -1,
                    0.0,
                    False,
                    None,
                )

            out = output_tensor(_run_direct())
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
                        f"variant={spec.name}, shape={shape.name}, ref={ref_kind}, "
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
                def _bench_run():
                    output_tensor(_run_direct())

                bench = triton.testing.do_bench(
                    _bench_run,
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
            "shape_name": args._worker_shape,
            "status": "skipped" if exc.name in ("torch", "triton") else "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - report worker failures as CSV rows.
        row = {
            "variant": args._worker_variant,
            "shape_name": args._worker_shape,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    print(RESULT_PREFIX + json.dumps(row, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    if args.list:
        for name, spec in VARIANTS.items():
            print(
                f"{name}: force_path={spec.force_path}, module={spec.module}, "
                f"shape={spec.shape_kind}, paged={spec.paged}"
            )
        return 0
    if args._worker:
        return _run_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
