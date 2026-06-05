#!/usr/bin/env python3
"""Run experimental Hopper FA3 warp-specialized variants.

The parent process schedules variants and enforces timeouts.  Each actual
kernel run happens in a child process so a hang or compiler failure is recorded
without stopping the whole matrix.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


SHAPE_CHOICES = ("smoke", "decode", "paged_decode", "small", "bench")
RESULT_PREFIX = "FA3_WS_RESULT_JSON="
REGISTRY_PATH = (
    SRC_ROOT
    / "flag_gems"
    / "runtime"
    / "backend"
    / "_nvidia"
    / "hopper"
    / "ops"
    / "fa3_ws"
    / "registry.py"
)


def _load_registry():
    try:
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws import registry

        return registry
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location("fa3_ws_registry", REGISTRY_PATH)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_REGISTRY = _load_registry()
WSVariant = _REGISTRY.WSVariant
get_variant = _REGISTRY.get_variant
resolve_variant_names = _REGISTRY.resolve_variant_names
variant_names = _REGISTRY.variant_names


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
    parser.add_argument("--_worker-dump-ir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _shape_key_for_variant(requested: str, spec: WSVariant) -> str | None:
    if requested == "smoke":
        return spec.shape_kind
    if requested == "bench":
        return "bench_paged_decode" if spec.paged else "bench_decode"
    if requested == "decode":
        return "paged_decode" if spec.paged else "decode"
    if requested == "paged_decode":
        return "paged_decode" if spec.paged else None
    if requested == "small":
        return "small" if not spec.paged else None
    return requested


def _shape_metadata(shape_key: str, spec: WSVariant) -> dict[str, Any]:
    if shape_key == "decode":
        if spec.persistent:
            return {
                "shape_name": "ws_persistent_dense_decode_b1_k512_d128_mha",
                "head_dim": 128,
                "max_seqlen_q": 1,
                "max_seqlen_k": 512,
            }
        return {
            "shape_name": "ws_dense_decode_b4_k1k_d128_gqa4",
            "head_dim": 128,
            "max_seqlen_q": 1,
            "max_seqlen_k": 1024,
        }
    if shape_key == "paged_decode":
        if spec.persistent:
            return {
                "shape_name": "ws_persistent_paged_decode_b1_k128_d128_mha",
                "head_dim": 128,
                "max_seqlen_q": 1,
                "max_seqlen_k": 128,
            }
        return {
            "shape_name": "ws_paged_decode_b1_k128_d128_mha",
            "head_dim": 128,
            "max_seqlen_q": 1,
            "max_seqlen_k": 128,
        }
    if shape_key == "small":
        return {
            "shape_name": "ws_small_dense_mixed_d128_gqa4",
            "head_dim": 128,
            "max_seqlen_q": 64,
            "max_seqlen_k": 128,
        }
    if shape_key == "bench_decode":
        if spec.persistent:
            return {
                "shape_name": "ws_bench_persistent_decode_b8_k2k_d128_mha",
                "head_dim": 128,
                "max_seqlen_q": 1,
                "max_seqlen_k": 2048,
            }
        return {
            "shape_name": "ws_bench_decode_b16_k2k_d128_gqa4",
            "head_dim": 128,
            "max_seqlen_q": 1,
            "max_seqlen_k": 2048,
        }
    if shape_key == "bench_paged_decode":
        return {
            "shape_name": "ws_bench_paged_decode_b8_kmix_d192_gqa4",
            "head_dim": 192,
            "max_seqlen_q": 1,
            "max_seqlen_k": 1920,
        }
    return {}


def _worker_env(base_env: dict[str, str], spec: WSVariant, dump_dir: Path | None) -> dict[str, str]:
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


def _install_variant_kernel(spec: WSVariant) -> None:
    from flag_gems.runtime.backend._nvidia.hopper.ops import flash_api_v3

    if spec.kernel_module == "fa_hopper_persistent_pingpong":
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws import (
            fa_hopper_persistent_pingpong,
        )

        flash_api_v3.flash_varlen_fwd_v3_tle_kernel = (
            fa_hopper_persistent_pingpong.flash_varlen_fwd_v3_tle_kernel
        )
        flash_api_v3.flash_varlen_fwd_v3_tle_ws_simple_kernel = (
            fa_hopper_persistent_pingpong.flash_varlen_fwd_v3_tle_ws_simple_kernel
        )
        return

    if spec.kernel_module == "fa_hopper_nonpersistent_tlx_style":
        from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws import (
            fa_hopper_nonpersistent_tlx_style,
        )

        flash_api_v3.flash_varlen_fwd_v3_tle_ws_short_kernel = (
            fa_hopper_nonpersistent_tlx_style.flash_varlen_fwd_v3_tle_ws_short_kernel
        )
        return

    raise RuntimeError(f"unknown FA3 WS kernel module {spec.kernel_module!r}")


def _worker_command(args: argparse.Namespace, spec: WSVariant, shape_key: str, dump_dir: Path | None) -> list[str]:
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
    if dump_dir is not None:
        cmd.extend(["--_worker-dump-ir", str(dump_dir)])
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


def _check_ir_sync(dump_dir: Path | None, spec: WSVariant) -> tuple[str, str]:
    if dump_dir is None or not dump_dir.exists():
        return "", ""
    text = ""
    for path in list(dump_dir.rglob("*.ttgir")) + list(dump_dir.rglob("*.ptx")):
        try:
            text += "\n" + path.read_text(errors="ignore")
        except OSError:
            continue

    messages: list[str] = []
    if re.search(r"bar\.(?:sync|arrive)\s+\d+\s*,\s*512\b", text):
        messages.append("named barrier contains 512; expected 256 for two consumer WGs")
    if spec.paged:
        if "fence.proxy.async.shared::cta" not in text and "fence_async_shared" not in text:
            messages.append("paged/manual path missing async shared fence marker")
        if "mbarrier.try_wait.parity" not in text and "wait_barrier" not in text:
            messages.append("paged/manual path missing mbarrier wait marker")
        if "mbarrier.arrive" not in text and "arrive_barrier" not in text:
            messages.append("paged/manual path missing mbarrier arrive marker")
    if messages:
        return "error", "; ".join(messages)
    return "ok", ""


def _write_rows(rows: list[dict[str, Any]], csv_path: str | None) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if csv_path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            file_writer = csv.DictWriter(f, fieldnames=fields)
            file_writer.writeheader()
            file_writer.writerows(rows)


def _run_parent(args: argparse.Namespace) -> int:
    requested_variants = args.variant or ["all"]
    selections = [item for group in requested_variants for item in group.split(",")]
    names = resolve_variant_names(selections)
    rows: list[dict[str, Any]] = []

    for name in names:
        spec = get_variant(name)
        shape_key = _shape_key_for_variant(args.shape, spec)
        base_row: dict[str, Any] = {
            "variant": spec.name,
            "force_path": spec.force_path,
            "shape_request": args.shape,
            "shape": shape_key or "",
            "paged": spec.paged,
            "persistent": spec.persistent,
            "sync_mode": spec.sync_mode,
            "kernel_module": spec.kernel_module,
        }
        if shape_key is None:
            rows.append({**base_row, "status": "skipped", "error": "shape incompatible with variant"})
            continue
        base_row.update(_shape_metadata(shape_key, spec))

        dump_dir = None
        if args.dump_ir:
            dump_dir = Path(args.dump_ir) / spec.name / shape_key
            dump_dir.mkdir(parents=True, exist_ok=True)

        cmd = _worker_command(args, spec, shape_key, dump_dir)
        env = _worker_env(os.environ, spec, dump_dir)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_sec,
            )
            elapsed = time.perf_counter() - start
        except subprocess.TimeoutExpired as exc:
            ttgir, ptx = _find_first_ir_files(dump_dir)
            rows.append(
                {
                    **base_row,
                    "status": "timeout",
                    "elapsed_sec": f"{args.timeout_sec:.3f}",
                    "error": f"timeout after {args.timeout_sec:.1f}s",
                    "ttgir": ttgir,
                    "ptx": ptx,
                    "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
                }
            )
            continue

        result = _parse_worker_result(proc.stdout)
        ttgir, ptx = _find_first_ir_files(dump_dir)
        ir_status, ir_error = _check_ir_sync(dump_dir, spec)
        if result is None:
            result = {
                "status": "error",
                "error": "worker did not emit result JSON",
            }
        status = result.get("status", "error")
        if proc.returncode != 0 and status == "ok":
            status = "error"
        row = {
            **base_row,
            **result,
            "status": status,
            "returncode": proc.returncode,
            "elapsed_sec": f"{elapsed:.3f}",
            "ir_check_status": ir_status,
            "ir_check_error": ir_error,
            "ttgir": ttgir,
            "ptx": ptx,
        }
        if args.verbose or status not in ("ok", "skipped") or ir_status == "error":
            row["stdout_tail"] = proc.stdout[-2000:]
            row["stderr_tail"] = proc.stderr[-2000:]
        rows.append(row)

    _write_rows(rows, args.csv)
    return 0 if all(row.get("status") in ("ok", "skipped") for row in rows) else 1


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


def _make_shape(shape_key: str, spec: WSVariant):
    from tests.hopper_fa3_utils import Shape

    if shape_key == "decode":
        if spec.persistent:
            return Shape(
                "ws_persistent_dense_decode_b1_k512_d128_mha",
                [(1, 512)],
                4,
                4,
                128,
                True,
            )
        return Shape("ws_dense_decode_b4_k1k_d128_gqa4", [(1, 1024)] * 4, 8, 2, 128, True)
    if shape_key == "paged_decode":
        if spec.persistent:
            return Shape(
                "ws_persistent_paged_decode_b1_k128_d128_mha",
                [(1, 128)],
                4,
                4,
                128,
                True,
                paged=True,
                block_size=16,
            )
        return Shape(
            "ws_paged_decode_b1_k128_d128_mha",
            [(1, 128)],
            4,
            4,
            128,
            True,
            paged=True,
            block_size=16,
        )
    if shape_key == "small":
        return Shape(
            "ws_small_dense_mixed_d128_gqa4",
            [(64, 64), (32, 96), (1, 128)],
            8,
            2,
            128,
            True,
        )
    if shape_key == "bench_decode":
        if spec.persistent:
            return Shape(
                "ws_bench_persistent_decode_b8_k2k_d128_mha",
                [(1, 2048)] * 8,
                32,
                32,
                128,
                True,
            )
        return Shape("ws_bench_decode_b16_k2k_d128_gqa4", [(1, 2048)] * 16, 32, 8, 128, True)
    if shape_key == "bench_paged_decode":
        return Shape(
            "ws_bench_paged_decode_b8_kmix_d192_gqa4",
            [(1, 1024 + 128 * i) for i in range(8)],
            32,
            8,
            192,
            True,
            paged=True,
            block_size=16,
        )
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

        spec = get_variant(args._worker_variant)
        shape = _make_shape(args._worker_shape, spec)
        row: dict[str, Any] = {
            "variant": spec.name,
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
            _install_variant_kernel(spec)
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
        for name in variant_names():
            spec = get_variant(name)
            print(
                f"{name}: force_path={spec.force_path}, persistent={spec.persistent}, "
                f"paged={spec.paged}, sync_mode={spec.sync_mode}, shape={spec.shape_kind}"
            )
        return 0
    if args._worker:
        return _run_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
