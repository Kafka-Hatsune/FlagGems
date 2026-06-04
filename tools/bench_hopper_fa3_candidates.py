"""Benchmark force-only Hopper FA3 TLE WGMMA candidate kernels.

This script intentionally does not change default dispatch.  It runs selected
candidate force paths in isolated measurements and writes one JSON file per
candidate so failed or slow candidates are easy to inspect independently.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Callable


CANDIDATES = (
    "ws_sync_decode",
    "ws_pipe2_decode",
    "ws_sync_small",
    "ws_sync_paged_decode",
    "ws_pipe2_paged_decode",
)

HOPPER_CASE_NAMES = {
    "decode_b16_kv1k_d128_gqa4",
    "decode_b8_kv1k_d192_gqa4",
    "decode_b8_kv1k_d256_gqa4",
    "decode_b32_kv2k_d128_gqa4",
    "paged_decode_b16_kvmix_bs16_d128_gqa4",
    "paged_decode_b8_bs16_d192_gqa4",
    "paged_decode_b8_bs16_d256_gqa4",
    "paged_decode_b64_bs16_d128_gqa4",
}
HOPPER_CASES = {}


def _standard_configs():
    all_cu_seq_lens_q = [
        (0, 512),
        (0, 1, 2, 72),
        tuple(range(0, 45))
        + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
        tuple(range(0, 196)) + (211, 226, 240, 253, 265),
    ]
    all_seqused_k = [
        (512,),
        (1, 1, 70),
        (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
        (2333,)
        + (2331,) * 20
        + (2330,) * 20
        + (2329,) * 14
        + (2328,) * 18
        + (2327,) * 15
        + (2326,) * 17
        + (2325,) * 18
        + (2324,) * 21
        + (2323,) * 22
        + (2322,) * 24
        + (2321,) * 5
        + (2320, 2319, 2318, 2317, 2316),
    ]
    for idx, (cu_q, used_k) in enumerate(zip(all_cu_seq_lens_q, all_seqused_k)):
        max_q = max(x - y for x, y in zip(cu_q[1:], cu_q[:-1]))
        max_k = max(used_k)
        yield {
            "name": f"std#{idx}_q{max_q}_k{max_k}_d128_h16_kv8_paged",
            "cu_query_lens": cu_q,
            "seqused_k": used_k,
            "num_query_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "block_size": 16,
            "num_blocks": 2000,
            "causal": True,
        }


STANDARD_CASES = {cfg["name"]: cfg for cfg in _standard_configs()}
RISKY_PAGED_SMALL_CASES = tuple(STANDARD_CASES)

CASE_GROUPS = {
    "ws_sync_decode": (
        "decode_b16_kv1k_d128_gqa4",
        "decode_b8_kv1k_d192_gqa4",
        "decode_b8_kv1k_d256_gqa4",
        "decode_b32_kv2k_d128_gqa4",
    ),
    "ws_pipe2_decode": (
        "decode_b16_kv1k_d128_gqa4",
        "decode_b8_kv1k_d192_gqa4",
        "decode_b8_kv1k_d256_gqa4",
        "decode_b32_kv2k_d128_gqa4",
    ),
    # Paged std#0..std#3 currently expose a ws_sync_small barrier deadlock.  Keep
    # this candidate dense-only by default; use the risky flags below for debug.
    "ws_sync_small": (),
    "ws_sync_paged_decode": (
        "paged_decode_b16_kvmix_bs16_d128_gqa4",
        "paged_decode_b8_bs16_d192_gqa4",
        "paged_decode_b8_bs16_d256_gqa4",
        "paged_decode_b64_bs16_d128_gqa4",
    ),
    "ws_pipe2_paged_decode": (
        "paged_decode_b16_kvmix_bs16_d128_gqa4",
        "paged_decode_b8_bs16_d192_gqa4",
        "paged_decode_b8_bs16_d256_gqa4",
        "paged_decode_b64_bs16_d128_gqa4",
    ),
}


def _dtype_from_name(name: str) -> torch.dtype:
    if name in ("fp16", "float16", "torch.float16"):
        return torch.float16
    if name in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def _load_runtime() -> None:
    global HOPPER_CASES
    global benchmark_shapes, flag_gems, is_fa3_supported, make_varlen, run_flag_gems
    global torch, triton

    import torch as _torch
    import triton as _triton

    import flag_gems as _flag_gems
    from tests.hopper_fa3_utils import (
        benchmark_shapes as _benchmark_shapes,
        is_fa3_supported as _is_fa3_supported,
        make_varlen as _make_varlen,
        run_flag_gems as _run_flag_gems,
    )

    torch = _torch
    triton = _triton
    flag_gems = _flag_gems
    benchmark_shapes = _benchmark_shapes
    is_fa3_supported = _is_fa3_supported
    make_varlen = _make_varlen
    run_flag_gems = _run_flag_gems
    HOPPER_CASES = {
        shape.name: shape
        for shape in benchmark_shapes()
        if shape.name in HOPPER_CASE_NAMES
    }


def _default_out_dir(tag: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "FlagTree" / f"hopper_fa3_candidates_{tag}"


def _make_standard_call(cfg: dict, dtype: torch.dtype, device: str, seed: int):
    torch.manual_seed(seed)
    cu_query_lens = cfg["cu_query_lens"]
    seqused_k = cfg["seqused_k"]
    num_query_heads = cfg["num_query_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    block_size = cfg["block_size"]
    num_blocks = cfg["num_blocks"]
    max_q = max(x - y for x, y in zip(cu_query_lens[1:], cu_query_lens[:-1]))
    max_k = max(seqused_k)
    num_seqs = len(seqused_k)

    query = torch.randn(
        cu_query_lens[-1],
        num_query_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens_t = torch.tensor(cu_query_lens, dtype=torch.int32, device=device)
    seqused_k_t = torch.tensor(seqused_k, dtype=torch.int32, device=device)
    max_num_blocks = (max_k + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks),
        dtype=torch.int32,
        device=device,
    )

    def call():
        return flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens_t,
            seqused_k=seqused_k_t,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=head_dim**-0.5,
            causal=cfg["causal"],
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            alibi_slopes=None,
            fa_version=3,
        )

    meta = {
        "case_group": "standard",
        "shape": {
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "paged": True,
            "block_size": block_size,
            "num_seqs": num_seqs,
            "total_q": cu_query_lens[-1],
        },
    }
    return call, meta


def _make_hopper_call(shape: Shape, dtype: torch.dtype, device: str, seed: int):
    tensors = make_varlen(shape, dtype, device, seed)

    def call():
        return run_flag_gems(tensors, shape, fa_version=3)

    meta = {
        "case_group": "hopper",
        "shape": {
            "seq_lens": shape.seq_lens,
            "num_query_heads": shape.nh_q,
            "num_kv_heads": shape.nh_k,
            "head_dim": shape.head_dim,
            "causal": shape.causal,
            "paged": shape.paged,
            "block_size": shape.block_size,
            "max_seqlen_q": tensors.max_seqlen_q,
            "max_seqlen_k": tensors.max_seqlen_k,
            "total_q": int(tensors.cu_seqlens_q[-1].item()),
        },
    }
    return call, meta


def _bench_call(call: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    def wrapped():
        out = call()
        if isinstance(out, tuple):
            out = out[0]
        # Keep the launch live without introducing a host-side reduction.
        return out

    return float(triton.testing.do_bench(wrapped, warmup=warmup, rep=rep))


def _run_one(
    *,
    candidate: str,
    case_name: str,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
    seed: int,
    allow_risky_paged_small: bool,
) -> dict:
    device = flag_gems.device
    os.environ["FLAG_GEMS_FA3_TLE_FORCE_PATH"] = candidate
    if allow_risky_paged_small:
        os.environ["FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL"] = "1"
    else:
        os.environ.pop("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL", None)
    os.environ.pop("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", None)
    os.environ.pop("FLAG_GEMS_FA3_TLE_SMALL_STRATEGY", None)
    os.environ.pop("FLAG_GEMS_FA3_TLE_WS_STRATEGY", None)

    if case_name in HOPPER_CASES:
        call, meta = _make_hopper_call(HOPPER_CASES[case_name], dtype, device, seed)
    elif case_name in STANDARD_CASES:
        call, meta = _make_standard_call(STANDARD_CASES[case_name], dtype, device, seed)
    else:
        raise KeyError(f"unknown case {case_name!r}")

    record = {
        "candidate": candidate,
        "case": case_name,
        "dtype": str(dtype),
        "status": "ok",
        "latency_ms": None,
        "error": None,
        **meta,
    }
    try:
        with torch.inference_mode():
            latency = _bench_call(call, warmup=warmup, rep=rep)
            torch.cuda.synchronize()
        record["latency_ms"] = latency
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    return record


def _selected_cases(
    candidates: tuple[str, ...],
    explicit_cases: tuple[str, ...],
    include_risky_paged_small: bool,
):
    selected = {}
    for candidate in candidates:
        cases = CASE_GROUPS[candidate]
        if include_risky_paged_small and candidate == "ws_sync_small":
            cases = cases + RISKY_PAGED_SMALL_CASES
        if explicit_cases:
            cases = tuple(case for case in cases if case in explicit_cases)
        selected[candidate] = cases
    return selected


def _parse_csv(value: str | None, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return allowed
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"unknown values {unknown}; allowed: {', '.join(allowed)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark force-only Hopper FA3 TLE candidate kernels."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Run fp16 compile smoke.")
    mode.add_argument("--full", action="store_true", help="Run fp16 and bf16 latency.")
    parser.add_argument("--tag", default="tle10")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--candidates", default=None, help="Comma-separated candidates.")
    parser.add_argument("--cases", default=None, help="Comma-separated case names.")
    parser.add_argument("--dtypes", default=None, help="Comma-separated fp16,bf16 list.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument(
        "--include-risky-paged-small",
        action="store_true",
        help="Include std#0..std#3 under ws_sync_small. This is known risky.",
    )
    parser.add_argument(
        "--allow-risky-paged-small-kernel",
        action="store_true",
        help="Actually let ws_sync_small launch paged small kernels.",
    )
    args = parser.parse_args()

    candidates = _parse_csv(args.candidates, CANDIDATES)
    all_cases = tuple(sorted(HOPPER_CASE_NAMES | set(STANDARD_CASES)))
    explicit_cases = () if not args.cases else _parse_csv(args.cases, all_cases)

    _load_runtime()
    if not is_fa3_supported():
        raise RuntimeError("requires CUDA Hopper with TLE FA3 support")

    dtype_names = ("fp16",) if args.smoke else ("fp16", "bf16")
    if args.dtypes:
        dtype_names = _parse_csv(args.dtypes, ("fp16", "bf16"))
    dtypes = tuple(_dtype_from_name(name) for name in dtype_names)

    out_dir = args.out_dir or _default_out_dir(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_cases(
        candidates, explicit_cases, args.include_risky_paged_small
    )

    for candidate in candidates:
        records = []
        if not selected[candidate]:
            print(
                f"[{candidate}] no default cases; use --include-risky-paged-small "
                "for std#0..std#3 debug",
                flush=True,
            )
        for case_idx, case_name in enumerate(selected[candidate]):
            for dtype in dtypes:
                print(f"[{candidate}] {case_name} {dtype}", flush=True)
                records.append(
                    _run_one(
                        candidate=candidate,
                        case_name=case_name,
                        dtype=dtype,
                        warmup=args.warmup,
                        rep=args.rep,
                        seed=args.seed + case_idx,
                        allow_risky_paged_small=args.allow_risky_paged_small_kernel,
                    )
                )
        payload = {
            "tag": args.tag,
            "candidate": candidate,
            "mode": "smoke" if args.smoke else "full",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "",
            "warmup": args.warmup,
            "rep": args.rep,
            "records": records,
        }
        out_file = out_dir / f"{candidate}.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_file}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
