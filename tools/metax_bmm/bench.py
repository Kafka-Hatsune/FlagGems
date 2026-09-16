"""Frozen 84-case native comparison. Run outside any use_gems context."""

import argparse
import importlib
import json
import pathlib
import subprocess
import sys

import torch
import triton
from common import capture, cases, check, emit, inputs, paired


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--ids", default="")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--label", default="candidate")
    args = p.parse_args()
    import flag_gems

    torch.backends.cuda.matmul.allow_tf32 = False
    emit(
        {
            "environment": {
                "python": sys.executable,
                "torch": torch.__version__,
                "triton": triton.__version__,
                "triton_path": triton.__file__,
                "flag_gems": flag_gems.__file__,
                "target": str(triton.runtime.driver.active.get_current_target()),
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "status": subprocess.check_output(
                    ["git", "status", "--short"], text=True
                ),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "active_cases": json.loads(
                    pathlib.Path(__file__).with_name("active_cases.json").read_text()
                ),
                "label": args.label,
            }
        },
        args.output,
    )
    for case in cases():
        if args.ids and str(case["id"]) not in args.ids.split(","):
            continue
        row = {"case": case, "label": args.label}
        try:
            a, b = inputs(case)
            expected = torch.bmm(a, b)
            got = flag_gems.bmm(a, b)
            check(got, expected, a.dtype)
            row["max_abs"] = (got.float() - expected.float()).abs().max().item()
            row["strides"] = [a.stride(), b.stride()]
            native_graph = capture(lambda: torch.bmm(a, b))
            candidate_graph = capture(lambda: flag_gems.bmm(a, b))
            native, candidate = paired(
                [native_graph, candidate_graph], rounds=args.rounds
            )
            row.update(
                native=native,
                candidate=candidate,
                ratio=native["us"] / candidate["us"],
                pass95=native["us"] / candidate["us"] >= 0.95,
            )
            module = importlib.import_module(flag_gems.bmm.__module__)
            if hasattr(module, "_selected_config"):
                row["config"] = module._selected_config(a, b, got)
        except Exception as e:
            row["error"] = repr(e)
        emit(row, args.output)


if __name__ == "__main__":
    main()
