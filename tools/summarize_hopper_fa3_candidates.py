"""Summarize Hopper FA3 TLE candidate benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import geometric_mean


def _std_case_name(index: int, shape_detail) -> str:
    args = shape_detail[0]
    q_shape = args[0]
    k_shape = args[1]
    max_q = args[3]
    max_k = args[5]
    return (
        f"std#{index}_q{max_q}_k{max_k}_d{q_shape[2]}"
        f"_h{q_shape[1]}_kv{k_shape[2]}_paged"
    )


def _record_name(op_name: str, index: int, record: dict) -> str:
    shape_detail = record.get("shape_detail")
    if isinstance(shape_detail, dict):
        return shape_detail["name"]
    if op_name == "flash_attn_varlen_func" and isinstance(shape_detail, list):
        return _std_case_name(index, shape_detail)
    return f"{op_name}#{index}"


def _load_baseline(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    latencies = {}
    for op_name, op_payload in payload.items():
        for detail in op_payload.get("details", []):
            dtype = detail.get("dtype", "")
            detail_op = detail.get("op_name", op_name)
            for index, record in enumerate(detail.get("result", [])):
                latency = record.get("latency")
                if latency is None:
                    continue
                name = _record_name(detail_op, index, record)
                latencies[(name, dtype)] = float(latency)
    return latencies


def _load_candidates(path: Path) -> list[dict]:
    files = []
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.json"))
    rows = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        candidate = payload.get("candidate", file.stem)
        for record in payload.get("records", []):
            rows.append(
                {
                    "candidate": record.get("candidate", candidate),
                    "case": record.get("case"),
                    "dtype": record.get("dtype"),
                    "status": record.get("status"),
                    "latency_ms": record.get("latency_ms"),
                    "error": record.get("error"),
                    "case_group": record.get("case_group"),
                }
            )
    return rows


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Hopper FA3 candidate latency versus FA2/FA3 baselines."
    )
    parser.add_argument("--fa2", type=Path, required=True)
    parser.add_argument("--fa3", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    fa2 = _load_baseline(args.fa2)
    fa3 = _load_baseline(args.fa3)
    candidate_rows = _load_candidates(args.candidates)
    summary = []
    ratios = []

    for row in sorted(
        candidate_rows,
        key=lambda r: (r["candidate"] or "", r["case"] or "", r["dtype"] or ""),
    ):
        case = row["case"]
        dtype = row["dtype"]
        latency = row["latency_ms"]
        fa2_latency = fa2.get((case, dtype))
        fa3_latency = fa3.get((case, dtype))
        ratio_fa2 = None
        ratio_tle9 = None
        if row["status"] == "ok" and latency:
            if fa2_latency:
                ratio_fa2 = fa2_latency / latency
                ratios.append(ratio_fa2)
            if fa3_latency:
                ratio_tle9 = fa3_latency / latency
        summary.append(
            {
                **row,
                "fa2_latency_ms": fa2_latency,
                "tle9_fa3_latency_ms": fa3_latency,
                "candidate_over_fa2": ratio_fa2,
                "candidate_over_tle9_fa3": ratio_tle9,
            }
        )

    header = (
        "| candidate | case | dtype | status | cand ms | FA3/FA2 | cand/tle9 FA3 | error |\n"
        "|---|---|---:|---|---:|---:|---:|---|"
    )
    lines = [header]
    for row in summary:
        lines.append(
            "| {candidate} | {case} | {dtype} | {status} | {lat} | {ratio_fa2} | "
            "{ratio_tle9} | {error} |".format(
                candidate=row["candidate"],
                case=row["case"],
                dtype=row["dtype"].replace("torch.", "") if row["dtype"] else "",
                status=row["status"],
                lat=_fmt(row["latency_ms"], 4),
                ratio_fa2=_fmt(row["candidate_over_fa2"]),
                ratio_tle9=_fmt(row["candidate_over_tle9_fa3"]),
                error=(row["error"] or "").replace("|", "/")[:180],
            )
        )

    if ratios:
        lines.append("")
        lines.append(
            f"candidate_over_fa2: min={min(ratios):.3f}, "
            f"geomean={geometric_mean(ratios):.3f}, "
            f"wins={sum(r > 1.0 for r in ratios)}/{len(ratios)}"
        )

    print("\n".join(lines))

    output = args.output
    if output is None:
        output = (
            args.candidates / "summary.json"
            if args.candidates.is_dir()
            else args.candidates.with_name(args.candidates.stem + "_summary.json")
        )
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
