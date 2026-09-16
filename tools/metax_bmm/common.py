"""Frozen input and graph timing utilities (benchmark/reference code only)."""

import json
import pathlib
import statistics

import torch


def cases():
    return json.loads(pathlib.Path(__file__).with_name("active_cases.json").read_text())


def inputs(case):
    batch, m, n, k = case["shape"]
    dtype = getattr(torch, case["dtype"])
    layout = case["layout"]
    torch.manual_seed(1729)
    a = torch.randn(
        (batch, k, m) if layout[0] == "T" else (batch, m, k), device="cuda", dtype=dtype
    )
    b = torch.randn(
        (batch, n, k) if layout[1] == "T" else (batch, k, n), device="cuda", dtype=dtype
    )
    return a.transpose(1, 2) if layout[0] == "T" else a, (
        b.transpose(1, 2) if layout[1] == "T" else b
    )


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(graph):
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    begin.record()
    for _ in range(20):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / 320


def paired(graphs, rounds=7):
    samples = [[] for _ in graphs]
    for r in range(rounds):
        order = range(len(graphs)) if r % 2 == 0 else reversed(range(len(graphs)))
        for i in order:
            samples[i].append(measure(graphs[i]))
    return [{"us": statistics.median(s), "samples": s} for s in samples]


def check(actual, reference, dtype):
    # Same tolerance for every optimization trial; no per-shape relaxation.
    atol, rtol = {
        torch.bfloat16: (0.125, 0.02),
        torch.float16: (0.016, 0.003),
        torch.float32: (0.002, 0.0001),
    }[dtype]
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)


def emit(row, output):
    line = json.dumps(row)
    print(line, flush=True)
    with open(output, "a") as f:
        f.write(line + "\n")
