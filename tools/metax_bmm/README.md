# MetaX BMM comparison tools

These tools and their frozen 84 cases were written for the earlier MetaX
optimization task, then migrated from `FlagGems-vllm-lt` commit
`53afd02734f732a8cee61ae1cf5632a35690e159`. They are not an upstream workload
standard. The destination FlagGems repository's original `tests/test_bmm.py`,
`tests/test_mm.py`, and `tests/test_mv.py` remain the primary correctness checks.

Inside `zhangshen_vllm_0.20.0_metax`:

```bash
conda activate flagtree36
cd /workspace/FlagGems-lt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0  # Select an idle device first.

# Repository correctness and migration contracts.
python -m pytest -q tests/test_bmm.py tests/test_mm.py tests/test_mv.py tests/test_metax_mv.py tests/test_metax_matmul_migration.py

# Repository benchmark with explicit CUDA Graph timing.
python -m pytest -q -s benchmark/test_bmm.py::test_bmm benchmark/test_mv.py::test_mv --mode cudagraph --warmup 5 --iter 20

# Historical 84-case comparison; choose a new output filename for each run.
python tools/metax_bmm/bench.py --output /workspace/bmm-lt-run01.jsonl
```

`bench.py` always captures CUDA Graphs after warmup. Each graph contains 16
complete calls; each measurement replays it 20 times. Seven rounds alternate
the order of native and candidate measurement. Packing and partial reduction
are included; compilation, tuning, and CPU view construction are excluded.
Use `--ids 2,4,7` to select frozen case IDs. Output is appended as JSON lines.

The historical comparison keeps `torch.backends.cuda.matmul.allow_tf32=False`
and the fixed checker in `common.py` for continuity. These settings do not add
an IEEE-only requirement to the project. Accuracy acceptance uses the
destination repository's unchanged tests and tolerances.

The migration keeps the destination's optimized MetaX `mm.py` and its tuning
configuration intact. The previous MM is archived as `mm.py.bak`, and the
destination MM also has an identical `mm.upstream.py.bak`. Neither backup is
imported. Compatible batch-one matrix products call the destination `mm_out`;
vectors call MV's shared executor directly, preserving the BMM FP32-output
contract without adding an input-preparation kernel. BMM owns its batched
split-K policy and matrix-partial reduction. Other batches and irregular matrix views use BMM's migrated
compute and scheduling. Thus earlier VLLM batch-one performance results do not
describe the destination MM. This migration does not establish 95% native
performance for all 84 cases.
