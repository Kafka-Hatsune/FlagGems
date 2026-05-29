import pytest
import torch

from . import base, consts

HOPPER_MM_VERSION_ENV = "FLAG_GEMS_HOPPER_MM_VERSION"


def _selected_hopper_mm_version(pytestconfig) -> str:
    return pytestconfig.getoption("hopper_mm_version")


def _set_hopper_mm_version(monkeypatch, pytestconfig) -> str:
    version = _selected_hopper_mm_version(pytestconfig)
    monkeypatch.setenv(HOPPER_MM_VERSION_ENV, version)
    return version


def _skip_unless_hopper_wasp(pytestconfig) -> None:
    if _selected_hopper_mm_version(pytestconfig) != "wasp":
        return
    if base.device != "cuda" or not torch.cuda.is_available():
        pytest.skip("WASP mm benchmark requires CUDA.")
    try:
        from flag_gems.runtime.backend._nvidia.hopper.ops.mm import (
            is_wasp_mm_supported,
        )
    except Exception:
        pytest.skip("WASP mm benchmark support probe requires Hopper backend.")
    if not is_wasp_mm_supported():
        pytest.skip("WASP mm benchmark requires CUDA Hopper with TLE support.")


def _wasp_shape_supported(m: int, n: int, k: int) -> bool:
    return (
        m >= 128
        and n >= 128
        and k >= 64
        and m % 128 == 0
        and n % 128 == 0
        and k % 64 == 0
    )


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


class HopperMmBenchmark(base.BlasBenchmark):
    def __init__(self, *args, hopper_mm_version: str = "original", **kwargs):
        super().__init__(*args, **kwargs)
        self.hopper_mm_version = hopper_mm_version

    def get_input_iter(self, dtype):
        for b, m, n, k in self.shapes:
            if self.hopper_mm_version == "wasp" and not _wasp_shape_supported(
                m, n, k
            ):
                continue
            yield from self.input_fn(b, m, n, k, dtype, self.device, False)

        if (
            self.hopper_mm_version != "wasp"
            and base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        ):
            for b, m, n, k in self.shapes:
                yield from self.input_fn(b, m, n, k, dtype, self.device, True)


@pytest.mark.mm
def test_mm(monkeypatch, pytestconfig):
    version = _set_hopper_mm_version(monkeypatch, pytestconfig)
    _skip_unless_hopper_wasp(pytestconfig)
    bench = HopperMmBenchmark(
        op_name="mm",
        input_fn=mm_input_fn,
        torch_op=torch.Tensor.mm,
        dtypes=[torch.float16] if version == "wasp" else consts.FLOAT_DTYPES,
        hopper_mm_version=version,
    )

    bench.run()


class MmSelfTransposeBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        m, k = args[0].shape
        return 2 * m * m * k


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    inp = torch.randn([k, m], dtype=cur_dtype, device=device).t()

    yield inp,


def torch_mm_self_transpose(inp):
    return torch.mm(inp, inp.t())


@pytest.mark.mm
def test_mm_self_transpose_benchmark(monkeypatch, pytestconfig):
    version = _set_hopper_mm_version(monkeypatch, pytestconfig)
    if version == "wasp":
        pytest.skip("WASP mm path currently covers row-major mm/mm_out only.")
    bench = MmSelfTransposeBenchmark(
        op_name="mm_self_transpose",
        input_fn=_input_fn,
        torch_op=torch_mm_self_transpose,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def mm_out_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2, {"out": out}


@pytest.mark.mm_out
def test_mm_out(monkeypatch, pytestconfig):
    version = _set_hopper_mm_version(monkeypatch, pytestconfig)
    _skip_unless_hopper_wasp(pytestconfig)
    bench = HopperMmBenchmark(
        op_name="mm_out",
        input_fn=mm_out_input_fn,
        torch_op=torch.mm,
        dtypes=[torch.float16] if version == "wasp" else consts.FLOAT_DTYPES,
        hopper_mm_version=version,
    )

    bench.run()
