# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX BMM layout and forwarding contracts preserved during migration."""

import importlib

import flag_gems
import pytest
import torch

from . import accuracy_utils as utils

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="MetaX backend migration contracts"
)
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _inputs(batch, m, n, k, dtype, layout="NN"):
    torch.manual_seed(1729)

    def matrix(rows, columns, transposed):
        shape = (batch, columns, rows) if transposed else (batch, rows, columns)
        tensor = torch.randn(shape, device=flag_gems.device, dtype=dtype)
        return tensor.transpose(1, 2) if transposed else tensor

    if layout == "slice":
        a = matrix(m * 2, k * 2, False)[:, 1::2, 1::2]
        b = matrix(k * 2, n * 2, False)[:, 1::2, 1::2]
    else:
        a = matrix(m, k, layout[0] == "T")
        b = matrix(k, n, layout[1] == "T")
    return a, b


def _assert_product(a, b, result):
    reference = torch.bmm(utils.to_reference(a, True), utils.to_reference(b, True))
    # Use the destination repository's unchanged accuracy checker.
    utils.gems_assert_close(result, reference, result.dtype, reduce_dim=a.shape[-1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["NN", "NT", "TN", "TT", "slice", "expand"])
def test_bmm_views_and_out(dtype, layout):
    a, b = _inputs(3, 17, 33, 71, dtype, "NN" if layout == "expand" else layout)
    if layout == "expand":
        a = a[:1].expand(3, -1, -1)
        b = b[:1].expand(3, -1, -1)
    storage = torch.full((3, 17, 66), 11, device=a.device, dtype=dtype)
    out = storage[:, :, 1::2]
    with flag_gems.use_gems():
        actual = torch.bmm(a, b, out=out)
    assert actual is out
    _assert_product(a, b, actual)
    assert torch.all(storage[:, :, ::2] == 11).item()


@pytest.mark.parametrize(
    "batch,m,n,k,layout,strided_out,expected",
    [
        (1, 17, 33, 71, "NN", False, "mm"),
        (1, 17, 33, 71, "TT", False, "mm"),
        (1, 17, 33, 71, "slice", False, "batched"),
        (1, 17, 33, 71, "NN", True, "batched"),
        (1, 17, 1, 257, "slice", True, "mv"),
        (1, 1, 33, 257, "slice", True, "mv"),
        (3, 2, 129, 1024, "NN", False, "batched"),
        (3, 67, 1, 257, "NN", False, "batched"),
        (3, 1, 129, 257, "NT", False, "batched"),
    ],
)
def test_bmm_forwarding_preserves_storage(
    monkeypatch, batch, m, n, k, layout, strided_out, expected
):
    a, b = _inputs(batch, m, n, k, torch.float32, layout)
    out = torch.empty((batch, m, n * (2 if strided_out else 1)), device=a.device)
    if strided_out:
        out = out[:, :, 1::2]
    backend = importlib.import_module(flag_gems.bmm.__module__)
    calls = []
    storage_ptrs = {a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()}

    def wrap(original, kind):
        def checked(lhs, rhs, out):
            calls.append(kind)
            assert lhs.untyped_storage().data_ptr() in storage_ptrs
            assert rhs.untyped_storage().data_ptr() in storage_ptrs
            assert out.untyped_storage().data_ptr() == output_ptr
            return original(lhs, rhs, out=out)

        return checked

    output_ptr = out.untyped_storage().data_ptr()
    monkeypatch.setattr(
        backend, "_optimized_mm_out", wrap(backend._optimized_mm_out, "mm")
    )
    monkeypatch.setattr(backend, "_launch_gemv", wrap(backend._launch_gemv, "mv"))
    result = flag_gems.bmm(a, b, out=out)
    assert result is out
    assert calls == ([] if expected == "batched" else [expected])
    _assert_product(a, b, result)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,m,n", [(1, 17, 33), (1, 17, 1), (3, 17, 33)])
def test_bmm_float32_output(dtype, batch, m, n):
    a, b = _inputs(batch, m, n, 71, dtype)
    out = torch.empty((batch, m, n), device=a.device, dtype=torch.float32)
    result = flag_gems.bmm(a, b, out_dtype=torch.float32, out=out)
    assert result is out
    _assert_product(a, b, result)


@pytest.mark.parametrize(
    "batch,m,n,k", [(0, 17, 33, 71), (3, 0, 33, 71), (1, 17, 33, 0)]
)
def test_bmm_empty_dimensions(batch, m, n, k):
    a, b = _inputs(batch, m, n, k, torch.float32)
    result = flag_gems.bmm(a, b)
    assert result.shape == (batch, m, n)
    torch.testing.assert_close(result, torch.bmm(a, b))


@pytest.mark.parametrize(
    "batch,m,n,k,layout",
    [
        (1, 17, 33, 71, "NN"),
        (1, 17, 1, 257, "NN"),
        (1, 1, 33, 257, "NT"),
        (3, 17, 33, 71, "slice"),
        (3, 2, 129, 1024, "NN"),
    ],
)
def test_bmm_graph_updated_inputs(batch, m, n, k, layout):
    a, b = _inputs(batch, m, n, k, torch.float16, layout)
    for _ in range(3):
        flag_gems.bmm(a, b)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flag_gems.bmm(a, b)
    for _ in range(2):
        a.normal_()
        b.normal_()
        graph.replay()
        _assert_product(a, b, result)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bmm_large_batch_graph(dtype):
    # Exercise the migrated long-K batched compute (including dual-dot/partial
    # workspace candidates), which batch-one MM forwarding does not cover.
    a, b = _inputs(2, 1024, 1024, 4096, dtype)
    for _ in range(3):
        result = flag_gems.bmm(a, b)
    _assert_product(a, b, result)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flag_gems.bmm(a, b)
    a.normal_()
    b.normal_()
    graph.replay()
    _assert_product(a, b, result)
