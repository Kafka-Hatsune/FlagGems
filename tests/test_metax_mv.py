# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX MV view, output and CUDA Graph contracts, independent of BMM."""

import importlib

import flag_gems
import pytest
import torch

from . import accuracy_utils as utils

pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX MV")
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _inputs(m, k, dtype, layout):
    torch.manual_seed(1729)
    if layout == "column":
        matrix = torch.randn((k, m), device=flag_gems.device, dtype=dtype).T
    elif layout == "slice":
        matrix = torch.randn((2 * m, 2 * k), device=flag_gems.device, dtype=dtype)[
            1::2, 1::2
        ]
    elif layout == "broadcast":
        matrix = torch.randn((1, k), device=flag_gems.device, dtype=dtype).expand(m, -1)
    else:
        matrix = torch.randn((m, k), device=flag_gems.device, dtype=dtype)
    vector = torch.randn((2 * k,), device=flag_gems.device, dtype=dtype)[1::2]
    return matrix, vector


def _check(matrix, vector, result):
    reference = torch.mv(
        utils.to_reference(matrix, True), utils.to_reference(vector, True)
    )
    utils.gems_assert_close(result, reference, result.dtype, reduce_dim=matrix.shape[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["row", "column", "slice", "broadcast"])
def test_mv_views_and_out(dtype, layout):
    matrix, vector = _inputs(65, 257, dtype, layout)
    storage = torch.full((130,), 11, device=matrix.device, dtype=dtype)
    out = storage[1::2]
    result = flag_gems.mv(matrix, vector, out=out)
    assert result is out
    _check(matrix, vector, result)
    assert torch.all(storage[::2] == 11).item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["row", "column"])
def test_mv_long_k(dtype, layout):
    matrix, vector = _inputs(33, 4096, dtype, layout)
    with flag_gems.use_gems():
        result = torch.mv(matrix, vector)
    _check(matrix, vector, result)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,k", [(0, 31), (17, 0)])
def test_mv_empty_dimensions(dtype, m, k):
    matrix, vector = _inputs(m, k, dtype, "row")
    result = flag_gems.mv(matrix, vector)
    assert result.shape == (m,)
    torch.testing.assert_close(result, torch.mv(matrix, vector))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["slice", "column"])
def test_mv_graph_updated_inputs(dtype, layout):
    matrix, vector = _inputs(65, 1024, dtype, layout)
    for _ in range(3):
        flag_gems.mv(matrix, vector)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flag_gems.mv(matrix, vector)
    for _ in range(2):
        matrix.normal_()
        vector.normal_()
        graph.replay()
        _check(matrix, vector, result)


@pytest.mark.parametrize(
    "invalid",
    [
        "matrix_rank",
        "vector_rank",
        "k",
        "input_dtype",
        "integer",
        "cpu",
        "out_shape",
        "out_dtype",
        "out_device",
        "out_overlap",
        "alias_matrix",
        "alias_vector",
        "grad",
        "negative_view",
    ],
)
def test_mv_invalid_calls_stop_before_dispatch(monkeypatch, invalid):
    matrix = torch.randn((8, 16), device=flag_gems.device)
    vector = torch.randn((16,), device=flag_gems.device)
    out = torch.full((8,), 17.0, device=flag_gems.device)
    expected = RuntimeError
    if invalid == "matrix_rank":
        matrix = matrix[0]
    elif invalid == "vector_rank":
        vector = vector[None, :]
    elif invalid == "k":
        vector = vector[:-1]
    elif invalid == "input_dtype":
        vector = vector.half()
    elif invalid == "integer":
        matrix, vector = matrix.int(), vector.int()
        expected = NotImplementedError
    elif invalid == "cpu":
        matrix, vector = matrix.cpu(), vector.cpu()
        expected = NotImplementedError
    elif invalid == "out_shape":
        out = out[:-1]
    elif invalid == "out_dtype":
        out = out.half()
    elif invalid == "out_device":
        out = out.cpu()
    elif invalid == "out_overlap":
        out = out[:1].expand(8)
        expected = NotImplementedError
    elif invalid == "alias_matrix":
        out = matrix[:, 0]
        expected = NotImplementedError
    elif invalid == "alias_vector":
        out = vector[:8]
        expected = NotImplementedError
    elif invalid == "grad":
        matrix.requires_grad_()
        expected = NotImplementedError
    elif invalid == "negative_view":
        vector = torch._neg_view(vector)
        expected = NotImplementedError
    before = out.detach().clone()
    backend = importlib.import_module(flag_gems.mv.__module__)

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("invalid MV reached dispatch")

    monkeypatch.setattr(backend, "_dispatch_mv", unexpected_dispatch)
    with pytest.raises(expected):
        flag_gems.mv(matrix, vector, out=out)
    torch.testing.assert_close(out, before)


@pytest.mark.parametrize("layout", ["row", "column"])
def test_mv_cold_graph_rejected_before_dispatch(monkeypatch, layout):
    matrix, vector = _inputs(39, 131, torch.float32, layout)
    backend = importlib.import_module(flag_gems.mv.__module__)
    monkeypatch.setattr(backend, "_MV_WARMED", set())

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("cold captured MV reached dispatch")

    monkeypatch.setattr(backend, "_dispatch_mv", unexpected_dispatch)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="warm MetaX mv"):
        with torch.cuda.graph(graph):
            flag_gems.mv(matrix, vector)
