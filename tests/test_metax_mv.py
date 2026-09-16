# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX MV view, output and CUDA Graph contracts, independent of BMM."""

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
