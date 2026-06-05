import pytest
import torch

import flag_gems
from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.planning import (
    fa3_tle_select_plan,
)

from .hopper_fa3_utils import (
    Shape,
    build_reference,
    is_fa3_supported,
    make_varlen,
    max_mean_abs,
    output_tensor,
    run_flag_gems,
    tolerances,
)


def _skip_unless_hopper_fa3(pytestconfig) -> None:
    if pytestconfig.getoption("flash_attn_varlen_fa_version") != 3:
        pytest.skip("Hopper FA3 smoke only runs with fa_version=3.")
    if not is_fa3_supported():
        pytest.skip("requires CUDA Hopper with TLE FA3 support.")


_PAGED_SMOKE_SHAPES = [
    Shape(
        "paged_decode_q1_flashdecoding",
        [(1, 128), (1, 192), (1, 256)],
        8,
        2,
        128,
        True,
        paged=True,
        block_size=16,
    ),
    Shape(
        "paged_decodeish_q16_longk_flashdecoding",
        [(16, 1024)] + [(1, 512)] * 15,
        8,
        2,
        128,
        True,
        paged=True,
        block_size=16,
    ),
    Shape(
        "paged_short_blockwise",
        [(64, 64), (32, 96), (1, 128)],
        8,
        2,
        128,
        True,
        paged=True,
        block_size=16,
    ),
    Shape(
        "paged_benchmark_mixed_long",
        [(1, 1328), (5, 18), (129, 463)],
        8,
        2,
        128,
        True,
        paged=True,
        block_size=32,
    ),
]


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    "total_q,batch_size,max_q,max_k,is_paged,expected_family",
    [
        (8, 2, 4, 256, True, "flashdecoding"),
        (31, 16, 16, 1024, True, "flashdecoding"),
        (97, 3, 64, 128, True, "short"),
        (135, 3, 129, 1328, True, "long"),
    ],
    ids=[
        "q1-gqa-swapped-flashdecoding",
        "q16-longk-flashdecoding",
        "short-paged",
        "benchmark-mixed-long",
    ],
)
def test_fa3_paged_plan_smoke(
    total_q, batch_size, max_q, max_k, is_paged, expected_family
):
    plan = fa3_tle_select_plan(
        total_q=total_q,
        batch_size=batch_size,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        head_dim=128,
        is_paged=is_paged,
        force_family_id=-1,
        num_heads=8,
        decode_strategy="auto",
        small_strategy="auto",
        ws_strategy="auto",
        num_sms=132,
    )
    assert plan.family == expected_family


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("paged_gather", ["auto", "blockwise", "legacy"])
@pytest.mark.parametrize("shape", _PAGED_SMOKE_SHAPES, ids=lambda shape: shape.name)
@torch.inference_mode()
def test_fa3_paged_gather_correctness_smoke(
    monkeypatch, pytestconfig, paged_gather, shape
):
    _skip_unless_hopper_fa3(pytestconfig)
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_FORCE_PATH", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_SMALL_STRATEGY", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_WS_STRATEGY", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", paged_gather)

    tensors = make_varlen(shape, torch.float16, flag_gems.device, seed=2041)
    ref, ref_kind = build_reference(tensors, shape, fa_version=3)
    out = output_tensor(run_flag_gems(tensors, shape, fa_version=3))
    atol, rtol = tolerances(torch.float16, tensors.max_seqlen_k, ref_kind)
    max_abs, mean_abs = max_mean_abs(out, ref)
    msg = (
        f"shape={shape.name}, paged_gather={paged_gather}, ref={ref_kind}, "
        f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol, msg=msg)
