import pytest
import torch

import flag_gems

from flag_gems.runtime.backend._nvidia.hopper.ops.fa3_ws.best_known import (
    ROUTE_CURRENT_FA3,
    ROUTE_FA2_FALLBACK,
    classify_fa3_workload,
    fa3_tle_best_route_mode,
    select_fa3_best_route,
)


def test_fa3_best_route_env_validation(monkeypatch):
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "bad")
    with pytest.raises(RuntimeError, match="FLAG_GEMS_FA3_TLE_BEST_ROUTE"):
        fa3_tle_best_route_mode()


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            dict(
                total_q=8192,
                batch_size=4,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                is_paged=False,
            ),
            "dense_prefill_or_long",
        ),
        (
            dict(
                total_q=16,
                batch_size=16,
                max_seqlen_q=1,
                max_seqlen_k=1024,
                is_paged=True,
            ),
            "paged_decode",
        ),
        (
            dict(
                total_q=4096,
                batch_size=32,
                max_seqlen_q=2048,
                max_seqlen_k=4096,
                is_paged=True,
            ),
            "paged_serve_mixed",
        ),
        (
            dict(
                total_q=16384,
                batch_size=4,
                max_seqlen_q=4096,
                max_seqlen_k=4096,
                is_paged=True,
            ),
            "paged_uniform_prefill",
        ),
    ],
)
def test_fa3_best_workload_classification(kwargs, expected):
    assert classify_fa3_workload(**kwargs) == expected


@pytest.mark.parametrize(
    "kwargs,expected_route",
    [
        (
            dict(
                total_q=8192,
                batch_size=4,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                is_paged=False,
            ),
            ROUTE_CURRENT_FA3,
        ),
        (
            dict(
                total_q=16,
                batch_size=16,
                max_seqlen_q=1,
                max_seqlen_k=1024,
                is_paged=True,
            ),
            ROUTE_FA2_FALLBACK,
        ),
        (
            dict(
                total_q=4096,
                batch_size=32,
                max_seqlen_q=2048,
                max_seqlen_k=4096,
                is_paged=True,
            ),
            ROUTE_CURRENT_FA3,
        ),
        (
            dict(
                total_q=16384,
                batch_size=4,
                max_seqlen_q=4096,
                max_seqlen_k=4096,
                is_paged=True,
            ),
            ROUTE_FA2_FALLBACK,
        ),
    ],
)
def test_fa3_best_route_auto(kwargs, expected_route):
    route = select_fa3_best_route(force_family_id=-1, **kwargs)
    assert route.route == expected_route


def test_fa3_best_route_force_family_keeps_fa3():
    route = select_fa3_best_route(
        total_q=16,
        batch_size=16,
        max_seqlen_q=1,
        max_seqlen_k=1024,
        is_paged=True,
        force_family_id=5,
    )
    assert route.route == ROUTE_CURRENT_FA3


def test_fa3_best_route_env_overrides(monkeypatch):
    kwargs = dict(
        total_q=8192,
        batch_size=4,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        is_paged=False,
        force_family_id=-1,
    )
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "fa2_only")
    assert select_fa3_best_route(**kwargs).route == ROUTE_FA2_FALLBACK
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "fa3_only")
    assert select_fa3_best_route(**kwargs).route == ROUTE_CURRENT_FA3


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@torch.inference_mode()
def test_fa3_best_route_auto_fallback_matches_fa2(monkeypatch, pytestconfig):
    from .hopper_fa3_utils import (
        Shape,
        is_fa3_supported,
        make_varlen,
        max_mean_abs,
        output_tensor,
        run_flag_gems,
    )

    if pytestconfig.getoption("flash_attn_varlen_fa_version") != 3:
        pytest.skip("Hopper FA3 fallback coverage only runs with fa_version=3.")
    if not is_fa3_supported():
        pytest.skip("requires CUDA Hopper with TLE FA3 support.")

    shape = Shape(
        "best_route_paged_decode_fallback",
        [(1, 128), (1, 256), (1, 384)],
        8,
        2,
        128,
        True,
        paged=True,
        block_size=16,
    )
    tensors = make_varlen(shape, torch.float16, flag_gems.device, seed=2051)

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_FORCE_PATH", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_BEST_ROUTE", "auto")
    out_auto = output_tensor(run_flag_gems(tensors, shape, fa_version=3))

    out_fa2 = output_tensor(run_flag_gems(tensors, shape, fa_version=2))
    max_abs, mean_abs = max_mean_abs(out_auto, out_fa2)
    torch.testing.assert_close(
        out_auto.float(),
        out_fa2.float(),
        atol=0,
        rtol=0,
        msg=f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}",
    )
