# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

device = flag_gems.device
vendor_name = flag_gems.vendor_name
HOPPER_AVAILABLE = (
    vendor_name == "nvidia"
    and device == "cuda"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 9
)

FA_VERSION_CASES = [
    pytest.param(2, id="fa2"),
    pytest.param(
        3,
        id="fa3",
        marks=pytest.mark.skipif(
            not HOPPER_AVAILABLE,
            reason="FA3 requires an NVIDIA Hopper GPU",
        ),
    ),
]

if QUICK_MODE:
    NUM_HEADS = [(8, 2)]
    HEAD_SIZES = [128]
    FLOAT_DTYPES = [torch.float16]
    ALIBI = [False]
    SOFT_CAPS = [None]
    NUM_BLOCKS = [2048]
    OPTIMIZE_INIT = [False]
    SWAP_SOFT_CAPS = [None]
    NONCONTIG_DTYPES = [torch.float16]
    NONCONTIG_OPTIMIZE_INIT = [False]
else:
    NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
    HEAD_SIZES = [128, 192, 256]
    FLOAT_DTYPES = [torch.float16, torch.bfloat16]
    ALIBI = [False, True]
    SOFT_CAPS = [None, 10.0, 50.0]
    NUM_BLOCKS = [32768, 2048]
    OPTIMIZE_INIT = [False, True]
    SWAP_SOFT_CAPS = [None, 10.0]
    NONCONTIG_DTYPES = [torch.float16, torch.bfloat16]
    NONCONTIG_OPTIMIZE_INIT = [False, True]


def make_paged_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    non_contiguous: bool,
    device: str = device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (num_blocks, block_size, num_kv_heads, head_size)
    if not non_contiguous:
        key_cache = torch.randn(*shape, dtype=dtype, device=device)
        value_cache = torch.randn_like(key_cache)
        return key_cache, value_cache

    storage_shape = (num_blocks * 2, block_size, num_kv_heads, head_size)
    key_storage = torch.randn(*storage_shape, dtype=dtype, device=device)
    value_storage = torch.randn_like(key_storage)
    key_cache = key_storage[::2][:num_blocks]
    value_cache = value_storage[::2][:num_blocks]

    assert key_cache.shape == shape
    assert value_cache.shape == shape
    assert key_cache.stride() == value_cache.stride()
    assert key_cache.stride(-1) == 1
    assert key_cache.stride(0) != block_size * key_cache.stride(1)
    return key_cache, value_cache


# Following varlen and paged attn tests are copied from
# https://github.com/vllm-project/flash-attention/blob/main/tests/test_vllm_flash_attn.py
def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)

    if causal:
        v = torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32)
        return v * slopes

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)

    return -slopes * relative_pos.to(dtype=slopes.dtype)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor = None,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    s_aux: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    softmax_lses: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        if s_aux is None:
            q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        if s_aux is None:
            attn = torch.einsum("qhd,khd->hqk", q, k)
        else:
            attn = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))

        if attn_bias is not None:
            attn = attn + attn_bias[i, :, :query_len, :kv_len]

        if s_aux is not None:
            # The attention sink is a final, unscaled logit whose value vector is
            # zero. Drop its probability before the PV product.
            sink_logits = s_aux.float()[:, None, None].expand(-1, query_len, 1)
            attn = torch.cat((attn.float(), sink_logits), dim=-1)
            if return_softmax_lse:
                softmax_lses.append(torch.logsumexp(attn, dim=-1))
            attn = torch.softmax(attn, dim=-1)[..., :-1]
            out = torch.einsum("hqk,khd->qhd", attn, v.float()).to(v.dtype)
        else:
            if return_softmax_lse:
                softmax_lses.append(torch.logsumexp(attn.float(), dim=-1))
            attn = torch.softmax(attn, dim=-1).to(v.dtype)
            out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    output = torch.cat(outputs, dim=0)
    if return_softmax_lse:
        return output, torch.cat(softmax_lses, dim=1)
    return output


@pytest.mark.flash_attn_varlen_func
@pytest.mark.flash_attn_varlen_opt_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("alibi", ALIBI)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("optimize_init", OPTIMIZE_INIT)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@torch.inference_mode()
def test_flash_attn_varlen_func(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    alibi: bool,
    soft_cap: Optional[float],
    num_blocks: int,
    optimize_init: bool,
) -> None:
    # (Issue) numerical stability concern
    if alibi is True and soft_cap is not None:
        return

    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)

        if vendor_name == "cambricon":
            torch.manual_seed(123456)
            torch.mlu.manual_seed_all(123456)

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(
            sum(query_lens), num_query_heads, head_size, dtype=dtype, device=device
        )
        key_cache, value_cache = make_paged_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device=device,
            non_contiguous=False,
        )
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        causal = True

        if alibi:
            # alibi_slopes = torch.rand(num_seqs, num_query_heads, device=device, dtype=torch.float32) * 0.3
            alibi_slopes = (
                torch.ones(
                    num_seqs, num_query_heads, device=device, dtype=torch.float32
                )
                * 0.3
            )
            attn_bias = attn_bias_from_alibi_slopes(
                alibi_slopes, max_query_len, max_kv_len, causal=causal
            )
        else:
            alibi_slopes, attn_bias = None, None

        if vendor_name in ["cambricon", "sunrise"]:
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=causal,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                alibi_slopes=alibi_slopes,
                fa_version=2,
            )
        else:
            if optimize_init:
                output = flag_gems.flash_attn_varlen_opt_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )
            else:
                output = flag_gems.flash_attn_varlen_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            attn_bias=attn_bias,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        msg = f"{torch.max(torch.abs(output - ref_output))}"
        if vendor_name == "sunrise":
            torch.testing.assert_close(
                output, ref_output, atol=3e-2, rtol=1e-2, msg=msg
            )
        else:
            torch.testing.assert_close(
                output, ref_output, atol=2e-2, rtol=1e-2, msg=msg
            )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.parametrize("dtype", NONCONTIG_DTYPES)
@pytest.mark.parametrize("optimize_init", NONCONTIG_OPTIMIZE_INIT)
@torch.inference_mode()
def test_flash_attn_varlen_func_noncontiguous_kv_cache(
    monkeypatch,
    dtype: torch.dtype,
    optimize_init: bool,
) -> None:
    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)

        seq_lens = [(1, 1328), (5, 18), (129, 463)]
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_seqs = len(seq_lens)
        num_query_heads = 8
        num_kv_heads = 2
        head_size = 128
        block_size = 32
        num_blocks = 2048
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (-1, -1)
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache, value_cache = make_paged_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            non_contiguous=True,
            device=device,
        )
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        op = (
            flag_gems.flash_attn_varlen_opt_func
            if optimize_init
            else flag_gems.flash_attn_varlen_func
        )
        output = op(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=0,
            fa_version=2,
        )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
        )

        max_diff = torch.max(torch.abs(output - ref_output))
        msg = f"max_diff={max_diff}, k_stride={key_cache.stride()}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not working")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (1, 18), (1, 463)]])
@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("soft_cap", SWAP_SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", [2048])
@pytest.mark.parametrize("fa_version", FA_VERSION_CASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@torch.inference_mode()
def test_flash_attn_varlen_func_swap_qg(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    fa_version: int,
) -> None:
    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)

        if vendor_name == "cambricon":
            torch.manual_seed(123456)
            torch.mlu.manual_seed_all(123456)

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(
            sum(query_lens), num_query_heads, head_size, dtype=dtype, device=device
        )
        key_cache, value_cache = make_paged_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device=device,
            non_contiguous=False,
        )
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        if vendor_name in ["cambricon", "sunrise"]:
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=fa_version,
            )
        else:
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=fa_version,
            )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        torch.testing.assert_close(
            output, ref_output, atol=2e-2, rtol=1e-2
        ), f"{torch.max(torch.abs(output - ref_output))}"


def _make_fa3_s_aux_case(query_len: int, kv_len: int):
    num_query_heads = 8
    num_kv_heads = 2
    head_size = 128
    block_size = 16
    num_kv_blocks = (kv_len + block_size - 1) // block_size
    dtype = torch.bfloat16
    scale = head_size**-0.5

    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    key_cache, value_cache = make_paged_kv_cache(
        num_kv_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        non_contiguous=False,
        device=device,
    )
    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=device)
    block_tables = torch.arange(
        num_kv_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)
    s_aux = torch.tensor(
        [float("-inf"), -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        dtype=torch.bfloat16,
        device=device,
    )
    return (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    )


def _make_fa3_wide_split_case(
    query_lens: List[int],
    kv_len: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    num_query_heads: int = 4,
    num_kv_heads: int = 1,
    block_size: int = 16,
    max_seqlen_k: Optional[int] = None,
):
    head_size = 256
    num_kv_blocks = (kv_len + block_size - 1) // block_size
    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    key_cache, value_cache = make_paged_kv_cache(
        num_kv_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        non_contiguous=False,
        device=device,
    )
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    seqused_k = torch.full(
        (len(query_lens),), kv_len, dtype=torch.int32, device=device
    )
    block_table_row = torch.arange(
        num_kv_blocks, dtype=torch.int32, device=device
    )
    block_table_width = (
        (max_seqlen_k + block_size - 1) // block_size
        if max_seqlen_k is not None
        else num_kv_blocks
    )
    if block_table_width > num_kv_blocks:
        block_table_row = torch.nn.functional.pad(
            block_table_row,
            (0, block_table_width - num_kv_blocks),
        )
    block_tables = block_table_row.expand(len(query_lens), -1)
    return (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        head_size**-0.5,
    )


def _fa3_scheduling_module():
    hopper_package = flag_gems.flash_attn_varlen_func.__module__.split(".ops.", 1)[0]
    return import_module(f"{hopper_package}.ops.attention_impl.scheduling")


_FA3_AUTOTUNE_EXPERIMENT_ENV = (
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_MMA_GROUPS",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_N",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_KV_BUFFERS",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_DECODE_TMA_QO",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_TMA_QO",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_Q_BUFFERS",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_WARP_MMA",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_Q_PIPE_ASYNC",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_REUSE_Q_SMEM_O",
    "FLAG_GEMS_FA3_TLE_EXPERIMENT_PIPE_ASYNC",
)


def _capture_fa3_plans(monkeypatch):
    """Observe scheduler results in tests without production instrumentation."""

    FA3Scheduler = _fa3_scheduling_module().FA3Scheduler
    original_build = FA3Scheduler.build
    plans = []

    def build_and_capture(cls, inputs, config=None):
        plan = original_build(inputs, config)
        plans.append(plan)
        return plan

    monkeypatch.setattr(FA3Scheduler, "build", classmethod(build_and_capture))
    return FA3Scheduler, plans


def _plan_signature(plan) -> Tuple[str, int]:
    num_splits = plan.persistent_num_splits if plan.persistent_split_kv else 1
    return plan.kernel_name, num_splits


@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    (
        "max_seqlen_q,head_dim,total_q,batch_size,explicit_split_k_chunk,"
        "expected_block_m,expected_compact"
    ),
    [
        pytest.param(64, 256, 67, 4, 1536, 8, True, id="d256-mixed"),
        pytest.param(
            13,
            256,
            13,
            1,
            1024,
            8,
            False,
            id="d256-short-single",
        ),
        pytest.param(
            13,
            256,
            13,
            1,
            1536,
            16,
            False,
            id="d256-short-single-default-chunk",
        ),
        pytest.param(
            13,
            256,
            1,
            1,
            1024,
            16,
            False,
            id="d256-padded-single-negative",
        ),
        pytest.param(
            1,
            256,
            4,
            4,
            1536,
            1,
            False,
            id="d256-uniform-decode-negative",
        ),
        pytest.param(64, 256, 64, 1, 1536, 64, False, id="d256-single"),
        pytest.param(64, 128, 67, 4, 1536, 64, False, id="d128-mixed"),
    ],
)
def test_flash_attn_varlen_fa3_split_combine_launch_plan(
    max_seqlen_q: int,
    head_dim: int,
    total_q: int,
    batch_size: int,
    explicit_split_k_chunk: int,
    expected_block_m: int,
    expected_compact: bool,
) -> None:
    plan = _fa3_scheduling_module().PersistentSchedulingHeuristics.combine_launch_plan(
        max_seqlen_q=max_seqlen_q,
        head_dim=head_dim,
        total_q=total_q,
        batch_size=batch_size,
        explicit_split_k_chunk=explicit_split_k_chunk,
    )

    assert plan.block_m == expected_block_m
    assert plan.compact_ragged is expected_compact


@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    "num_heads,num_heads_k,block_size,expected_block_m",
    [
        pytest.param(16, 2, 32, 1, id="gqa8-page32-measured-profile"),
        pytest.param(4, 1, 16, 8, id="gqa4-negative-control"),
    ],
)
def test_flash_attn_varlen_fa3_tiny_gqa8_combine_block_m(
    num_heads: int,
    num_heads_k: int,
    block_size: int,
    expected_block_m: int,
) -> None:
    plan = _fa3_scheduling_module().PersistentSchedulingHeuristics.combine_launch_plan(
        max_seqlen_q=64,
        head_dim=256,
        total_q=67,
        batch_size=4,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        block_size=block_size,
    )

    assert plan.block_m == expected_block_m
    assert plan.compact_ragged


@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_persistent_autotune_key_buckets() -> None:
    persistent = import_module(
        f"{flag_gems.flash_attn_varlen_func.__module__.split('.ops.', 1)[0]}"
        ".ops.attention_impl.persistent"
    )
    tuner = persistent.flash_varlen_fwd_v3_tle_kernel.fn
    strategies = dict(zip(tuner.keys, tuner.strategy))

    assert strategies["b"](5) == 5
    assert strategies["seqlen_q"](513) == 544
    assert strategies["seqlen_k"](4107) == 4128
    assert strategies["total_q"](16384) == 16384
    assert {
        "is_seqused_k",
        "window_size_left",
        "window_size_right",
        "is_softcap",
        "EXPLICIT_SPLIT_K_CHUNK",
        "PAGED_PREFILL_PROFILE",
        "DENSE_KV_TMA_PROFILE",
        "AUTOTUNE_POLICY_VERSION",
    } <= set(tuner.keys)
    assert (
        _fa3_scheduling_module().PersistentSchedulingHeuristics.AUTOTUNE_POLICY_VERSION
        == 13
    )


@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_dense_tma_profile_partitions_aligned_key() -> None:
    scheduling = _fa3_scheduling_module()
    heuristics = scheduling.PersistentSchedulingHeuristics

    common = dict(
        batch_size=12,
        num_heads=32,
        num_heads_k=32,
        head_dim=64,
        max_seqlen_k=128,
        is_paged=False,
        is_causal=False,
        is_local=False,
        is_alibi=False,
        is_softcap=False,
        pack_gqa=False,
        ragged_scheduler=False,
        split_kv=False,
    )
    q255_tma = heuristics.use_dense_kv_tma(
        **common,
        max_seqlen_q=255,
        total_q=3060,
    )
    q256_tma = heuristics.use_dense_kv_tma(
        **common,
        max_seqlen_q=256,
        total_q=3072,
    )

    assert q255_tma is True
    assert q256_tma is False
    # The aligned raw lengths collide, so the derived profile must remain an
    # exact autotune-key component.
    persistent = import_module(
        f"{flag_gems.flash_attn_varlen_func.__module__.split('.ops.', 1)[0]}"
        ".ops.attention_impl.persistent"
    )
    tuner = persistent.flash_varlen_fwd_v3_tle_kernel.fn
    strategies = dict(
        zip(tuner.keys, tuner.strategy)
    )
    assert strategies["seqlen_q"](255) == strategies["seqlen_q"](256)
    assert strategies["total_q"](3060) == strategies["total_q"](3072)
    assert "DENSE_KV_TMA_PROFILE" in strategies


@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    "base_work,num_sms,expected_splits",
    [
        pytest.param(8, 114, 16, id="h100-full-wave"),
        pytest.param(8, 132, 16, id="h800-near-full-wave"),
        pytest.param(8, 144, 32, id="below-ninety-percent"),
        pytest.param(103, 114, 1, id="unsplit-near-full-wave"),
        pytest.param(102, 114, 2, id="unsplit-below-ninety-percent"),
    ],
)
def test_flash_attn_varlen_fa3_adaptive_split_wave_target(
    base_work: int,
    num_sms: int,
    expected_splits: int,
) -> None:
    scheduler = _fa3_scheduling_module().FA3Scheduler

    assert scheduler._splits_for_target_wave(base_work, num_sms) == expected_splits


@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    "max_seqlen_q,forced_block_m,expected_splits",
    [
        pytest.param(16, 0, 16, id="uniform-upper-bound-regression"),
        pytest.param(17, 0, 8, id="production-bm64"),
        pytest.param(17, 128, 16, id="forced-bm128"),
    ],
)
def test_flash_attn_varlen_fa3_uniform_short_query_uses_exact_split_work(
    monkeypatch,
    max_seqlen_q: int,
    forced_block_m: int,
    expected_splits: int,
) -> None:
    scheduling = _fa3_scheduling_module()
    scheduler = scheduling.FA3Scheduler

    monkeypatch.setenv(
        "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M",
        str(forced_block_m),
    )
    scheduler.clear_config_cache()
    try:
        plan = scheduler.build(
            SimpleNamespace(
                q=SimpleNamespace(dtype=torch.float16, element_size=lambda: 2),
                batch_size=8,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=65536,
                total_q=8 * max_seqlen_q,
                num_heads=4,
                num_heads_k=1,
                head_dim=256,
                has_cache_kv=True,
                is_paged=True,
                block_size=16,
                qo_tma_aligned=True,
                kv_tma_aligned=True,
                arch=90,
                num_sms=132,
                alibi_slopes=None,
                window=SimpleNamespace(causal=True, local=False),
                is_softcap=False,
                seqused_k=object(),
                max_num_splits=32,
            )
        )
    finally:
        scheduler.clear_config_cache()

    assert plan.kernel_name == f"persistent_splitkv_s{expected_splits}"
    assert plan.persistent_num_splits == expected_splits


@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    (
        "batch_size,max_seqlen_q,total_q,num_heads,num_heads_k,block_size,"
        "num_sms,gather_mode,expected_splits,expected_gather"
    ),
    [
        pytest.param(
            4,
            64,
            67,
            16,
            2,
            32,
            114,
            "auto",
            5,
            "BLOCKWISE",
            id="tp1-b4-one-long",
        ),
        pytest.param(
            8,
            132,
            139,
            16,
            2,
            32,
            114,
            "auto",
            4,
            "BLOCKWISE",
            id="tp1-b8-fallback",
        ),
        pytest.param(
            4,
            64,
            67,
            16,
            2,
            32,
            114,
            "legacy",
            5,
            "LEGACY",
            id="tp1-b4-explicit-legacy",
        ),
        pytest.param(
            4,
            64,
            68,
            16,
            2,
            32,
            114,
            "auto",
            8,
            "LEGACY",
            id="tp1-non-one-long-negative",
        ),
        pytest.param(
            8,
            121,
            128,
            4,
            1,
            16,
            114,
            "auto",
            7,
            "BLOCKWISE",
            id="tp4-b8-one-long",
        ),
        pytest.param(
            4,
            64,
            67,
            4,
            1,
            16,
            132,
            "auto",
            18,
            "BLOCKWISE",
            id="tp4-h800-one-long",
        ),
    ],
)
def test_flash_attn_varlen_fa3_single_long_ragged_split_work(
    monkeypatch,
    batch_size: int,
    max_seqlen_q: int,
    total_q: int,
    num_heads: int,
    num_heads_k: int,
    block_size: int,
    num_sms: int,
    gather_mode: str,
    expected_splits: int,
    expected_gather: str,
) -> None:
    scheduling = _fa3_scheduling_module()
    scheduler = scheduling.FA3Scheduler

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M", "0")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT", "0")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", gather_mode)
    scheduler.clear_config_cache()
    try:
        plan = scheduler.build(
            SimpleNamespace(
                q=SimpleNamespace(dtype=torch.float16, element_size=lambda: 2),
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=65560,
                total_q=total_q,
                num_heads=num_heads,
                num_heads_k=num_heads_k,
                head_dim=256,
                has_cache_kv=True,
                is_paged=True,
                block_size=block_size,
                qo_tma_aligned=True,
                kv_tma_aligned=True,
                arch=90,
                num_sms=num_sms,
                alibi_slopes=None,
                window=SimpleNamespace(causal=True, local=False),
                is_softcap=False,
                seqused_k=object(),
                max_num_splits=32,
            )
        )
    finally:
        scheduler.clear_config_cache()

    assert plan.kernel_name == f"persistent_splitkv_s{expected_splits}"
    assert plan.persistent_num_splits == expected_splits
    assert plan.paged_gather_mode == int(
        getattr(scheduling.PagedGatherMode, expected_gather)
    )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@torch.inference_mode()
def test_flash_attn_varlen_fa3_page32_one_long_blockwise_gather(
    monkeypatch,
) -> None:
    scheduling = _fa3_scheduling_module()
    FA3Scheduler, plans = _capture_fa3_plans(monkeypatch)
    query_lens = [1, 1, 1, 64]
    kv_len = 4096

    utils.init_seed(1234567890)
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        scale,
    ) = _make_fa3_wide_split_case(
        query_lens,
        kv_len,
        torch.float16,
        num_query_heads=16,
        num_kv_heads=2,
        block_size=32,
    )
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=[kv_len] * len(query_lens),
        block_tables=block_tables,
        scale=scale,
    )

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT", "0")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", "auto")
    FA3Scheduler.clear_config_cache()

    try:
        blockwise_output = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            num_splits=32,
            fa_version=3,
        )

        monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", "legacy")
        FA3Scheduler.clear_config_cache()
        legacy_output = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            num_splits=32,
            fa_version=3,
        )
    finally:
        FA3Scheduler.clear_config_cache()

    assert len(plans) == 2
    blockwise_plan, legacy_plan = plans
    assert blockwise_plan.persistent_split_kv
    assert blockwise_plan.paged_gather_mode == int(
        scheduling.PagedGatherMode.BLOCKWISE
    )
    assert legacy_plan.persistent_split_kv
    assert legacy_plan.paged_gather_mode == int(scheduling.PagedGatherMode.LEGACY)
    torch.testing.assert_close(
        blockwise_output,
        legacy_output,
        atol=2e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        blockwise_output,
        ref_output,
        atol=2e-2,
        rtol=1e-2,
    )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    "num_heads,num_heads_k,block_size",
    [
        pytest.param(4, 1, 16, id="page16-gqa4"),
        pytest.param(16, 2, 32, id="page32-gqa8"),
    ],
)
def test_flash_attn_varlen_fa3_pack_off_disables_packed_prefill_profile(
    monkeypatch,
    num_heads: int,
    num_heads_k: int,
    block_size: int,
) -> None:
    scheduling = _fa3_scheduling_module()
    scheduler = scheduling.FA3Scheduler

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_HEADS_IN_L2", "auto")
    scheduler.clear_config_cache()
    try:
        plan = scheduler.build(
            SimpleNamespace(
                q=SimpleNamespace(dtype=torch.float16, element_size=lambda: 2),
                batch_size=1,
                max_seqlen_q=1024,
                max_seqlen_k=1024,
                total_q=1024,
                num_heads=num_heads,
                num_heads_k=num_heads_k,
                head_dim=256,
                has_cache_kv=True,
                is_paged=True,
                block_size=block_size,
                qo_tma_aligned=True,
                kv_tma_aligned=True,
                arch=90,
                num_sms=132,
                alibi_slopes=None,
                window=SimpleNamespace(causal=True, local=False),
                is_softcap=False,
                seqused_k=object(),
                max_num_splits=0,
            )
        )
    finally:
        scheduler.clear_config_cache()

    assert plan.kernel_name == "long_paged_prefill"
    assert plan.pack_gqa is False
    assert plan.paged_d256_prefill_profile is False
    assert plan.paged_gather_mode == int(scheduling.PagedGatherMode.AUTO)
    assert plan.heads_in_l2.mode is scheduling.HeadsInL2Mode.EXPLICIT
    assert plan.heads_in_l2.value == 1


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@torch.inference_mode()
def test_flash_attn_varlen_fa3_adaptive_wide_mixed_split(monkeypatch) -> None:
    FA3Scheduler, plans = _capture_fa3_plans(monkeypatch)
    query_lens = [1, 1, 1, 64]
    kv_len = 4096

    utils.init_seed(1234567890)
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        scale,
    ) = _make_fa3_wide_split_case(query_lens, kv_len)
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=[kv_len] * len(query_lens),
        block_tables=block_tables,
        scale=scale,
    )

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_WIDE_PACK_GQA", "0")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT", "0")
    FA3Scheduler.clear_config_cache()

    try:
        output = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            num_splits=32,
            fa_version=3,
        )
        flag_gems.flash_attn_varlen_func(
            q=query[:1],
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens[:2],
            seqused_k=seqused_k[:1],
            max_seqlen_q=1,
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables[:1],
            num_splits=32,
            fa_version=3,
        )
        (
            decode_query,
            decode_key_cache,
            decode_value_cache,
            decode_cu_query_lens,
            decode_seqused_k,
            decode_block_tables,
            decode_scale,
        ) = _make_fa3_wide_split_case(
            [1] * 16,
            1,
            torch.float16,
            num_query_heads=16,
            num_kv_heads=2,
            block_size=32,
            max_seqlen_k=73728,
        )
        flag_gems.flash_attn_varlen_func(
            q=decode_query,
            k=decode_key_cache,
            v=decode_value_cache,
            cu_seqlens_q=decode_cu_query_lens,
            seqused_k=decode_seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=73728,
            softmax_scale=decode_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=decode_block_tables,
            num_splits=32,
            fa_version=3,
        )
        (
            tp4_decode_query,
            tp4_decode_key_cache,
            tp4_decode_value_cache,
            tp4_decode_cu_query_lens,
            tp4_decode_seqused_k,
            tp4_decode_block_tables,
            tp4_decode_scale,
        ) = _make_fa3_wide_split_case(
            [1] * 112,
            1,
            torch.float16,
            block_size=16,
            max_seqlen_k=73728,
        )
        tp4_decode_output = flag_gems.flash_attn_varlen_func(
            q=tp4_decode_query,
            k=tp4_decode_key_cache,
            v=tp4_decode_value_cache,
            cu_seqlens_q=tp4_decode_cu_query_lens,
            seqused_k=tp4_decode_seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=73728,
            softmax_scale=tp4_decode_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=tp4_decode_block_tables,
            num_splits=32,
            fa_version=3,
        )
        tp4_unsplit_output = flag_gems.flash_attn_varlen_func(
            q=tp4_decode_query,
            k=tp4_decode_key_cache,
            v=tp4_decode_value_cache,
            cu_seqlens_q=tp4_decode_cu_query_lens,
            seqused_k=tp4_decode_seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=73728,
            softmax_scale=tp4_decode_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=tp4_decode_block_tables,
            num_splits=1,
            fa_version=3,
        )
        (
            short_prefill_query,
            short_prefill_key_cache,
            short_prefill_value_cache,
            short_prefill_cu_query_lens,
            short_prefill_seqused_k,
            short_prefill_block_tables,
            short_prefill_scale,
        ) = _make_fa3_wide_split_case(
            [12],
            32780,
            torch.float16,
        )
        short_prefill_output = flag_gems.flash_attn_varlen_func(
            q=short_prefill_query,
            k=short_prefill_key_cache,
            v=short_prefill_value_cache,
            cu_seqlens_q=short_prefill_cu_query_lens,
            seqused_k=short_prefill_seqused_k,
            max_seqlen_q=12,
            max_seqlen_k=32780,
            softmax_scale=short_prefill_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=short_prefill_block_tables,
            num_splits=32,
            fa_version=3,
        )
        short_prefill_unsplit_output = flag_gems.flash_attn_varlen_func(
            q=short_prefill_query,
            k=short_prefill_key_cache,
            v=short_prefill_value_cache,
            cu_seqlens_q=short_prefill_cu_query_lens,
            seqused_k=short_prefill_seqused_k,
            max_seqlen_q=12,
            max_seqlen_k=32780,
            softmax_scale=short_prefill_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=short_prefill_block_tables,
            num_splits=1,
            fa_version=3,
        )
        (
            padded_decode_query,
            padded_decode_key_cache,
            padded_decode_value_cache,
            padded_decode_cu_query_lens,
            padded_decode_seqused_k,
            padded_decode_block_tables,
            padded_decode_scale,
        ) = _make_fa3_wide_split_case(
            [1],
            32780,
            torch.float16,
            max_seqlen_k=73728,
        )
        padded_decode_output = flag_gems.flash_attn_varlen_func(
            q=padded_decode_query,
            k=padded_decode_key_cache,
            v=padded_decode_value_cache,
            cu_seqlens_q=padded_decode_cu_query_lens,
            seqused_k=padded_decode_seqused_k,
            max_seqlen_q=12,
            max_seqlen_k=73728,
            softmax_scale=padded_decode_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=padded_decode_block_tables,
            num_splits=32,
            fa_version=3,
        )
    finally:
        FA3Scheduler.clear_config_cache()

    assert [_plan_signature(plan) for plan in plans] == [
        ("persistent_splitkv_s3", 3),
        ("persistent_splitkv_s32", 32),
        ("direct_packed_gqa", 1),
        ("direct_packed_gqa", 1),
        ("direct_packed_gqa", 1),
        ("persistent_splitkv_s32", 32),
        ("direct", 1),
        ("persistent_splitkv_s32", 32),
    ]
    assert plans[3] == plans[4]
    assert plans[5].explicit_split_k_chunk == 8 * 128
    assert plans[6].explicit_split_k_chunk == 12 * 128
    assert plans[7].explicit_split_k_chunk == 12 * 128
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)
    assert torch.isfinite(tp4_decode_output).all()
    torch.testing.assert_close(
        tp4_decode_output,
        tp4_unsplit_output,
        atol=2e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        short_prefill_output,
        short_prefill_unsplit_output,
        atol=2e-2,
        rtol=1e-2,
    )
    assert torch.isfinite(padded_decode_output).all()


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@torch.inference_mode()
def test_flash_attn_varlen_fa3_h800_near_full_uniform_decode_uses_direct(
    monkeypatch,
) -> None:
    """A 128/132-SM direct wave must not manufacture Split-KV work."""

    scheduling = _fa3_scheduling_module()
    FA3Scheduler = scheduling.FA3Scheduler
    original_build = FA3Scheduler.build
    plans = []

    def build_for_h800(cls, inputs, config=None):
        plan = original_build(replace(inputs, num_sms=132), config)
        plans.append(plan)
        return plan

    monkeypatch.setattr(FA3Scheduler, "build", classmethod(build_for_h800))
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_WIDE_PACK_GQA", "0")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT", "0")
    FA3Scheduler.clear_config_cache()

    utils.init_seed(1234567890)
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        scale,
    ) = _make_fa3_wide_split_case(
        [1] * 128,
        1,
        torch.float16,
        block_size=16,
        max_seqlen_k=73728,
    )
    try:
        output = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=73728,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            num_splits=32,
            fa_version=3,
        )
        (
            below_threshold_query,
            below_threshold_key_cache,
            below_threshold_value_cache,
            below_threshold_cu_query_lens,
            below_threshold_seqused_k,
            below_threshold_block_tables,
            below_threshold_scale,
        ) = _make_fa3_wide_split_case(
            [1] * 118,
            1,
            torch.float16,
            block_size=16,
            max_seqlen_k=73728,
        )
        below_threshold_output = flag_gems.flash_attn_varlen_func(
            q=below_threshold_query,
            k=below_threshold_key_cache,
            v=below_threshold_value_cache,
            cu_seqlens_q=below_threshold_cu_query_lens,
            seqused_k=below_threshold_seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=73728,
            softmax_scale=below_threshold_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=below_threshold_block_tables,
            num_splits=32,
            fa_version=3,
        )
    finally:
        FA3Scheduler.clear_config_cache()

    assert [_plan_signature(plan) for plan in plans] == [
        ("direct_packed_gqa", 1),
        ("persistent_splitkv_s32", 32),
    ]
    direct_plan, split_plan = plans
    assert direct_plan.dynamic_scheduler is False
    assert direct_plan.ragged_scheduler is False
    assert direct_plan.paged_gather_mode == int(scheduling.PagedGatherMode.AUTO)
    assert direct_plan.requires_tma_alignment is False
    assert split_plan.dynamic_scheduler is True
    assert split_plan.paged_gather_mode == int(scheduling.PagedGatherMode.BLOCKWISE)
    assert split_plan.requires_tma_alignment is False
    assert torch.isfinite(output).all()
    assert torch.isfinite(below_threshold_output).all()


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@pytest.mark.parametrize(
    (
        "dtype,query_lens,kv_len,expected_kernel,expected_pack,"
        "expected_gather,expected_l2_mode,expected_workload"
    ),
    [
        pytest.param(
            torch.float16,
            [1024],
            1024,
            "long_paged_prefill",
            True,
            "BLOCKWISE",
            "L2_AUTO",
            "prefill",
            id="fp16-measured-profile",
        ),
        pytest.param(
            torch.float16,
            [1] * 7 + [512],
            1040,
            "long_paged_prefill_ragged",
            True,
            "BLOCKWISE",
            "L2_AUTO",
            "serving_prefill",
            id="fp16-serving-prefill-profile",
        ),
        pytest.param(
            torch.bfloat16,
            [1024],
            1024,
            "direct",
            False,
            "AUTO",
            "EXPLICIT",
            "prefill",
            id="bf16-negative-control",
        ),
    ],
)
@torch.inference_mode()
def test_flash_attn_varlen_fa3_d256_prefill_cost_model(
    monkeypatch,
    dtype: torch.dtype,
    query_lens: List[int],
    kv_len: int,
    expected_kernel: str,
    expected_pack: bool,
    expected_gather: str,
    expected_l2_mode: str,
    expected_workload: str,
) -> None:
    scheduling = _fa3_scheduling_module()
    FA3Scheduler, plans = _capture_fa3_plans(monkeypatch)
    utils.init_seed(1234567890)
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        scale,
    ) = _make_fa3_wide_split_case(query_lens, kv_len, dtype)
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=[kv_len] * len(query_lens),
        block_tables=block_tables,
        scale=scale,
    )

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_ROUTE", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_Q", "1024")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_AVG_Q", "128")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_GATHER", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_HEADS_IN_L2", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "auto")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_WIDE_PACK_GQA", "0")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT", "0")
    FA3Scheduler.clear_config_cache()

    try:
        output = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            num_splits=0,
            fa_version=3,
        )
    finally:
        FA3Scheduler.clear_config_cache()

    assert len(plans) == 1
    plan = plans[0]
    assert plan.kernel_name == expected_kernel
    assert plan.workload == expected_workload
    assert plan.pack_gqa is expected_pack
    assert plan.pack_factor == (4 if expected_pack else 1)
    assert plan.paged_kv_non_tma
    assert plan.paged_gather_mode == int(
        getattr(scheduling.PagedGatherMode, expected_gather)
    )
    assert plan.heads_in_l2.mode is getattr(
        scheduling.HeadsInL2Mode, expected_l2_mode
    )
    assert not plan.persistent_split_kv
    assert plan.persistent_num_splits == 0
    assert plan.paged_d256_prefill_profile is (dtype == torch.float16)
    assert (
        "kernel_profile=h100_paged_d256_prefill" in plan.reason
    ) is (dtype == torch.float16)
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    "seqlen_q,min_prefill_q",
    [
        pytest.param(1024, None, id="default-threshold"),
        pytest.param(512, 512, id="lowered-threshold"),
    ],
)
def test_flash_attn_varlen_fa3_d256_prefill_autotune_candidates(
    monkeypatch,
    seqlen_q: int,
    min_prefill_q: Optional[int],
) -> None:
    scheduling = _fa3_scheduling_module()
    heuristics = scheduling.PersistentSchedulingHeuristics
    for name in _FA3_AUTOTUNE_EXPERIMENT_ENV + (
        "FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_AVG_Q",
    ):
        monkeypatch.delenv(name, raising=False)
    if min_prefill_q is None:
        monkeypatch.delenv("FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_Q", raising=False)
    else:
        monkeypatch.setenv(
            "FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_Q",
            str(min_prefill_q),
        )
    scheduling.FA3Scheduler.clear_config_cache()

    try:
        configs = heuristics.prune_autotune_configs(
            heuristics.autotune_configs(),
            {},
            d=256,
            b=1,
            h=4,
            hk=1,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_q,
            total_q=seqlen_q,
            is_paged=True,
            is_seqused_k=True,
            block_size=16,
            is_causal=True,
            is_local=False,
            is_alibi=False,
            is_softcap=False,
            PAGED_KV_NON_TMA=True,
            PACK_GQA=True,
            RAGGED_SCHEDULER=False,
            SPLIT_KV=False,
            PAGED_PREFILL_PROFILE=True,
        )
    finally:
        scheduling.FA3Scheduler.clear_config_cache()
    signatures = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["NUM_MMA_GROUPS"],
            config.kwargs["USE_TMA_QO"],
            config.kwargs["STAGGER_KV"],
            config.kwargs["ACTIVE_WGMMA_N"],
            config.kwargs.get("RESCALE_O_BEFORE_PV", False),
            config.kwargs.get("EARLY_CAST_P", False),
        )
        for config in configs
    }

    assert signatures == {
        (128, 64, 2, True, False, 64, False, False),
        (128, 64, 2, True, True, 64, False, False),
        (128, 128, 2, True, True, 80, False, True),
        (128, 128, 2, True, True, 80, True, True),
        (64, 64, 1, False, False, 64, False, False),
    }


@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_active_wgmma_n_heuristic(
    monkeypatch,
) -> None:
    scheduling = _fa3_scheduling_module()
    heuristics = scheduling.PersistentSchedulingHeuristics
    for name in _FA3_AUTOTUNE_EXPERIMENT_ENV:
        monkeypatch.delenv(name, raising=False)

    assert heuristics.active_wgmma_n_candidates(64, tiled_extent_eligible=False) == (
        128,
        64,
    )
    assert heuristics.active_wgmma_n_candidates(128, tiled_extent_eligible=False) == (
        128,
        64,
    )
    assert heuristics.active_wgmma_n_candidates(192, tiled_extent_eligible=False) == (
        64,
    )
    assert heuristics.active_wgmma_n_candidates(256, tiled_extent_eligible=True) == (
        80,
        64,
    )
    assert heuristics.active_wgmma_n_candidates(256, tiled_extent_eligible=False) == (
        64,
    )

    common_nargs = {
        "b": 1,
        "h": 8,
        "hk": 8,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "total_q": 1024,
        "is_paged": False,
        "is_seqused_k": False,
        "block_size": 1,
        "is_causal": True,
        "is_local": False,
        "is_alibi": False,
        "is_softcap": False,
        "PAGED_KV_NON_TMA": False,
        "PACK_GQA": False,
        "RAGGED_SCHEDULER": False,
        "SPLIT_KV": False,
        "PAGED_PREFILL_PROFILE": False,
        "DENSE_KV_TMA_PROFILE": True,
    }
    for head_dim, expected_active_n in (
        (64, {64, 128}),
        (128, {64, 128}),
        (192, {64}),
        (256, {64}),
    ):
        kept = heuristics.prune_autotune_configs(
            heuristics.autotune_configs(),
            {},
            d=head_dim,
            **common_nargs,
        )
        assert {
            config.kwargs["ACTIVE_WGMMA_N"]
            for config in kept
            if config.kwargs["NUM_MMA_GROUPS"] == 2
        } == expected_active_n

    for config in heuristics.autotune_configs():
        kwargs = config.kwargs
        active_n = kwargs["ACTIVE_WGMMA_N"]
        if active_n != kwargs["BLOCK_N"]:
            assert active_n == 80
        else:
            assert active_n == kwargs["BLOCK_N"]

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_Q_BUFFERS", "2")
    config = heuristics.make_config(block_n=128, num_buffers_kv=2)
    assert config.kwargs["ACTIVE_WGMMA_N"] == 128


@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_active_wgmma_n_old_schema(
    monkeypatch,
) -> None:
    scheduling = _fa3_scheduling_module()
    heuristics = scheduling.PersistentSchedulingHeuristics
    hopper_package = flag_gems.flash_attn_varlen_func.__module__.split(".ops.", 1)[0]
    persistent = import_module(f"{hopper_package}.ops.attention_impl.persistent")
    for name in _FA3_AUTOTUNE_EXPERIMENT_ENV:
        monkeypatch.delenv(name, raising=False)

    configs = [
        heuristics.make_config(block_n=128, num_buffers_kv=2),
        heuristics.make_config(block_n=64, num_buffers_kv=2),
        heuristics.make_config(
            block_m=64,
            block_n=64,
            num_buffers_kv=2,
            num_mma_groups=1,
        ),
    ]
    for config in configs:
        config.kwargs.pop("ACTIVE_WGMMA_N")

    normalized = persistent._normalize_persistent_config_schema(configs)
    assert [config.kwargs["ACTIVE_WGMMA_N"] for config in normalized] == [
        128,
        64,
        64,
    ]


@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_stagger_kv_is_prefill_only() -> None:
    scheduling = _fa3_scheduling_module()
    heuristics = scheduling.PersistentSchedulingHeuristics
    configs = heuristics.autotune_configs()

    for seqlen_q, split_kv, prefill_profile in (
        (1, False, False),
        (64, True, True),
    ):
        kept = heuristics.prune_autotune_configs(
            configs,
            {},
            d=256,
            b=4,
            h=4,
            hk=1,
            seqlen_q=seqlen_q,
            seqlen_k=4096,
            total_q=4 * seqlen_q,
            is_paged=True,
            block_size=16,
            is_causal=True,
            is_local=False,
            is_alibi=False,
            is_softcap=False,
            PAGED_KV_NON_TMA=True,
            PACK_GQA=True,
            RAGGED_SCHEDULER=False,
            SPLIT_KV=split_kv,
            PAGED_PREFILL_PROFILE=prefill_profile,
        )
        assert all(
            not config.kwargs.get("STAGGER_KV", False) for config in kept
        )
        assert all(
            not config.kwargs.get("RESCALE_O_BEFORE_PV", False)
            for config in kept
        )
        assert all(
            not config.kwargs.get("EARLY_CAST_P", False) for config in kept
        )
        assert all(
            config.kwargs["ACTIVE_WGMMA_N"] == config.kwargs["BLOCK_N"]
            for config in kept
        )


@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_tiled_extent_policy_matrix() -> None:
    scheduling = _fa3_scheduling_module()
    heuristics = scheduling.PersistentSchedulingHeuristics
    configs = heuristics.autotune_configs()
    base = {
        "d": 256,
        "b": 1,
        "h": 4,
        "hk": 1,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "total_q": 1024,
        "is_paged": True,
        "is_seqused_k": True,
        "block_size": 16,
        "is_causal": True,
        "is_local": False,
        "is_alibi": False,
        "is_softcap": False,
        "is_s_aux": False,
        "PAGED_KV_NON_TMA": True,
        "PACK_GQA": True,
        "RAGGED_SCHEDULER": False,
        "SPLIT_KV": False,
        "PAGED_PREFILL_PROFILE": True,
    }

    for positive in (
        {},
        {"h": 8, "hk": 1, "block_size": 32},
    ):
        kept = heuristics.prune_autotune_configs(
            configs, {}, **(base | positive)
        )
        tiled = [
            config
            for config in kept
            if config.kwargs["ACTIVE_WGMMA_N"] != config.kwargs["BLOCK_N"]
        ]
        assert {
            (
                config.kwargs["STAGGER_KV"],
                config.kwargs["ACTIVE_WGMMA_N"],
                config.kwargs.get("RESCALE_O_BEFORE_PV", False),
                config.kwargs.get("EARLY_CAST_P", False),
            )
            for config in tiled
        } == {
            (True, 80, False, True),
            (True, 80, True, True),
        }

    negative_overrides = (
        {"h": 8, "hk": 1},
        {"block_size": 32},
        {"block_size": 64},
        {"is_seqused_k": False},
        {"is_s_aux": True},
        {"is_causal": False},
        {"is_local": True},
        {"is_alibi": True},
        {"is_softcap": True},
        {"is_paged": False},
        {"PAGED_KV_NON_TMA": False},
        {"PACK_GQA": False},
        {"SPLIT_KV": True},
        {"PAGED_PREFILL_PROFILE": False},
    )
    for override in negative_overrides:
        kept = heuristics.prune_autotune_configs(
            configs, {}, **(base | override)
        )
        assert all(
            not config.kwargs.get("RESCALE_O_BEFORE_PV", False)
            for config in kept
        ), override
        assert all(
            not config.kwargs.get("EARLY_CAST_P", False) for config in kept
        ), override
        assert all(
            config.kwargs["ACTIVE_WGMMA_N"] != 80 for config in kept
        ), override


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@pytest.mark.parametrize(
    "route,kv_len,soft_cap,num_splits,expected_plans",
    [
        pytest.param(
            "direct",
            256,
            0.0,
            (1,),
            [("direct_packed_gqa", 1)],
            id="direct",
        ),
        pytest.param(
            "long",
            512,
            10.0,
            (1,),
            [("long_paged_prefill", 1)],
            id="persistent-long-softcap",
        ),
        pytest.param(
            "long",
            2048,
            0.0,
            (1, 2),
            [("long_paged_prefill", 1), ("persistent_splitkv_s2", 2)],
            id="split-kv",
        ),
    ],
)
@torch.inference_mode()
def test_flash_attn_varlen_fa3_s_aux(
    monkeypatch,
    route: str,
    kv_len: int,
    soft_cap: float,
    num_splits: Tuple[int, ...],
    expected_plans: List[Tuple[str, int]],
) -> None:
    FA3Scheduler, plans = _capture_fa3_plans(monkeypatch)

    utils.init_seed(1234567890)
    query_len = 2
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    ) = _make_fa3_s_aux_case(query_len, kv_len)

    ref_output, ref_lse = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap if soft_cap > 0 else None,
        s_aux=s_aux,
        return_softmax_lse=True,
    )

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", route)
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "on")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    FA3Scheduler.clear_config_cache()

    results = {}
    try:
        for split_count in num_splits:
            output, lse = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=query_len,
                max_seqlen_k=kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
                block_table=block_tables,
                softcap=soft_cap,
                s_aux=s_aux,
                return_softmax_lse=True,
                num_splits=split_count,
                fa_version=3,
            )
            results[split_count] = (output, lse)
    finally:
        FA3Scheduler.clear_config_cache()

    assert [_plan_signature(plan) for plan in plans] == expected_plans
    for output, lse in results.values():
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=1e-2)

    if len(num_splits) > 1:
        one_pass_output, one_pass_lse = results[1]
        split_output, split_lse = results[2]
        torch.testing.assert_close(split_output, one_pass_output, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(split_lse, one_pass_lse, atol=2e-2, rtol=1e-2)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@pytest.mark.parametrize("invalid_contract", ["dtype", "shape", "noncontiguous"])
@torch.inference_mode()
def test_flash_attn_varlen_fa3_s_aux_contract(invalid_contract: str) -> None:
    utils.init_seed(1234567890)
    query_len = 2
    kv_len = 64
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    ) = _make_fa3_s_aux_case(query_len, kv_len)

    if invalid_contract == "dtype":
        s_aux = s_aux.float()
    elif invalid_contract == "shape":
        s_aux = s_aux[:-1]
    else:
        s_aux = torch.empty(16, dtype=torch.bfloat16, device=device)[::2]
        assert not s_aux.is_contiguous()

    with pytest.raises(RuntimeError, match="s_aux must be a contiguous bf16"):
        flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            s_aux=s_aux,
            num_splits=1,
            fa_version=3,
        )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@pytest.mark.parametrize(
    "route,expected_kernel",
    [
        pytest.param("direct", "direct_packed_gqa", id="direct"),
        pytest.param("long", "long_paged_prefill", id="persistent-long"),
    ],
)
@torch.inference_mode()
def test_flash_attn_varlen_fa3_s_aux_empty_kv(
    monkeypatch, route: str, expected_kernel: str
) -> None:
    FA3Scheduler, plans = _capture_fa3_plans(monkeypatch)

    utils.init_seed(1234567890)
    query_len = 2
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    ) = _make_fa3_s_aux_case(query_len, kv_len=1)
    seqused_k.zero_()

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", route)
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "on")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    FA3Scheduler.clear_config_cache()

    try:
        output, lse = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=query_len,
            max_seqlen_k=1,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            s_aux=s_aux,
            return_softmax_lse=True,
            num_splits=1,
            fa_version=3,
        )
    finally:
        FA3Scheduler.clear_config_cache()

    assert [_plan_signature(plan) for plan in plans] == [(expected_kernel, 1)]
    assert torch.count_nonzero(output).item() == 0
    expected_lse = s_aux.float().unsqueeze(1).expand(-1, query_len)
    torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=0)
