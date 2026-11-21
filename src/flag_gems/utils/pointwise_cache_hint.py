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

"""Native policies for measured streaming pointwise layouts."""

from inspect import signature

import torch
import triton.language as tl

try:
    from triton.experimental.tle.language import gpu
except ImportError:
    gpu = None

_HAS_POLICY = all(
    "cache_policy" in signature(op).parameters for op in (tl.load, tl.store)
)


def _policy(backend, name, *args, **kwargs):
    namespace = gpu if backend is None else getattr(gpu, backend, None)
    constructor = getattr(namespace, name, None)
    return (
        constructor(*args, **kwargs)
        if _HAS_POLICY and constructor is not None
        else None
    )


NVIDIA_L2_FIRST = _policy(
    None, "CachePolicy", l2=_policy(None, "FractionalEvictionPolicy", "evict_first")
)
METAX_CG = _policy("metax", "CachePolicy", "cg")
PPU_LU = _policy("ppu", "CachePolicy", ".lu")
HCU_CS = _policy("hcu", "CachePolicy", ".cs")
MUSA_PERSIST = _policy(
    "mthreads", "LSUCachePolicy", inner_persistence=3, outer_persistence=3, l2_policy=1
)

POLICIES = {
    "nvidia_read_first": ("NVIDIA_L2_FIRST", None),
    "metax_read_cg": ("METAX_CG", None),
    "ppu_read_lu": ("PPU_LU", None),
    "hcu_write_cs": (None, "HCU_CS"),
    "musa_write_persist": (None, "MUSA_PERSIST"),
}

_VENDOR_FAMILIES = {
    "nvidia": {
        "abs_func",
        "ceil_func",
        "cos_func",
        "floor_func",
        "gelu_none",
        "neg_func",
        "relu_forward",
        "rsqrt_func",
        "sin_func",
        "square_func",
    },
    "mthreads": {
        "abs_func",
        "ceil_func",
        "cos_func",
        "exp_func",
        "floor_func",
        "gelu_none",
        "log1p_func",
        "neg_func",
        "relu_forward",
        "rsqrt_func",
        "sigmoid_forward",
        "silu_forward",
        "sin_func",
        "square_func",
    },
    "hygon": {
        "abs_func",
        "ceil_func",
        "exp_func",
        "floor_func",
        "frac_func",
        "neg_func",
        "relu_forward",
        "sqrt_func",
        "square_func",
        "trunc_func",
    },
    "metax": {
        "abs_func",
        "ceil_func",
        "cos_func",
        "exp_func",
        "floor_func",
        "gelu_none",
        "log1p_func",
        "neg_func",
        "relu_forward",
        "rsqrt_func",
        "sigmoid_forward",
        "silu_forward",
        "sin_func",
        "square_func",
    },
    "thead": {
        "abs_func",
    },
}


def pointwise_family(fn):
    if not _HAS_POLICY:
        return None
    # Selected scalar kernels only; backward and complex kernels keep their defaults.
    name = fn.__name__
    if any(name in families for families in _VENDOR_FAMILIES.values()):
        return name
    return None


def _contiguous(tensor):
    stride = 1
    for size, actual in zip(reversed(tensor.shape), reversed(tensor.stride())):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return True


def select_policy(family, args, outputs):
    import flag_gems

    if len(outputs) != 1 or outputs["out0"].numel() < 2**20:
        return None
    vendor = flag_gems.vendor_name
    if family not in _VENDOR_FAMILIES.get(vendor, ()):
        return None
    tensors = [arg for arg in args if hasattr(arg, "stride")]
    if not tensors:
        return None
    out = outputs["out0"]
    if any(t.dtype != torch.float32 for t in (*tensors, out)) or not _contiguous(out):
        return None
    if any(
        t.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        for t in tensors
    ):
        return None
    count = out.numel()
    if vendor in ("nvidia", "metax") and len(tensors) == 1:
        x = tensors[0]
        strides = x.stride()
        # Interleaved FP32 views whose storage spans many cache lines.
        if not strides or min(strides) < 4 or min(strides) > 16:
            return None
        if (
            vendor == "nvidia"
            and NVIDIA_L2_FIRST is not None
            and 2**20 <= count <= 2**22
            and count * min(strides) * 4 >= 2**25
        ):
            return "nvidia_read_first"
        if (
            vendor == "metax"
            and METAX_CG is not None
            and 2**23 <= count <= 2**25
            and min(strides) >= 8
        ):
            return "metax_read_cg"
    if vendor == "hygon" and HCU_CS is not None and 2**25 <= count <= 2**28:
        if all(0 in t.stride() for t in tensors):
            return "hcu_write_cs"
    if vendor == "thead" and PPU_LU is not None and len(tensors) == 1:
        if 2**23 <= count <= 2**24 and _contiguous(tensors[0]):
            return "ppu_read_lu"
    if vendor == "mthreads" and MUSA_PERSIST is not None and len(tensors) == 1:
        if 2**23 <= count <= 2**24 and _contiguous(tensors[0]):
            return "musa_write_persist"
    return None
