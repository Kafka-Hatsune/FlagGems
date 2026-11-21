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

"""Optional cache hints for compiler versions with different memory APIs."""

from inspect import signature

import triton.language as tl
from triton.language.core import builtin

try:
    from triton.experimental.tle.language import gpu
except ImportError:
    gpu = None


_HAS_MEMORY_POLICY = all(
    "cache_policy" in signature(op).parameters for op in (tl.load, tl.store)
)


def _memory_policy(backend, name, *args, **kwargs):
    constructor = getattr(getattr(gpu, backend, None), name, None)
    if _HAS_MEMORY_POLICY and constructor is not None:
        return constructor(*args, **kwargs)
    return None


HCU_CS = _memory_policy("hcu", "CachePolicy", ".cs")
METAX_CS = _memory_policy("metax", "CachePolicy", "cs")
PPU_LU = _memory_policy("ppu", "CachePolicy", ".lu")
PPU_NC = _memory_policy("ppu", "CachePolicy", ".nc")
PPU_CS = _memory_policy("ppu", "CachePolicy", ".cs")
MUSA_NORM_STORE = _memory_policy(
    "mthreads", "LSUCachePolicy", inner_persistence=3, outer_persistence=3, l2_policy=1
)

if _HAS_MEMORY_POLICY:
    load = tl.load
    store = tl.store
else:

    @builtin
    def load(
        pointer, *args, cache_policy=None, _semantic=None, _builder=None, **kwargs
    ):
        if _semantic is not None:
            return tl.load(pointer, *args, _semantic=_semantic, **kwargs)
        return tl.load(pointer, *args, _builder=_builder, **kwargs)

    @builtin
    def store(
        pointer,
        value,
        *args,
        cache_policy=None,
        _semantic=None,
        _builder=None,
        **kwargs,
    ):
        if _semantic is not None:
            return tl.store(pointer, value, *args, _semantic=_semantic, **kwargs)
        return tl.store(pointer, value, *args, _builder=_builder, **kwargs)
