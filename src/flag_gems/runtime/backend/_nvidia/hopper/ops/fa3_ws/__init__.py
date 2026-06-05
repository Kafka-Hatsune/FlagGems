"""Experimental Hopper FA3 warp-specialized kernel suite.

The modules in this package provide script-facing entry points for comparing
FA3 warp-specialization variants.  They intentionally do not change the default
``flash_attn_varlen_func`` dispatch path.
"""

from .registry import (
    WSVariant,
    get_variant,
    iter_variants,
    resolve_variant_names,
    variant_names,
)

__all__ = [
    "WSVariant",
    "get_variant",
    "iter_variants",
    "resolve_variant_names",
    "variant_names",
]
