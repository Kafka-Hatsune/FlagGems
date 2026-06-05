"""Decode-first staged WS FA3 TLE experimental entrypoint.

This first candidate reserves the seesaw experiment slot while reusing the
existing persistent WS kernel.  It gives the decode benchmark suite a barriered
producer-staged dense/paged K/V candidate with user-promised WGMMA waits, while
keeping the production dispatcher untouched.  A true KV/output-split seesaw
kernel should live in this module once the baseline data justifies it.
"""

from .fa_hopper_persistent_pingpong import (
    flash_varlen_fwd_v3_tle_ws_simple_kernel as flash_varlen_fwd_v3_tle_decode_seesaw_kernel,
)

__all__ = ["flash_varlen_fwd_v3_tle_decode_seesaw_kernel"]
