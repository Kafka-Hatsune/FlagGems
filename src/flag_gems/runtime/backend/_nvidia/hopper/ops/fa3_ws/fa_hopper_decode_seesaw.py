"""Decode-first 1P2C ping-pong WS FA3 TLE experimental entrypoint.

The first experimental seesaw candidate is wired to the existing persistent
ping-pong 1-producer/2-consumer kernel.  It gives the decode benchmark suite a
barriered WS candidate with producer-staged dense/paged K/V, named-barrier
consumer alternation, and user-promised WGMMA waits, while keeping the
production dispatcher untouched.
"""

from .fa_hopper_persistent_pingpong import (
    flash_varlen_fwd_v3_tle_ws_simple_kernel as flash_varlen_fwd_v3_tle_decode_seesaw_kernel,
)

__all__ = ["flash_varlen_fwd_v3_tle_decode_seesaw_kernel"]
