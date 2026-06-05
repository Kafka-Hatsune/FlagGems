"""FlashInfer-style load-balanced paged decode FA3 TLE entrypoint.

This module intentionally reuses the Flash-Decoding split/combine protocol.
The experiment script launches it with ``SPLIT_POLICY=1`` so KV blocks are
assigned round-robin across splits, which keeps long or paged KV work balanced
without using the warp-specialized shared-memory copy path.
"""

from .fa_hopper_decode_flashdecoding import (
    flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel as flash_varlen_fwd_v3_tle_decode_paged_lb_combine_kernel,
    flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel as flash_varlen_fwd_v3_tle_decode_paged_lb_kernel,
)


__all__ = [
    "flash_varlen_fwd_v3_tle_decode_paged_lb_kernel",
    "flash_varlen_fwd_v3_tle_decode_paged_lb_combine_kernel",
]
