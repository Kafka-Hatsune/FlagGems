"""Public compatibility surface for Hopper FA3 TLE kernels.

The implementation is split by kernel family in this package.  This module keeps
one stable import surface for ``flash_kernel_v3.py`` and ``flash_api_v3.py``.
"""

from .utils import (
    TLE_FA3_AVAILABLE,
    fa3_tle_paged_gather_mode,
    fa3_tle_paged_gather_name,
)
from .planning import (
    FA3TlePlan,
    fa3_tle_decode_strategy,
    fa3_tle_force_family_id,
    fa3_tle_mixed_long_plan,
    fa3_tle_select_plan,
    fa3_tle_small_strategy,
    fa3_tle_ws_strategy,
)
from .fa_hopper_persistent_pingpong import (
    flash_varlen_fwd_v3_tle_kernel,
    flash_varlen_fwd_v3_tle_ws_simple_kernel,
)
from .best_known import (
    FA3BestRoute,
    ROUTE_CURRENT_FA3,
    ROUTE_FA2_FALLBACK,
    ROUTE_RESTORED_FA3,
    classify_fa3_workload,
    fa3_tle_best_route_mode,
    select_fa3_best_route,
)
from .fa_hopper_nonpersistent_tlx_style import flash_varlen_fwd_v3_tle_ws_short_kernel
from .fa_hopper_short import flash_varlen_fwd_v3_tle_short_kernel
from .fa_hopper_direct import flash_varlen_fwd_v3_tle_direct_kernel
from .fa_hopper_decode_onepass import flash_varlen_fwd_v3_tle_decode_onepass_kernel
from .fa_hopper_decode_splitkv import (
    flash_varlen_fwd_v3_tle_decode_splitkv_combine_kernel,
    flash_varlen_fwd_v3_tle_decode_splitkv_kernel,
)
from .fa_hopper_decode_flashdecoding import (
    flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel,
    flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel,
)
from .fa_hopper_decode_paged_lb import (
    flash_varlen_fwd_v3_tle_decode_paged_lb_combine_kernel,
    flash_varlen_fwd_v3_tle_decode_paged_lb_kernel,
)
from .fa_hopper_decode_seesaw import flash_varlen_fwd_v3_tle_decode_seesaw_kernel
from .fa_hopper_splitkv import (
    flash_varlen_fwd_v3_tle_splitkv_combine_kernel,
    flash_varlen_fwd_v3_tle_splitkv_kernel,
)

__all__ = [
    "TLE_FA3_AVAILABLE",
    "FA3TlePlan",
    "fa3_tle_force_family_id",
    "fa3_tle_decode_strategy",
    "fa3_tle_small_strategy",
    "fa3_tle_ws_strategy",
    "fa3_tle_paged_gather_mode",
    "fa3_tle_paged_gather_name",
    "fa3_tle_select_plan",
    "fa3_tle_mixed_long_plan",
    "FA3BestRoute",
    "ROUTE_CURRENT_FA3",
    "ROUTE_RESTORED_FA3",
    "ROUTE_FA2_FALLBACK",
    "classify_fa3_workload",
    "fa3_tle_best_route_mode",
    "select_fa3_best_route",
    "flash_varlen_fwd_v3_tle_kernel",
    "flash_varlen_fwd_v3_tle_ws_simple_kernel",
    "flash_varlen_fwd_v3_tle_ws_short_kernel",
    "flash_varlen_fwd_v3_tle_short_kernel",
    "flash_varlen_fwd_v3_tle_direct_kernel",
    "flash_varlen_fwd_v3_tle_decode_onepass_kernel",
    "flash_varlen_fwd_v3_tle_decode_splitkv_kernel",
    "flash_varlen_fwd_v3_tle_decode_splitkv_combine_kernel",
    "flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel",
    "flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel",
    "flash_varlen_fwd_v3_tle_decode_paged_lb_kernel",
    "flash_varlen_fwd_v3_tle_decode_paged_lb_combine_kernel",
    "flash_varlen_fwd_v3_tle_decode_seesaw_kernel",
    "flash_varlen_fwd_v3_tle_splitkv_kernel",
    "flash_varlen_fwd_v3_tle_splitkv_combine_kernel",
]
