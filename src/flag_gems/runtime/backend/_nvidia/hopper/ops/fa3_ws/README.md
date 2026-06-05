# Hopper FA3 TLE Kernels

This package contains the active Hopper FA3 TLE forward kernels used through
`ops/flash_kernel_v3.py` and `flash_api_v3.py`.  Decode-only prototypes that
are not part of the current dispatch path live in `bak/`.

## Active Dispatch Surface

| File | Role |
| --- | --- |
| `best_known.py` | Best-known route policy: keep proven FA3 workloads, route weak workloads to FA2 fallback. |
| `kernels.py` | Public re-export layer for the active FA3 kernels. |
| `planning.py` | FA3 family planning for forced paths and active FA3 workloads. |
| `utils.py` | Shared masks, ALiBi, paged gather, TLE copy/barrier helpers, and autotune helpers. |
| `fa_hopper_persistent_pingpong.py` | Persistent long/prefill kernel and `ws_simple` force-path experiments. |
| `fa_hopper_short.py` | Short/mixed/serve family used by active FA3 plans. |
| `fa_hopper_direct.py` | Direct one-pass family used by force paths and selected small dense paths. |
| `fa_hopper_splitkv.py` | Split-KV family plus combine kernel. |
| `fa_hopper_decode_flashdecoding.py` | Flash-Decoding-style split-KV family used by forced paged decode paths. |
| `fa_hopper_nonpersistent_tlx_style.py` | Nonpersistent WS family used by WS force-path tests. |
| `registry.py` | Script/test registry for WS force-path variants. |

## Best-Known Route

`FLAG_GEMS_FA3_TLE_BEST_ROUTE=auto` is the default.  In this mode the current
FA3 path is used only for workloads that have been benchmarked as faster:

- `dense_prefill_or_long`
- `paged_serve_mixed`

Other workloads currently route to the existing FA2 launcher from inside the
FA3 API path.  This keeps `flash_attn_varlen_func(..., fa_version=3)` stable
while avoiding known slow FA3 decode and paged-small cases.

Debug/override knobs:

```bash
FLAG_GEMS_FA3_TLE_BEST_ROUTE=auto      # default best-known policy
FLAG_GEMS_FA3_TLE_BEST_ROUTE=fa3_only  # disable FA2 fallback
FLAG_GEMS_FA3_TLE_BEST_ROUTE=fa2_only  # route every FA3 request to FA2
FLAG_GEMS_FA3_TLE_FORCE_PATH=...       # force a specific FA3 family, bypassing best-route fallback
FLAG_GEMS_FA3_TLE_LOG_PLAN=1           # print route/family information
```

## Archived Experiments

The following decode-only experiments are archived in `bak/`:

- `fa_hopper_decode_onepass.py`
- `fa_hopper_decode_splitkv.py`
- `fa_hopper_decode_paged_lb.py`
- `fa_hopper_decode_seesaw.py`
- `experimental_registry.py`

They remain importable for `scripts/run_fa3_decode_experiments.py`, but are not
re-exported by `kernels.py` and are not selected by default dispatch.
