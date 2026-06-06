"""Minimal vLLM-style FA3 metadata dispatch helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FA3MetadataDispatch:
    is_paged: bool
    has_cache_kv: bool
    mode: str
    split_kv: bool
    requested_num_splits: int
    has_scheduler_metadata: bool

    @property
    def layout(self) -> str:
        return "paged" if self.is_paged else "dense"


def fa3_tle_metadata_dispatch(
    *,
    max_query_len: int,
    is_paged: bool,
    has_cache_kv: bool,
    num_splits: int,
    has_scheduler_metadata: bool = False,
) -> FA3MetadataDispatch:
    if max_query_len <= 1 and has_cache_kv:
        mode = "normal_decode"
    elif max_query_len > 1 and has_cache_kv:
        mode = "multi_token_decode"
    else:
        mode = "prefill"

    requested_num_splits = max(0, int(num_splits or 0))
    return FA3MetadataDispatch(
        is_paged=is_paged,
        has_cache_kv=has_cache_kv,
        mode=mode,
        split_kv=requested_num_splits > 1,
        requested_num_splits=requested_num_splits,
        has_scheduler_metadata=has_scheduler_metadata,
    )


__all__ = ["FA3MetadataDispatch", "fa3_tle_metadata_dispatch"]
