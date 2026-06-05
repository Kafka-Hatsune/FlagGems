"""FA3 TLE dispatch planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os

from .utils import *  # noqa: F401,F403
def _next_power_of_2_host(value: int) -> int:
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class FA3TlePlan:
    family: str
    shape_bucket: int
    force_family_id: int
    min_q_len: int = 0
    max_q_len: int = 2**31 - 1
    num_splits: int = 1
    decode_strategy: str = "auto"
    direct_kind: str = ""
    pack_gqa: bool = False
    effective_batch_heads: int = 0
    split_policy: int = 0


def fa3_tle_force_family_id() -> int:
    value = os.getenv("FLAG_GEMS_FA3_TLE_FORCE_PATH", "auto").strip().lower()
    if value not in _FA3_TLE_FORCE_PATHS:
        allowed = ", ".join(sorted(_FA3_TLE_FORCE_PATHS))
        raise RuntimeError(
            f"invalid FLAG_GEMS_FA3_TLE_FORCE_PATH={value!r}; expected one of {allowed}"
        )
    return _FA3_TLE_FORCE_PATHS[value]


def fa3_tle_decode_strategy() -> str:
    value = os.getenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", "auto").strip().lower()
    allowed = ("auto", "onepass", "splitkv", "flashdecoding", "short")
    if value not in allowed:
        raise RuntimeError(
            "invalid FLAG_GEMS_FA3_TLE_DECODE_STRATEGY="
            f"{value!r}; expected one of {', '.join(allowed)}"
        )
    return value


def fa3_tle_small_strategy() -> str:
    value = os.getenv("FLAG_GEMS_FA3_TLE_SMALL_STRATEGY", "auto").strip().lower()
    allowed = ("auto", "direct", "short")
    if value not in allowed:
        raise RuntimeError(
            "invalid FLAG_GEMS_FA3_TLE_SMALL_STRATEGY="
            f"{value!r}; expected one of {', '.join(allowed)}"
        )
    return value


def fa3_tle_ws_strategy() -> str:
    value = os.getenv("FLAG_GEMS_FA3_TLE_WS_STRATEGY", "auto").strip().lower()
    allowed = ("auto", "ws_short", "ws_simple", "legacy")
    if value not in allowed:
        raise RuntimeError(
            "invalid FLAG_GEMS_FA3_TLE_WS_STRATEGY="
            f"{value!r}; expected one of {', '.join(allowed)}"
        )
    return value


def _fa3_tle_ws_candidate_plan(
    *,
    force_family_id: int,
    avg_q: float,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    is_paged: bool,
) -> FA3TlePlan:
    name = _FA3_TLE_WS_CANDIDATE_NAMES[force_family_id]
    bucket = _FA3_TLE_WS_CANDIDATE_BUCKETS[force_family_id]
    is_decode_candidate = force_family_id in (
        _FA3_TLE_FAMILY_WS_SYNC_DECODE,
        _FA3_TLE_FAMILY_WS_PIPE2_DECODE,
        _FA3_TLE_FAMILY_WS_SYNC_PAGED_DECODE,
        _FA3_TLE_FAMILY_WS_PIPE2_PAGED_DECODE,
    )
    is_paged_candidate = force_family_id in (
        _FA3_TLE_FAMILY_WS_SYNC_PAGED_DECODE,
        _FA3_TLE_FAMILY_WS_PIPE2_PAGED_DECODE,
    )
    is_dense_decode_candidate = force_family_id in (
        _FA3_TLE_FAMILY_WS_SYNC_DECODE,
        _FA3_TLE_FAMILY_WS_PIPE2_DECODE,
    )

    if is_dense_decode_candidate and is_paged:
        raise RuntimeError(f"{name} expects dense K/V decode input, got paged K/V")
    if is_paged_candidate and not is_paged:
        raise RuntimeError(f"{name} expects paged K/V decode input, got dense K/V")
    if (
        is_paged_candidate
        and head_dim < 192
        and os.getenv("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_D128") != "1"
    ):
        raise RuntimeError(
            f"{name} is only enabled for paged high-D decode by default. "
            f"Got head_dim={head_dim}; set FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_D128=1 "
            "only for timeout-guarded debug runs."
        )
    if is_decode_candidate and (avg_q > 4 or max_seqlen_q > 64):
        raise RuntimeError(
            f"{name} expects decode-like input with avg_q<=4 and max_q<=64, "
            f"got avg_q={avg_q:.3f}, max_q={max_seqlen_q}"
        )
    if force_family_id == _FA3_TLE_FAMILY_WS_SYNC_SMALL:
        if is_paged and os.getenv("FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL") != "1":
            raise RuntimeError(
                f"{name} is currently dense-small only. Paged small/medium varlen "
                "uses a risky hand-staged K/V barrier path; set "
                "FLAG_GEMS_FA3_TLE_ALLOW_RISKY_PAGED_SMALL=1 only for timeout-guarded "
                "debug runs."
            )
        if max_seqlen_q > 640:
            raise RuntimeError(
                f"{name} expects small/medium Q input with max_q<=640, "
                f"got max_q={max_seqlen_q}"
            )
        if max_seqlen_k > 4096:
            raise RuntimeError(
                f"{name} expects bounded K input with max_k<=4096, "
                f"got max_k={max_seqlen_k}"
            )

    direct_kind = "small"
    if is_paged_candidate:
        direct_kind = "paged_decode"
    elif is_dense_decode_candidate:
        direct_kind = "decode"
    elif is_paged:
        direct_kind = "paged_small"

    return FA3TlePlan(
        "ws_short",
        bucket,
        force_family_id,
        decode_strategy=name,
        direct_kind=direct_kind,
        pack_gqa=is_decode_candidate and max_seqlen_q > 1,
        effective_batch_heads=0,
    )


def fa3_tle_select_plan(
    *,
    total_q: int,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    is_paged: bool,
    force_family_id: int,
    num_heads: int = 1,
    decode_strategy: str = "auto",
    small_strategy: str = "auto",
    ws_strategy: str = "auto",
    num_sms: int = 0,
) -> FA3TlePlan:
    avg_q = total_q / max(batch_size, 1)

    if force_family_id in _FA3_TLE_WS_CANDIDATE_NAMES:
        return _fa3_tle_ws_candidate_plan(
            force_family_id=force_family_id,
            avg_q=avg_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            head_dim=head_dim,
            is_paged=is_paged,
        )

    if force_family_id == _FA3_TLE_FAMILY_LONG:
        return FA3TlePlan("long", _FA3_TLE_BUCKET_LONG, force_family_id)
    if force_family_id == _FA3_TLE_FAMILY_SHORT:
        return FA3TlePlan("short", _FA3_TLE_BUCKET_SHORT, force_family_id)
    if force_family_id == _FA3_TLE_FAMILY_SPLITKV:
        return FA3TlePlan(
            "splitkv",
            _FA3_TLE_BUCKET_SPLITKV,
            force_family_id,
            num_splits=_fa3_tle_decode_splits(max_seqlen_k, head_dim),
            decode_strategy="splitkv",
        )
    if force_family_id == _FA3_TLE_FAMILY_DIRECT:
        if avg_q <= 4 and max_seqlen_q <= 64:
            return FA3TlePlan(
                "direct",
                _FA3_TLE_BUCKET_DIRECT_PAGED_DECODE
                if is_paged
                else _FA3_TLE_BUCKET_DIRECT_DECODE,
                force_family_id,
                decode_strategy="onepass",
                direct_kind="paged_decode" if is_paged else "decode",
            )
        return FA3TlePlan(
            "direct",
            _FA3_TLE_BUCKET_DIRECT_SMALL,
            force_family_id,
            decode_strategy="onepass",
            direct_kind="small_dense",
        )
    if force_family_id == _FA3_TLE_FAMILY_WS_SIMPLE:
        if avg_q <= 4 and max_seqlen_q <= 64:
            return FA3TlePlan(
                "ws_simple",
                _FA3_TLE_BUCKET_WS_PAGED_DECODE
                if is_paged
                else _FA3_TLE_BUCKET_WS_DECODE,
                force_family_id,
                decode_strategy="ws_simple",
                direct_kind="paged_decode" if is_paged else "decode",
            )
        return FA3TlePlan(
            "ws_simple",
            _FA3_TLE_BUCKET_WS_SMALL_DENSE,
            force_family_id,
            decode_strategy="ws_simple",
            direct_kind="small_dense",
        )
    if force_family_id == _FA3_TLE_FAMILY_WS_SHORT:
        if avg_q <= 4 and max_seqlen_q <= 64:
            return FA3TlePlan(
                "ws_short",
                _FA3_TLE_BUCKET_WS_SHORT_PAGED_DECODE
                if is_paged
                else _FA3_TLE_BUCKET_WS_SHORT_DECODE,
                force_family_id,
                decode_strategy="ws_short",
                direct_kind="paged_decode" if is_paged else "decode",
                pack_gqa=max_seqlen_q > 1,
                effective_batch_heads=batch_size * max_seqlen_q,
            )
        return FA3TlePlan(
            "ws_short",
            _FA3_TLE_BUCKET_WS_SHORT_SMALL_DENSE,
            force_family_id,
            decode_strategy="ws_short",
            direct_kind="small_dense",
        )
    if force_family_id == _FA3_TLE_FAMILY_MIXED:
        return FA3TlePlan(
            "paged_serve" if is_paged else "serve",
            _FA3_TLE_BUCKET_PAGED_SERVE_SHORT
            if is_paged
            else _FA3_TLE_BUCKET_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )
    if force_family_id == _FA3_TLE_FAMILY_DECODE:
        return FA3TlePlan(
            "direct",
            _FA3_TLE_BUCKET_DIRECT_DECODE,
            force_family_id,
            decode_strategy="onepass",
            direct_kind="decode",
        )
    if force_family_id == _FA3_TLE_FAMILY_PAGED_DECODE:
        return _fa3_tle_flashdecoding_plan(
            force_family_id=force_family_id,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            head_dim=head_dim,
            is_paged=True,
            num_sms=num_sms,
        )
    if force_family_id == _FA3_TLE_FAMILY_SERVE:
        return FA3TlePlan(
            "serve",
            _FA3_TLE_BUCKET_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )
    if force_family_id == _FA3_TLE_FAMILY_PAGED_SERVE:
        return FA3TlePlan(
            "paged_serve",
            _FA3_TLE_BUCKET_PAGED_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )

    q_to_max_ratio = total_q / max(max_seqlen_q, 1)
    is_mixed_by_distribution = (
        batch_size >= 4
        and max_seqlen_q >= 1024
        and max_seqlen_q <= 4096
        and q_to_max_ratio <= 2.0
    )
    if max_seqlen_q > 512 and (avg_q <= 128 or is_mixed_by_distribution):
        return FA3TlePlan(
            "paged_serve" if is_paged else "serve",
            _FA3_TLE_BUCKET_PAGED_SERVE_SHORT
            if is_paged
            else _FA3_TLE_BUCKET_SERVE_SHORT,
            force_family_id,
            max_q_len=64,
        )
    if is_paged and avg_q <= 4 and max_seqlen_q <= 32 and max_seqlen_k >= 1024:
        if decode_strategy == "short":
            return FA3TlePlan(
                "short",
                _FA3_TLE_BUCKET_PAGED_DECODE,
                force_family_id,
                decode_strategy="short",
                direct_kind="paged_decode",
                pack_gqa=max_seqlen_q > 1,
                effective_batch_heads=batch_size * max_seqlen_q,
            )
        if decode_strategy == "splitkv":
            return FA3TlePlan(
                "splitkv",
                _FA3_TLE_BUCKET_SPLITKV,
                force_family_id,
                num_splits=_fa3_tle_decode_splits(max_seqlen_k, head_dim),
                decode_strategy="splitkv",
                pack_gqa=max_seqlen_q > 1,
                effective_batch_heads=batch_size * max_seqlen_q,
            )
        if decode_strategy == "onepass":
            return FA3TlePlan(
                "direct",
                _FA3_TLE_BUCKET_DIRECT_PAGED_DECODE,
                force_family_id,
                decode_strategy="onepass",
                direct_kind="paged_decode",
                pack_gqa=max_seqlen_q > 1,
                effective_batch_heads=batch_size * max_seqlen_q,
            )
        return _fa3_tle_flashdecoding_plan(
            force_family_id=force_family_id,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            head_dim=head_dim,
            is_paged=is_paged,
            num_sms=num_sms,
        )
    if avg_q <= 4 and max_seqlen_q <= 8:
        if decode_strategy == "short":
            return FA3TlePlan(
                "short",
                _FA3_TLE_BUCKET_SHORT,
                force_family_id,
                decode_strategy="short",
            )
        if decode_strategy == "flashdecoding" and is_paged:
            return _fa3_tle_flashdecoding_plan(
                force_family_id=force_family_id,
                batch_size=batch_size,
                num_heads=num_heads,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                head_dim=head_dim,
                is_paged=is_paged,
                num_sms=num_sms,
            )
        should_split = False
        if decode_strategy == "splitkv":
            should_split = max_seqlen_k >= 1024
        elif decode_strategy == "onepass":
            should_split = False
        if should_split:
            return FA3TlePlan(
                "splitkv",
                _FA3_TLE_BUCKET_SPLITKV,
                force_family_id,
                num_splits=_fa3_tle_decode_splits(max_seqlen_k, head_dim),
                decode_strategy="splitkv",
                pack_gqa=max_seqlen_q > 1,
                effective_batch_heads=batch_size * max_seqlen_q,
            )
        if decode_strategy == "onepass":
            return FA3TlePlan(
                "direct",
                _FA3_TLE_BUCKET_DIRECT_PAGED_DECODE
                if is_paged
                else _FA3_TLE_BUCKET_DIRECT_DECODE,
                force_family_id,
                decode_strategy="onepass",
                direct_kind="paged_decode" if is_paged else "decode",
                pack_gqa=max_seqlen_q > 1,
                effective_batch_heads=batch_size * max_seqlen_q,
            )
        if ws_strategy == "ws_short":
            if (not is_paged and max_seqlen_k < 4096) or (
                is_paged and head_dim >= 192
            ):
                return FA3TlePlan(
                    "ws_short",
                    _FA3_TLE_BUCKET_WS_SHORT_PAGED_DECODE
                    if is_paged
                    else _FA3_TLE_BUCKET_WS_SHORT_DECODE,
                    force_family_id,
                    decode_strategy="ws_short",
                    direct_kind="paged_decode" if is_paged else "decode",
                    pack_gqa=max_seqlen_q > 1,
                    effective_batch_heads=batch_size * max_seqlen_q,
                )
        if ws_strategy == "ws_simple":
            if (not is_paged and max_seqlen_k < 4096) or (
                is_paged and head_dim >= 192
            ):
                return FA3TlePlan(
                    "ws_simple",
                    _FA3_TLE_BUCKET_WS_PAGED_DECODE
                    if is_paged
                    else _FA3_TLE_BUCKET_WS_DECODE,
                    force_family_id,
                    decode_strategy="ws_simple",
                    direct_kind="paged_decode" if is_paged else "decode",
                    pack_gqa=max_seqlen_q > 1,
                    effective_batch_heads=batch_size * max_seqlen_q,
                )
        if is_paged:
            return _fa3_tle_flashdecoding_plan(
                force_family_id=force_family_id,
                batch_size=batch_size,
                num_heads=num_heads,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                head_dim=head_dim,
                is_paged=is_paged,
                num_sms=num_sms,
            )
        return FA3TlePlan(
            "direct",
            _FA3_TLE_BUCKET_DIRECT_DECODE,
            force_family_id,
            decode_strategy="onepass",
            direct_kind="decode",
            pack_gqa=max_seqlen_q > 1,
            effective_batch_heads=batch_size * max_seqlen_q,
        )
    if (
        small_strategy != "short"
        and ws_strategy == "ws_short"
        and not is_paged
        and total_q <= 512
        and max_seqlen_q <= 640
        and max_seqlen_k <= 1024
    ):
        return FA3TlePlan(
            "ws_short",
            _FA3_TLE_BUCKET_WS_SHORT_SMALL_DENSE,
            force_family_id,
            decode_strategy="ws_short",
            direct_kind="small_dense",
        )
    if (
        small_strategy != "short"
        and ws_strategy == "ws_simple"
        and not is_paged
        and total_q <= 512
        and max_seqlen_q <= 640
        and max_seqlen_k <= 1024
    ):
        return FA3TlePlan(
            "ws_simple",
            _FA3_TLE_BUCKET_WS_SMALL_DENSE,
            force_family_id,
            decode_strategy="ws_simple",
            direct_kind="small_dense",
        )
    if (
        small_strategy == "direct"
        and not is_paged
        and avg_q <= 64
        and max_seqlen_q <= 1024
    ):
        return FA3TlePlan(
            "direct",
            _FA3_TLE_BUCKET_DIRECT_SMALL,
            force_family_id,
            decode_strategy="onepass",
            direct_kind="small_dense",
        )
    if is_paged and max_seqlen_q <= 512 and max_seqlen_k <= 1024:
        if max_seqlen_q <= 128:
            return FA3TlePlan(
                "short", _FA3_TLE_BUCKET_PAGED_SMALL, force_family_id
            )
        return FA3TlePlan(
            "short", _FA3_TLE_BUCKET_PAGED_MEDIUM, force_family_id
        )
    if avg_q <= 64 or max_seqlen_q <= 128:
        return FA3TlePlan("short", _FA3_TLE_BUCKET_SHORT, force_family_id)
    return FA3TlePlan("long", _FA3_TLE_BUCKET_LONG, force_family_id)


def _fa3_tle_decode_splits(max_seqlen_k: int, head_dim: int) -> int:
    del head_dim
    if max_seqlen_k >= 4096:
        return 8
    if max_seqlen_k >= 2048:
        return 4
    if max_seqlen_k >= 1024:
        return 2
    return 2


def _fa3_tle_flashdecoding_splits(
    *,
    batch_size: int,
    num_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_sms: int,
) -> int:
    if max_seqlen_k >= 8192:
        by_k = 16
    elif max_seqlen_k >= 4096:
        by_k = 8
    elif max_seqlen_k >= 2048:
        by_k = 4
    else:
        by_k = 2

    active_qh = max(1, batch_size * max(num_heads, 1) * max(max_seqlen_q, 1))
    by_occupancy = 1
    if num_sms > 0:
        by_occupancy = max(1, (num_sms + active_qh - 1) // active_qh)

    splits = max(by_k, by_occupancy)
    if active_qh >= max(1, num_sms):
        splits = min(splits, 2 if max_seqlen_k < 4096 else 4)
    return max(2, min(16, splits))


def _fa3_tle_flashdecoding_plan(
    *,
    force_family_id: int,
    batch_size: int,
    num_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    is_paged: bool,
    num_sms: int,
) -> FA3TlePlan:
    del head_dim
    return FA3TlePlan(
        "flashdecoding",
        _FA3_TLE_BUCKET_SPLITKV,
        force_family_id,
        num_splits=_fa3_tle_flashdecoding_splits(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_sms=num_sms,
        ),
        decode_strategy="flashdecoding",
        direct_kind="paged_decode" if is_paged else "decode",
        pack_gqa=max_seqlen_q > 1,
        effective_batch_heads=batch_size * max_seqlen_q,
        split_policy=0,
    )


def _fa3_tle_should_splitkv(
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    is_paged: bool,
    num_sms: int,
) -> bool:
    n_blocks = (max_seqlen_k + 63) // 64
    if n_blocks <= 4 or max_seqlen_k < 1024:
        return False

    if not is_paged:
        if max_seqlen_k >= 2048 and batch_size <= 32:
            return True
        return batch_size <= 16

    if head_dim >= 192 and max_seqlen_k >= 1536 and batch_size <= 16:
        return True

    effective_m_blocks = batch_size * max(max_seqlen_q, 1)
    if effective_m_blocks >= max(num_sms, 1):
        return False
    return batch_size >= 64


def fa3_tle_mixed_long_plan(force_family_id: int) -> FA3TlePlan:
    return FA3TlePlan(
        "mixed_long",
        _FA3_TLE_BUCKET_MIXED_LONG,
        force_family_id,
        min_q_len=65,
    )


__all__ = [
    "FA3TlePlan",
    "fa3_tle_force_family_id",
    "fa3_tle_decode_strategy",
    "fa3_tle_small_strategy",
    "fa3_tle_ws_strategy",
    "fa3_tle_select_plan",
    "fa3_tle_mixed_long_plan",
]
