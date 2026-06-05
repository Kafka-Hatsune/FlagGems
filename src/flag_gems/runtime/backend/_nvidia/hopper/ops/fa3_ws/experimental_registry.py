"""Registry for decode-first FA3 experimental kernels.

These variants are intentionally script-facing only.  They are not selected by
``flash_api_v3.py`` or the normal FlagGems autotune/dispatch path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeExperiment:
    name: str
    family: str
    module: str
    paged: bool
    description: str


EXPERIMENTS: dict[str, DecodeExperiment] = {
    "decode_onepass_dense": DecodeExperiment(
        name="decode_onepass_dense",
        family="onepass",
        module="fa_hopper_decode_onepass.py",
        paged=False,
        description="Single-row PackGQA dense decode.",
    ),
    "decode_onepass_paged": DecodeExperiment(
        name="decode_onepass_paged",
        family="onepass",
        module="fa_hopper_decode_onepass.py",
        paged=True,
        description="Single-row PackGQA paged decode.",
    ),
    "decode_splitkv_dense": DecodeExperiment(
        name="decode_splitkv_dense",
        family="splitkv",
        module="fa_hopper_decode_splitkv.py",
        paged=False,
        description="Dense decode split over KV blocks with combine.",
    ),
    "decode_splitkv_paged": DecodeExperiment(
        name="decode_splitkv_paged",
        family="splitkv",
        module="fa_hopper_decode_splitkv.py",
        paged=True,
        description="Paged decode split over KV blocks with combine.",
    ),
    "decode_flashdecoding_dense": DecodeExperiment(
        name="decode_flashdecoding_dense",
        family="flashdecoding",
        module="fa_hopper_decode_flashdecoding.py",
        paged=False,
        description="Dense Flash-Decoding-style contiguous split-KV decode.",
    ),
    "decode_flashdecoding_paged": DecodeExperiment(
        name="decode_flashdecoding_paged",
        family="flashdecoding",
        module="fa_hopper_decode_flashdecoding.py",
        paged=True,
        description="Paged Flash-Decoding-style contiguous split-KV decode.",
    ),
    "decode_paged_lb_dense": DecodeExperiment(
        name="decode_paged_lb_dense",
        family="paged_lb",
        module="fa_hopper_decode_paged_lb.py",
        paged=False,
        description="Dense load-balanced round-robin KV-block decode.",
    ),
    "decode_paged_lb_paged": DecodeExperiment(
        name="decode_paged_lb_paged",
        family="paged_lb",
        module="fa_hopper_decode_paged_lb.py",
        paged=True,
        description="Paged load-balanced round-robin KV-block decode.",
    ),
    "decode_seesaw_dense": DecodeExperiment(
        name="decode_seesaw_dense",
        family="seesaw",
        module="fa_hopper_decode_seesaw.py",
        paged=False,
        description="Dense decode nonpersistent staged WS baseline for future seesaw work.",
    ),
    "decode_seesaw_paged": DecodeExperiment(
        name="decode_seesaw_paged",
        family="seesaw",
        module="fa_hopper_decode_seesaw.py",
        paged=True,
        description="Paged decode nonpersistent staged WS baseline for future seesaw work.",
    ),
}

ALIASES: dict[str, tuple[str, ...]] = {
    "all": tuple(EXPERIMENTS),
    "dense": tuple(name for name, exp in EXPERIMENTS.items() if not exp.paged),
    "paged": tuple(name for name, exp in EXPERIMENTS.items() if exp.paged),
    "onepass": ("decode_onepass_dense", "decode_onepass_paged"),
    "splitkv": ("decode_splitkv_dense", "decode_splitkv_paged"),
    "flashdecoding": ("decode_flashdecoding_dense", "decode_flashdecoding_paged"),
    "paged_lb": ("decode_paged_lb_dense", "decode_paged_lb_paged"),
    "seesaw": ("decode_seesaw_dense", "decode_seesaw_paged"),
}


def resolve_experiment_names(selection: list[str] | None) -> tuple[str, ...]:
    requested = selection or ["all"]
    resolved: list[str] = []
    for item in requested:
        names = ALIASES.get(item, (item,))
        for name in names:
            if name not in EXPERIMENTS:
                allowed = ", ".join(sorted((*EXPERIMENTS, *ALIASES)))
                raise KeyError(
                    f"unknown FA3 decode experiment {name!r}; expected one of {allowed}"
                )
            if name not in resolved:
                resolved.append(name)
    return tuple(resolved)


def get_experiment(name: str) -> DecodeExperiment:
    return EXPERIMENTS[name]


def iter_experiments():
    return EXPERIMENTS.values()


__all__ = [
    "DecodeExperiment",
    "EXPERIMENTS",
    "ALIASES",
    "get_experiment",
    "iter_experiments",
    "resolve_experiment_names",
]
