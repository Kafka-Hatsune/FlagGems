# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared MetaX device property queries and device context management."""

from functools import lru_cache

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import get_device_properties


class MetaXDeviceInfo:
    """Device capacities and context access, without eager GPU queries.

    The global instance follows the current device. for_device() returns a
    cached instance bound to an explicit logical device index, so its resource
    attributes remain consistent when the current device changes. Capacities
    are queried lazily and cached per device.
    """

    def __init__(self, device=None):
        self._device_index = None if device is None else self.device_index(device)

    @staticmethod
    def current_device():
        return torch_device_fn.current_device()

    def device_index(self, device=None):
        if device is None:
            return self.current_device()
        if isinstance(device, int):
            return device
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"Expected a MetaX CUDA device, got {device}")
        return self.current_device() if device.index is None else device.index

    def for_device(self, device=None):
        return self._for_device(self.device_index(device))

    @classmethod
    @lru_cache(None)
    def _for_device(cls, device_index):
        return cls(device_index)

    @property
    def sm_count(self) -> int:
        return self._properties.multi_processor_count

    @property
    def shared_bytes(self) -> int:
        """Maximum shared-memory bytes per CTA."""
        return self._properties.shared_memory_per_block

    @property
    def l2_bytes(self) -> int:
        return self._properties.L2_cache_size

    @property
    def _properties(self):
        return self._get_properties(self.device_index(self._device_index))

    @staticmethod
    @lru_cache(None)
    def _get_properties(device_index):
        # Reuse the framework's backend-aware query. Errors propagate instead
        # of replacing missing resources with another vendor's defaults.
        return get_device_properties(device_index)

    @staticmethod
    def use_device(device):
        return torch_device_fn.device(device)


# Shared by MetaX operators; device-dependent initialization remains lazy.
device_info = MetaXDeviceInfo()
