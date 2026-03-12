"""Base interface for federated LoRA aggregators."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch


class BaseAggregator(ABC):
    """Abstract base for LoRA aggregation methods."""

    name: str = "Base"

    @abstractmethod
    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client LoRA states into a global state."""
        pass

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes for one client round-trip (upload + download)."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # float16 * 2 directions
