"""
FedIT Aggregator: Baseline FedAvg on LoRA

Simply averages A and B matrices independently across clients.
This is the baseline method from Zhang et al., ICASSP 2024.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FedITAggregator:
    """
    FedIT: Standard FedAvg on LoRA parameters.

    Aggregation rule:
        global_A = Σ (w_k * A_k)
        global_B = Σ (w_k * B_k)

    where w_k is weight for client k.
    """

    def __init__(self):
        self.name = "FedIT"

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client states using weighted average.

        Args:
            client_states: List of LoRA state dicts from clients
            weights: Optional weights (default: uniform)

        Returns:
            Aggregated global state dict
        """
        if not client_states:
            raise ValueError("No client states to aggregate")

        # Default to uniform weights
        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Aggregate each parameter
        aggregated = {}
        for name in client_states[0].keys():
            weighted_sum = sum(
                w * state[name].float()
                for w, state in zip(weights, client_states)
            )
            aggregated[name] = weighted_sum.to(client_states[0][name].dtype)

        return aggregated

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes for one client round-trip."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # float16 * 2 (upload + download)
