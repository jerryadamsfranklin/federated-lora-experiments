"""
FlexLoRA Aggregator: SVD-based Weight Redistribution

From Bai et al., NeurIPS 2024.
Compute full ΔW = Σ(B_k @ A_k), then decompose back to low-rank.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FlexLoRAAggregator:
    """
    FlexLoRA: SVD-based aggregation.

    1. Compute ΔW = Σ w_k * (B_k @ A_k) for all clients
    2. Apply SVD to get low-rank approximation
    3. Redistribute to clients
    """

    def __init__(self, global_rank: int = 32):
        self.name = "FlexLoRA"
        self.global_rank = global_rank

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using SVD on weight products.
        """
        if not client_states:
            raise ValueError("No client states")

        # Default weights
        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)
        total = sum(weights)
        weights = [w / total for w in weights]

        aggregated = {}

        # Find A/B pairs
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k]

        for a_key in a_keys:
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in client_states[0]:
                continue

            # Compute weighted sum of BA products
            accumulated = None
            for w, state in zip(weights, client_states):
                a = state[a_key].float()
                b = state[b_key].float()
                update = b @ a

                if accumulated is None:
                    accumulated = w * update
                else:
                    accumulated += w * update

            # SVD decomposition
            U, S, Vh = torch.linalg.svd(accumulated, full_matrices=False)

            # Truncate to global_rank
            r = min(self.global_rank, S.shape[0])
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]

            # Reconstruct A and B
            sqrt_s = torch.sqrt(S)
            new_b = (U * sqrt_s.unsqueeze(0)).to(
                client_states[0][b_key].dtype
            )
            new_a = (sqrt_s.unsqueeze(1) * Vh).to(
                client_states[0][a_key].dtype
            )

            aggregated[a_key] = new_a
            aggregated[b_key] = new_b

        return aggregated

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes transmitted."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2
