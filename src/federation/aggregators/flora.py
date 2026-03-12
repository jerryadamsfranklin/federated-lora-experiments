"""
FLoRA Aggregator: Stacking-based Aggregation

From Wang et al., NeurIPS 2024.
Instead of averaging, stack LoRA modules then compress with SVD.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FLoRAAggregator:
    """
    FLoRA: Stack client LoRAs, compress with SVD.

    - Stack A matrices: [A_1; A_2; ... A_K]
    - Concat B matrices: [B_1, B_2, ..., B_K]
    - Compress using SVD to limit size
    """

    def __init__(self, max_rank: int = 64):
        self.name = "FLoRA"
        self.max_rank = max_rank

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stack and compress client LoRAs.
        """
        if not client_states:
            raise ValueError("No client states")

        aggregated = {}

        # Find A/B pairs (PEFT uses lora_A / lora_B)
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k]

        for a_key in a_keys:
            # Find corresponding B key
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in client_states[0]:
                continue

            # Stack A matrices (along dim 0): (r, in) -> (K*r, in)
            a_matrices = [s[a_key] for s in client_states]
            stacked_a = torch.cat(a_matrices, dim=0)

            # Concat B matrices (along dim 1): (out, r) -> (out, K*r)
            b_matrices = [s[b_key] for s in client_states]
            stacked_b = torch.cat(b_matrices, dim=1)

            # Compress if needed
            if stacked_a.shape[0] > self.max_rank:
                # Compute BA product (out x in)
                ba = stacked_b.float() @ stacked_a.float()

                # SVD
                U, S, Vh = torch.linalg.svd(ba, full_matrices=False)

                # Truncate
                r = min(self.max_rank, S.shape[0])
                U = U[:, :r]
                S = S[:r]
                Vh = Vh[:r, :]

                # Reconstruct A (r x in), B (out x r)
                sqrt_s = torch.sqrt(S)
                new_b = (U * sqrt_s.unsqueeze(0)).to(b_matrices[0].dtype)
                new_a = (sqrt_s.unsqueeze(1) * Vh).to(a_matrices[0].dtype)

                aggregated[a_key] = new_a
                aggregated[b_key] = new_b
            else:
                aggregated[a_key] = stacked_a
                aggregated[b_key] = stacked_b

        return aggregated

    def get_communication_cost(
        self,
        state: Dict[str, torch.Tensor],
        num_clients: int = 1,
    ) -> int:
        """Bytes transmitted."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # Approximate
