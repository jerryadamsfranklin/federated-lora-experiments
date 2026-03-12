"""
Data Partitioning for Federated Learning

This module splits datasets among clients for federated learning simulation.
Supports:
- IID: Each client gets random sample (ideal case)
- Label Skew: Clients have different label distributions (realistic)
- Quantity Skew: Clients have different amounts of data (realistic)

CURSOR AI: Implement this file with all methods as specified.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset


class DataPartitioner:
    """
    Partition dataset for federated learning simulation.

    Example:
        partitioner = DataPartitioner(dataset, num_clients=10, seed=42)

        # IID split
        client_datasets = partitioner.iid_partition()

        # Non-IID label skew
        client_datasets = partitioner.label_skew_partition(alpha=0.5)

        # Non-IID quantity skew
        client_datasets = partitioner.quantity_skew_partition(alpha=0.3)
    """

    def __init__(self, dataset: Dataset, num_clients: int, seed: int = 42):
        """
        Initialize partitioner.

        Args:
            dataset: HuggingFace Dataset to partition
            num_clients: Number of clients
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)

    def iid_partition(self) -> List[Dataset]:
        """
        IID partition: random uniform split.

        Each client gets len(dataset)/num_clients samples randomly.
        All clients have similar data distributions.
        """
        indices = np.random.permutation(len(self.dataset))
        splits = np.array_split(indices, self.num_clients)

        client_datasets = []
        for split_idx in splits:
            client_datasets.append(self.dataset.select(split_idx.tolist()))

        return client_datasets

    def label_skew_partition(
        self,
        label_column: str = "label",
        alpha: float = 0.5,
    ) -> List[Dataset]:
        """
        Non-IID partition with label distribution skew.

        Uses Dirichlet distribution to assign different proportions of
        each label to each client.

        Args:
            label_column: Column containing labels
            alpha: Dirichlet concentration (lower = more skew)
                   alpha=0.1: extreme skew (each client gets ~1 label)
                   alpha=0.5: moderate skew (recommended)
                   alpha=10: nearly IID
        """
        # Group indices by label
        label_to_indices = defaultdict(list)
        for idx, example in enumerate(self.dataset):
            label = example[label_column]
            label_to_indices[label].append(idx)

        # Initialize client index lists
        client_indices = [[] for _ in range(self.num_clients)]

        # Distribute each label's data according to Dirichlet
        for label, indices in label_to_indices.items():
            np.random.shuffle(indices)

            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            counts = (proportions * len(indices)).astype(int)
            counts[-1] = len(indices) - counts[:-1].sum()  # Fix rounding

            # Assign to clients
            start = 0
            for client_id, count in enumerate(counts):
                client_indices[client_id].extend(indices[start : start + count])
                start += count

        # Create datasets
        client_datasets = []
        for indices in client_indices:
            np.random.shuffle(indices)
            client_datasets.append(self.dataset.select(indices))

        return client_datasets

    def quantity_skew_partition(
        self,
        alpha: float = 0.5,
        min_samples: int = 10,
    ) -> List[Dataset]:
        """
        Non-IID partition with quantity skew.

        Different clients get different amounts of data.

        Args:
            alpha: Dirichlet concentration (lower = more imbalanced)
            min_samples: Minimum samples per client
        """
        total = len(self.dataset)

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * self.num_clients)

        # Ensure minimum samples
        base = np.array([min_samples] * self.num_clients)
        remaining = total - base.sum()
        if remaining <= 0:
            # Not enough data; give min_samples to each and split rest
            indices = np.random.permutation(total)
            splits = np.array_split(indices, self.num_clients)
            return [self.dataset.select(s.tolist()) for s in splits]
        extra = (proportions * remaining).astype(int)
        extra[-1] = remaining - extra[:-1].sum()

        sizes = base + extra

        # Shuffle and split
        indices = np.random.permutation(total)
        client_datasets = []
        start = 0
        for size in sizes:
            client_datasets.append(
                self.dataset.select(indices[start : start + size].tolist())
            )
            start += size

        return client_datasets

    def get_stats(self, client_datasets: List[Dataset]) -> Dict:
        """Get partition statistics."""
        sizes = [len(d) for d in client_datasets]
        return {
            "num_clients": len(client_datasets),
            "total_samples": sum(sizes),
            "min_samples": min(sizes),
            "max_samples": max(sizes),
            "mean_samples": float(np.mean(sizes)),
            "std_samples": float(np.std(sizes)),
        }
