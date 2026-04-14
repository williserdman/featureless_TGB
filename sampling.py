from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SamplingReport:
    strategy: str
    original_events: int
    kept_events: int

    @property
    def retention(self) -> float:
        if self.original_events == 0:
            return 0.0
        return self.kept_events / self.original_events

    @property
    def speedup_estimate(self) -> float:
        if self.kept_events == 0:
            return float("inf")
        return self.original_events / self.kept_events


def temporal_subsample_indices(
    *,
    split_mask: Tensor,
    timestamps: Tensor,
    strategy: str,
    stride: int,
    ratio: float,
    seed: int,
) -> tuple[Tensor, SamplingReport]:
    """Return selected edge indices for a split based on temporal subsampling.

    Supported strategies:
    - none: no subsampling
    - stride: keep every k-th event in temporal order
    - uniform: random sample events from the split
    """

    edge_idx = torch.where(split_mask)[0]
    original_count = int(edge_idx.numel())

    if strategy == "none" or original_count == 0:
        return edge_idx, SamplingReport("none", original_count, original_count)

    if strategy == "stride":
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        kept = edge_idx[::stride]
        return kept, SamplingReport("stride", original_count, int(kept.numel()))

    if strategy == "uniform":
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        # Sort candidate edges by time, then sample within split to preserve chronology
        split_times = timestamps[edge_idx]
        sorted_pos = torch.argsort(split_times)
        sorted_idx = edge_idx[sorted_pos]

        sample_count = max(1, int(original_count * ratio))
        gen = torch.Generator(device=sorted_idx.device)
        gen.manual_seed(seed)
        sampled_pos = torch.randperm(
            original_count, generator=gen, device=sorted_idx.device
        )[:sample_count]
        sampled = sorted_idx[sampled_pos]
        sampled, _ = torch.sort(sampled)
        return sampled, SamplingReport("uniform", original_count, int(sampled.numel()))

    raise ValueError(f"Unknown temporal sampling strategy: {strategy}")
