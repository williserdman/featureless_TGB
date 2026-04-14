from __future__ import annotations

import torch
from torch import Tensor


def transform_event_message(
    *,
    msg: Tensor,
    mode: str,
    noise_std: float,
    noise_seed: int,
    event_index: int,
) -> Tensor:
    if mode == "full":
        return msg

    if mode != "gaussian_noise":
        raise ValueError(f"Unsupported feature mode: {mode}")

    # Deterministic per-event noise so repeated runs are reproducible.
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(int(noise_seed) + int(event_index))
    noise_cpu = torch.randn(msg.shape, generator=cpu_gen, dtype=msg.dtype)
    noise = noise_cpu.to(msg.device) * noise_std
    return noise
