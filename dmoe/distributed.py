from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank, local_rank, world_size, device, distributed)


def seed_everything(seed: int, rank: int = 0) -> None:
    value = seed + rank * 10_007
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def all_reduce_sum(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_objects(value: Any, context: DistributedContext) -> list[Any]:
    if not context.distributed:
        return [value]
    values: list[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(values, value)
    return values


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])

