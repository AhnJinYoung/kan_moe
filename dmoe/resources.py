from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_GIB = 1024**3


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _read_positive_int(path: str) -> int | None:
    value = _read_text(path)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _host_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def _cgroup_cpu_quota() -> float | None:
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass
    quota = _read_positive_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_positive_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and period is not None:
        return quota / period
    return None


def _cgroup_memory_limit(host_memory: int | None) -> int | None:
    memory_max = _read_text("/sys/fs/cgroup/memory.max")
    if memory_max and memory_max != "max":
        try:
            limit = int(memory_max)
        except ValueError:
            limit = 0
        if limit > 0 and (host_memory is None or limit < host_memory):
            return limit
    limit = _read_positive_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None and (host_memory is None or limit < host_memory):
        return limit
    return None


def _cgroup_memory_current() -> int | None:
    return _read_positive_int("/sys/fs/cgroup/memory.current") or _read_positive_int(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )


@dataclass(frozen=True)
class ResourceLimits:
    affinity_cpus: int
    cgroup_cpu_quota: float | None
    effective_cpus: int
    host_memory_bytes: int | None
    cgroup_memory_limit_bytes: int | None
    cgroup_memory_current_bytes: int | None
    cpu_threads: int
    data_workers: int
    tokenizer_batch_limit: int
    parquet_batch_limit: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "host_memory_bytes",
            "cgroup_memory_limit_bytes",
            "cgroup_memory_current_bytes",
        ):
            value = result[key]
            result[key.replace("_bytes", "_gib")] = (
                round(value / _GIB, 3) if value is not None else None
            )
        return result


def detect_resource_limits() -> ResourceLimits:
    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpus = os.cpu_count() or 1
    cpu_quota = _cgroup_cpu_quota()
    quota_cpus = (
        max(1, math.floor(cpu_quota)) if cpu_quota is not None else affinity_cpus
    )
    effective_cpus = max(1, min(affinity_cpus, quota_cpus))
    host_memory = _host_memory_bytes()
    memory_limit = _cgroup_memory_limit(host_memory)
    memory_budget = memory_limit or host_memory

    # Pretraining is GPU-bound. One or two CPU threads are enough for bounded
    # Parquet reads and fast-tokenizer calls, and avoid pod-level fork/thread
    # explosions.
    cpu_threads = 1 if effective_cpus <= 4 else 2
    data_workers = 1
    if memory_budget is not None and memory_budget <= 8 * _GIB:
        tokenizer_batch_limit = 1
        parquet_batch_limit = 1
    elif memory_budget is not None and memory_budget <= 16 * _GIB:
        tokenizer_batch_limit = 2
        parquet_batch_limit = 2
    elif memory_budget is not None and memory_budget <= 32 * _GIB:
        tokenizer_batch_limit = 4
        parquet_batch_limit = 4
    else:
        tokenizer_batch_limit = 4
        parquet_batch_limit = 4
    return ResourceLimits(
        affinity_cpus=affinity_cpus,
        cgroup_cpu_quota=cpu_quota,
        effective_cpus=effective_cpus,
        host_memory_bytes=host_memory,
        cgroup_memory_limit_bytes=memory_limit,
        cgroup_memory_current_bytes=_cgroup_memory_current(),
        cpu_threads=cpu_threads,
        data_workers=data_workers,
        tokenizer_batch_limit=tokenizer_batch_limit,
        parquet_batch_limit=parquet_batch_limit,
    )


def _cap_thread_environment(name: str, limit: int) -> None:
    current = os.environ.get(name)
    try:
        requested = int(current) if current is not None else limit
    except ValueError:
        requested = limit
    os.environ[name] = str(max(1, min(requested, limit)))


def configure_conservative_cpu_runtime() -> ResourceLimits:
    limits = detect_resource_limits()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        _cap_thread_environment(variable, limits.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return limits


def configure_torch_threads(torch_module: Any, limits: ResourceLimits) -> None:
    torch_module.set_num_threads(limits.cpu_threads)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work.
        pass
