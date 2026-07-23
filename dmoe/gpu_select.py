from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


DEFAULT_MAX_IDLE_MEMORY_MIB = 1_024
DEFAULT_SELECTION_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    name: str
    used_memory_mib: int


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    bus_id: str
    total_memory_mib: int
    used_memory_mib: int
    free_memory_mib: int
    utilization_percent: int
    processes: tuple[GPUProcess, ...] = ()

    @property
    def is_process_idle(self) -> bool:
        return not self.processes


@dataclass(frozen=True)
class GPUSelection:
    mode: str
    physical_indices: tuple[int, ...] = ()
    visible_devices: str = ""
    detail: str = ""

    @property
    def auto_selected(self) -> bool:
        return self.mode == "auto"


_HELD_GPU_LOCKS: list[TextIO] = []


def _integer(value: str, default: int = 0) -> int:
    match = re.search(r"-?\d+", value)
    return int(match.group(0)) if match else default


def _csv_rows(output: str) -> list[list[str]]:
    return [
        [field.strip() for field in row]
        for row in csv.reader(output.splitlines())
        if row and any(field.strip() for field in row)
    ]


def _run_nvidia_smi(query: str) -> str:
    command = [
        "nvidia-smi",
        f"--query-{query}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout


def query_gpus() -> list[GPUInfo]:
    gpu_rows = _csv_rows(
        _run_nvidia_smi(
            "gpu=index,uuid,pci.bus_id,memory.total,memory.used,"
            "memory.free,utilization.gpu"
        )
    )
    process_rows = _csv_rows(
        _run_nvidia_smi(
            "compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_gpu_memory"
        )
    )

    current_pid = os.getpid()
    processes_by_uuid: dict[str, list[GPUProcess]] = {}
    processes_by_bus: dict[str, list[GPUProcess]] = {}
    for row in process_rows:
        if len(row) < 5 or not row[0].startswith(("GPU-", "MIG-")):
            continue
        process = GPUProcess(
            pid=_integer(row[2], -1),
            name=row[3],
            used_memory_mib=_integer(row[4]),
        )
        if process.pid == current_pid:
            continue
        processes_by_uuid.setdefault(row[0], []).append(process)
        processes_by_bus.setdefault(row[1].lower(), []).append(process)

    gpus: list[GPUInfo] = []
    for row in gpu_rows:
        if len(row) < 7:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
        gpu_processes = processes_by_uuid.get(row[1])
        if gpu_processes is None:
            gpu_processes = processes_by_bus.get(row[2].lower(), [])
        gpus.append(
            GPUInfo(
                index=_integer(row[0], -1),
                uuid=row[1],
                bus_id=row[2],
                total_memory_mib=_integer(row[3]),
                used_memory_mib=_integer(row[4]),
                free_memory_mib=_integer(row[5]),
                utilization_percent=_integer(row[6]),
                processes=tuple(gpu_processes),
            )
        )
    return gpus


def idle_candidates(
    gpus: list[GPUInfo], max_used_memory_mib: int
) -> list[GPUInfo]:
    candidates = [
        gpu
        for gpu in gpus
        if gpu.is_process_idle and gpu.used_memory_mib <= max_used_memory_mib
    ]
    return sorted(
        candidates,
        key=lambda gpu: (gpu.free_memory_mib, -gpu.utilization_percent),
        reverse=True,
    )


def _lock_path(gpu: GPUInfo) -> Path:
    safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", gpu.uuid)
    return Path(tempfile.gettempdir()) / f"dmoe-gpu-{safe_uuid}.lock"


def _try_lock_gpu(gpu: GPUInfo) -> TextIO | None:
    handle = _lock_path(gpu).open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} selected_at={time.time():.6f}\n")
    handle.flush()
    return handle


def _format_gpu_state(gpus: list[GPUInfo]) -> str:
    rows = []
    for gpu in sorted(gpus, key=lambda item: item.index):
        process_text = (
            ",".join(f"{item.pid}:{item.name}" for item in gpu.processes)
            if gpu.processes
            else "none"
        )
        rows.append(
            f"gpu={gpu.index} free={gpu.free_memory_mib}/{gpu.total_memory_mib}MiB "
            f"used={gpu.used_memory_mib}MiB util={gpu.utilization_percent}% "
            f"compute_processes={process_text}"
        )
    return "; ".join(rows)


def select_and_lock_idle_gpus(
    count: int, max_used_memory_mib: int = DEFAULT_MAX_IDLE_MEMORY_MIB
) -> list[GPUInfo]:
    if count <= 0:
        raise ValueError("GPU count must be positive")
    first_snapshot = query_gpus()
    selected: list[GPUInfo] = []
    locks: list[TextIO] = []
    for gpu in idle_candidates(first_snapshot, max_used_memory_mib):
        lock = _try_lock_gpu(gpu)
        if lock is None:
            continue
        selected.append(gpu)
        locks.append(lock)
        if len(selected) == count:
            break

    if len(selected) != count:
        for lock in locks:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        raise RuntimeError(
            f"requested {count} idle CUDA device(s), but only found "
            f"{len(selected)}. {_format_gpu_state(first_snapshot)}"
        )

    # Close the small race between the nvidia-smi snapshot and lock acquisition.
    second_snapshot = {gpu.uuid: gpu for gpu in query_gpus()}
    no_longer_idle: list[GPUInfo] = []
    missing_uuids: list[str] = []
    for gpu in selected:
        current = second_snapshot.get(gpu.uuid)
        if current is None:
            missing_uuids.append(gpu.uuid)
        elif (
            not current.is_process_idle
            or current.used_memory_mib > max_used_memory_mib
        ):
            no_longer_idle.append(current)
    if no_longer_idle or missing_uuids:
        for lock in locks:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        missing_text = (
            f" missing UUIDs={','.join(missing_uuids)};" if missing_uuids else ""
        )
        raise RuntimeError(
            "a selected CUDA device disappeared or became busy during reservation:"
            + missing_text
            + " "
            + _format_gpu_state(no_longer_idle)
        )

    _HELD_GPU_LOCKS.extend(locks)
    return selected


def _coordination_path() -> Path:
    identity = "|".join(
        [
            str(os.getuid()),
            str(os.getppid()),
            os.environ.get("MASTER_ADDR", "localhost"),
            os.environ.get("MASTER_PORT", "none"),
            os.environ.get("TORCHELASTIC_RUN_ID", "none"),
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"dmoe-gpu-selection-{digest}.json"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_rank_zero_selection(path: Path, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for local rank 0 GPU selection: {path}")


def _selection_message(selection: GPUSelection) -> str:
    if selection.auto_selected:
        indices = ",".join(str(index) for index in selection.physical_indices)
        return (
            f"auto-selected idle physical CUDA device(s): [{indices}]; "
            f"CUDA_VISIBLE_DEVICES={selection.visible_devices}"
        )
    if selection.mode == "explicit":
        return (
            "using explicit CUDA_VISIBLE_DEVICES="
            f"{selection.visible_devices!r}; automatic selection skipped"
        )
    return selection.detail


def configure_cuda_visibility(
    *,
    enabled: bool = True,
    max_used_memory_mib: int | None = None,
    timeout_seconds: float = DEFAULT_SELECTION_TIMEOUT_SECONDS,
) -> GPUSelection:
    """Set CUDA_VISIBLE_DEVICES before PyTorch imports or initializes CUDA."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        selection = GPUSelection(
            mode="explicit",
            visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        )
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(_selection_message(selection), flush=True)
        return selection
    if not enabled:
        return GPUSelection(
            mode="disabled",
            detail="automatic CUDA selection disabled by command-line flag",
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
    )
    maximum_used = (
        int(os.environ.get("DMOE_GPU_MAX_USED_MEMORY_MIB", DEFAULT_MAX_IDLE_MEMORY_MIB))
        if max_used_memory_mib is None
        else max_used_memory_mib
    )

    try:
        if local_world_size == 1:
            selected = select_and_lock_idle_gpus(1, maximum_used)
            payload: dict[str, object] = {
                "gpus": [asdict(gpu) for gpu in selected],
            }
        else:
            coordination_path = _coordination_path()
            if local_rank == 0:
                try:
                    selected = select_and_lock_idle_gpus(
                        local_world_size, maximum_used
                    )
                    payload = {"gpus": [asdict(gpu) for gpu in selected]}
                except Exception as error:
                    payload = {"error": f"{type(error).__name__}: {error}"}
                    _write_json_atomic(coordination_path, payload)
                    raise
                _write_json_atomic(coordination_path, payload)
            else:
                payload = _read_rank_zero_selection(
                    coordination_path, timeout_seconds
                )
                if "error" in payload:
                    raise RuntimeError(str(payload["error"]))
            selected = [
                GPUInfo(
                    **{
                        **raw,
                        "processes": tuple(
                            GPUProcess(**item) for item in raw["processes"]
                        ),
                    }
                )
                for raw in payload["gpus"]  # type: ignore[index,union-attr]
            ]
    except FileNotFoundError:
        return GPUSelection(
            mode="unavailable",
            detail="nvidia-smi was not found; CUDA auto-selection was not applied",
        )
    except subprocess.CalledProcessError as error:
        message = (error.stderr or error.stdout or str(error)).strip()
        if "no devices were found" in message.lower():
            return GPUSelection(mode="unavailable", detail=message)
        raise RuntimeError(f"nvidia-smi query failed: {message}") from error

    visible_devices = ",".join(gpu.uuid for gpu in selected)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    selection = GPUSelection(
        mode="auto",
        physical_indices=tuple(gpu.index for gpu in selected),
        visible_devices=visible_devices,
    )
    if local_rank == 0:
        print(_selection_message(selection), flush=True)
    return selection
