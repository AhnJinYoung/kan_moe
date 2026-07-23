import os
import unittest
from unittest.mock import patch

from dmoe.gpu_select import (
    GPUInfo,
    GPUProcess,
    configure_cuda_visibility,
    idle_candidates,
    query_gpus,
)


def gpu(
    index: int,
    *,
    used: int,
    free: int,
    utilization: int = 0,
    processes: tuple[GPUProcess, ...] = (),
) -> GPUInfo:
    return GPUInfo(
        index=index,
        uuid=f"GPU-{index}",
        bus_id=f"00000000:{index:02x}:00.0",
        total_memory_mib=81_920,
        used_memory_mib=used,
        free_memory_mib=free,
        utilization_percent=utilization,
        processes=processes,
    )


class GPUSelectTest(unittest.TestCase):
    def test_nvidia_smi_rows_map_compute_processes_to_gpu(self) -> None:
        gpu_output = (
            "0, GPU-zero, 00000000:17:00.0, 81920, 8, 81912, 0\n"
            "1, GPU-one, 00000000:65:00.0, 81920, 12000, 69920, 91\n"
        )
        process_output = (
            "GPU-one, 00000000:65:00.0, 4242, python3, 12000\n"
        )
        with patch(
            "dmoe.gpu_select._run_nvidia_smi",
            side_effect=[gpu_output, process_output],
        ):
            observed = query_gpus()
        self.assertEqual(len(observed), 2)
        self.assertTrue(observed[0].is_process_idle)
        self.assertEqual(observed[1].processes[0].pid, 4242)

    def test_idle_candidates_exclude_compute_processes_and_used_memory(self) -> None:
        busy = GPUProcess(pid=123, name="python", used_memory_mib=10_000)
        candidates = idle_candidates(
            [
                gpu(0, used=0, free=81_920),
                gpu(1, used=0, free=81_920, processes=(busy,)),
                gpu(2, used=2_048, free=79_872),
                gpu(3, used=512, free=81_408),
            ],
            max_used_memory_mib=1_024,
        )
        self.assertEqual([item.index for item in candidates], [0, 3])

    def test_explicit_cuda_visible_devices_is_never_overwritten(self) -> None:
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "3", "LOCAL_RANK": "0"},
            clear=False,
        ):
            selection = configure_cuda_visibility()
            self.assertEqual(selection.mode, "explicit")
            self.assertEqual(selection.visible_devices, "3")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_auto_selection_can_be_disabled(self) -> None:
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        with patch.dict(os.environ, environment, clear=True):
            selection = configure_cuda_visibility(enabled=False)
            self.assertEqual(selection.mode, "disabled")

    def test_single_rank_auto_selection_uses_gpu_uuid(self) -> None:
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment["LOCAL_RANK"] = "0"
        environment["LOCAL_WORLD_SIZE"] = "1"
        selected = gpu(7, used=0, free=81_920)
        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "dmoe.gpu_select.select_and_lock_idle_gpus",
                return_value=[selected],
            ),
        ):
            selection = configure_cuda_visibility()
            self.assertEqual(selection.physical_indices, (7,))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-7")


if __name__ == "__main__":
    unittest.main()
