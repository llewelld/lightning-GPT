import torch
import torchvision
import intel_extension_for_pytorch as ipex
from typing import Any, MutableSequence, Tuple, Union, Dict, Optional, List

#from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.device_parser import _check_data_type
from lightning.pytorch.utilities import rank_zero_info

class XPUAccelerator(Accelerator):
    """Accelerator for Intel XPU devices."""

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise ValueError(f"Device should be XPU, got {device} instead.")
        if torch.xpu.get_fp32_math_mode() == torch.xpu.FP32MathMode.FP32:
            rank_zero_info(
                f"You are using an XPU device ({torch.xpu.get_device_name(device)!r}). To properly utilize computation "
                "power, you can set `torch.xpu.set_fp32_math_mode(mode=torch.xpu.FP32MathMode.FP32, device='xpu')` "
                "which will trade-off precision for performance. For more details, read https://intel.github.io/"
                "intel-extension-for-pytorch/xpu/latest/tutorials/api_doc.html#torch.xpu.set_fp32_math_mode"
            )

        torch.xpu.set_device(device)

    def teardown(self) -> None:
        torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        xpus = devices
        _check_data_type(xpus)

        # Convert strings to ints
        if not isinstance(xpus, str):
            xpus = xpus
        elif xpus == "-1":
            xpus =  -1
        elif "," in xpus:
            xpus = [int(x.strip()) for x in xpus.split(",") if len(x) > 0]
        else:
            xpus = int(xpus.strip())

        available_xpus = list(range(torch.xpu.device_count()))
        if isinstance(xpus, (MutableSequence, tuple)):
            xpus = list(xpus)
        elif not xpus:
            xpus = None
        elif xpus == -1:
            xpus = available_xpus
        else:
            xpus = list(range(xpus))

        if not xpus:
            raise MisconfigurationException("xpus requested but none are available.")

        for gpu in xpus:
            if gpu not in available_xpus:
                raise MisconfigurationException(
                    f"You requested gpu: {xpus}\n But your machine only has: {available_xpus}"
                )

        return xpus


    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return torch.xpu.memory_stats(device)

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description=cls.__class__.__name__,
        )
