import os
import platform
import logging
from typing import Dict, Any, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def detect_compute_device(requested_device: str = "auto") -> Tuple[str, Dict[str, Any]]:
    """
    Detects and selects the optimal available compute accelerator.
    
    Priority order for 'auto':
    1. CUDA GPU (NVIDIA)
    2. Other platform accelerators (XPU/ROCm/MPS)
    3. CPU fallback
    
    Returns:
        device_type: e.g. 'cuda', 'cuda:0', 'mps', 'cpu'
        device_info: Dictionary containing device capabilities and metadata
    """
    device_info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "device_name": "CPU",
        "device_count": 0,
        "precision_support": {
            "fp16": False,
            "bf16": False
        }
    }

    if not TORCH_AVAILABLE:
        return "cpu", device_info

    if torch.cuda.is_available():
        device_info["cuda_available"] = True
        device_info["device_count"] = torch.cuda.device_count()
        device_info["device_name"] = torch.cuda.get_device_name(0)
        device_info["precision_support"]["fp16"] = True
        device_info["precision_support"]["bf16"] = torch.cuda.is_bf16_supported()

    if requested_device == "auto":
        if torch.cuda.is_available():
            chosen = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chosen = "mps"
        else:
            chosen = "cpu"
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            chosen = "cuda"
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            chosen = "cpu"
    elif requested_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chosen = "mps"
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            chosen = "cpu"
    else:
        chosen = requested_device

    device_info["selected_device"] = chosen
    return chosen, device_info
