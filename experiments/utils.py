"""
Utility functions for experiments.
Includes seed management, device configuration, and helper functions.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = True) -> dict:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (may impact performance)

    Returns:
        Dictionary with seed information
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # PyTorch backends
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Fallback for older PyTorch versions
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    seed_info = {
        'seed': seed,
        'deterministic': deterministic,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        seed_info['cuda_version'] = torch.version.cuda
        seed_info['cudnn_version'] = torch.backends.cudnn.version()

    return seed_info


def get_device(device: str = "auto") -> torch.device:
    """
    Get PyTorch device.

    Args:
        device: Device specification ('cuda', 'cpu', 'auto')

    Returns:
        PyTorch device object
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(device)

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        torch_device = torch.device("cpu")

    return torch_device


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'total_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }


def worker_init_fn(worker_id: int, seed: int = 42):
    """
    Initialize DataLoader worker with proper seed.

    Args:
        worker_id: Worker ID
        seed: Base seed
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
