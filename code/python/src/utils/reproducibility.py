# ========================= Presented by: Reza Saadatyar (2026) ================================
# ============================= E-mail: R.Saadatyar90@gmail.com ==================================

import os
import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=True, verbose=True):
    """
    Set random seeds for Python, NumPy and PyTorch.

    Args:
        seed          (int): Random seed.
        deterministic(bool): Request deterministic PyTorch/CUDA behaviour.
        verbose       (bool): Print the selected seed.

    Returns:
        int: The seed used by the experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"✅ Random seed set to {seed}")

    return seed


def seed_worker(worker_id):
    """
    Seed each PyTorch DataLoader worker from PyTorch's initial worker seed.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
