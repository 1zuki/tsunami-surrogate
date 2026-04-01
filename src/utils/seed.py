from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    return seed


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    def _init_fn(worker_id: int) -> None:
        seed_worker(worker_id, base_seed)

    return _init_fn


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
