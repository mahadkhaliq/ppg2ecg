"""Utility functions: seeding, logging, IO helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Trade off: deterministic algorithms can be slow. Keep cuDNN benchmarking on for speed.
    torch.backends.cudnn.benchmark = True


def get_logger(name: str = "ppg2ecg") -> logging.Logger:
    """Stdout logger with timestamps. Idempotent."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str | Path) -> None:
    """Save dict as JSON with reasonable defaults for numpy types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=default)


def device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    """Simple context-manager timer. `with Timer() as t:` then read `t.elapsed`."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
