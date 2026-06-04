from __future__ import annotations

import numpy as np


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return float(np.mean((reference - estimate) ** 2))


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    error = mse(reference, estimate)
    if error == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(error))
