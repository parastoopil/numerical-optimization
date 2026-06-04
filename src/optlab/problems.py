from __future__ import annotations

import numpy as np


def make_phantom(size: int = 96) -> np.ndarray:
    """Build a synthetic piecewise-smooth image for denoising demos."""

    grid = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")

    image = 0.12 + 0.15 * (xx + 1.0) / 2.0 + 0.08 * (yy + 1.0) / 2.0

    ellipses = [
        (((xx + 0.30) / 0.32) ** 2 + ((yy + 0.15) / 0.22) ** 2 <= 1.0, 0.72),
        (((xx - 0.32) / 0.18) ** 2 + ((yy - 0.30) / 0.16) ** 2 <= 1.0, 0.94),
        (((xx + 0.05) / 0.13) ** 2 + ((yy - 0.05) / 0.13) ** 2 <= 1.0, 0.58),
    ]
    for mask, value in ellipses:
        image[mask] = value

    bars = [
        ((np.abs(xx + 0.55) < 0.14) & (np.abs(yy - 0.45) < 0.07), 0.84),
        ((np.abs(xx - 0.55) < 0.12) & (np.abs(yy + 0.42) < 0.09), 0.42),
    ]
    for mask, value in bars:
        image[mask] = value

    return np.clip(image, 0.0, 1.0)


def add_noise(image: np.ndarray, sigma: float = 0.08, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = np.asarray(image, dtype=float) + sigma * rng.standard_normal(image.shape)
    return np.clip(noisy, 0.0, 1.0)
