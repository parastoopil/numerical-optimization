from __future__ import annotations

import numpy as np


def gradient(image: np.ndarray) -> np.ndarray:
    """Forward finite differences with periodic boundary conditions."""

    image = np.asarray(image, dtype=float)
    vertical = np.roll(image, -1, axis=0) - image
    horizontal = np.roll(image, -1, axis=1) - image
    return np.stack((vertical, horizontal), axis=0)


def divergence(field: np.ndarray) -> np.ndarray:
    """Adjoint of :func:`gradient` under the periodic inner product."""

    field = np.asarray(field, dtype=float)
    if field.shape[0] != 2:
        raise ValueError("field must have shape (2, height, width)")

    vertical = field[0]
    horizontal = field[1]
    return np.roll(vertical, 1, axis=0) - vertical + np.roll(horizontal, 1, axis=1) - horizontal


def tv_norm(image: np.ndarray, epsilon: float = 0.0) -> float:
    grad = gradient(image)
    magnitude = np.sqrt(np.sum(grad * grad, axis=0) + float(epsilon) ** 2)
    return float(np.sum(magnitude))


def smoothed_tv_gradient(image: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    grad = gradient(image)
    magnitude = np.sqrt(np.sum(grad * grad, axis=0) + float(epsilon) ** 2)
    return divergence(grad / magnitude)


def adjoint_gap(image: np.ndarray, field: np.ndarray) -> float:
    lhs = np.vdot(gradient(image), field)
    rhs = np.vdot(image, divergence(field))
    return float(abs(lhs - rhs))
