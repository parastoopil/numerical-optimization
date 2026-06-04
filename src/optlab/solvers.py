from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .operators import divergence, gradient, smoothed_tv_gradient, tv_norm


@dataclass
class SolverResult:
    method: str
    image: np.ndarray
    objective_history: list[float]
    residual_history: list[float]
    step_history: list[float]
    iterations: int
    converged: bool
    runtime_seconds: float


def rof_objective(image: np.ndarray, noisy: np.ndarray, lam: float) -> float:
    return float(0.5 * np.sum((image - noisy) ** 2) + lam * tv_norm(image))


def smooth_rof_objective(image: np.ndarray, noisy: np.ndarray, lam: float, epsilon: float) -> float:
    return float(0.5 * np.sum((image - noisy) ** 2) + lam * tv_norm(image, epsilon=epsilon))


def smooth_rof_gradient(image: np.ndarray, noisy: np.ndarray, lam: float, epsilon: float) -> np.ndarray:
    return image - noisy + lam * smoothed_tv_gradient(image, epsilon=epsilon)


def _backtracking_step(
    current: np.ndarray,
    direction: np.ndarray,
    objective,
    initial_step: float,
    shrink: float,
    armijo: float,
) -> tuple[np.ndarray, float, float]:
    current_objective = objective(current)
    grad_norm_sq = float(np.sum(direction * direction))
    step = float(initial_step)

    while True:
        candidate = current - step * direction
        candidate_objective = objective(candidate)
        if candidate_objective <= current_objective - armijo * step * grad_norm_sq or step < 1e-10:
            return candidate, step, candidate_objective
        step *= shrink


def denoise_with_gradient_descent(
    noisy: np.ndarray,
    lam: float = 0.08,
    epsilon: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-5,
    initial_step: float = 1.0,
) -> SolverResult:
    x = np.asarray(noisy, dtype=float).copy()
    objective = lambda value: smooth_rof_objective(value, noisy, lam, epsilon)
    objective_history: list[float] = []
    residual_history: list[float] = []
    step_history: list[float] = []
    converged = False

    start = perf_counter()
    for iteration in range(max_iter):
        value = objective(x)
        grad = smooth_rof_gradient(x, noisy, lam, epsilon)
        grad_norm = float(np.linalg.norm(grad))

        objective_history.append(value)
        residual_history.append(grad_norm)

        if grad_norm <= tol * (1.0 + np.linalg.norm(noisy)):
            converged = True
            break

        x, step, _ = _backtracking_step(x, grad, objective, initial_step, 0.5, 1e-4)
        step_history.append(step)

    runtime_seconds = perf_counter() - start
    return SolverResult(
        method="gradient_descent",
        image=x,
        objective_history=objective_history,
        residual_history=residual_history,
        step_history=step_history,
        iterations=len(objective_history),
        converged=converged,
        runtime_seconds=runtime_seconds,
    )


def denoise_with_accelerated_gradient(
    noisy: np.ndarray,
    lam: float = 0.08,
    epsilon: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-5,
    initial_step: float = 1.0,
) -> SolverResult:
    x = np.asarray(noisy, dtype=float).copy()
    y = x.copy()
    momentum = 1.0
    objective = lambda value: smooth_rof_objective(value, noisy, lam, epsilon)
    objective_history: list[float] = []
    residual_history: list[float] = []
    step_history: list[float] = []
    converged = False

    start = perf_counter()
    for iteration in range(max_iter):
        value = objective(y)
        grad = smooth_rof_gradient(y, noisy, lam, epsilon)
        grad_norm = float(np.linalg.norm(grad))

        objective_history.append(value)
        residual_history.append(grad_norm)

        if grad_norm <= tol * (1.0 + np.linalg.norm(noisy)):
            x = y.copy()
            converged = True
            break

        x_next, step, _ = _backtracking_step(y, grad, objective, initial_step, 0.5, 1e-4)
        step_history.append(step)

        momentum_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        y = x_next + ((momentum - 1.0) / momentum_next) * (x_next - x)
        x = x_next
        momentum = momentum_next

    runtime_seconds = perf_counter() - start
    return SolverResult(
        method="accelerated_gradient",
        image=x,
        objective_history=objective_history,
        residual_history=residual_history,
        step_history=step_history,
        iterations=len(objective_history),
        converged=converged,
        runtime_seconds=runtime_seconds,
    )


def _laplacian_eigenvalues(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    ky = 2.0 * np.pi * np.fft.fftfreq(height)
    kx = 2.0 * np.pi * np.fft.fftfreq(width)
    return (2.0 - 2.0 * np.cos(ky))[:, None] + (2.0 - 2.0 * np.cos(kx))[None, :]


def _solve_fft(rhs: np.ndarray, rho: float, eigenvalues: np.ndarray) -> np.ndarray:
    rhs_hat = np.fft.fft2(rhs)
    solution_hat = rhs_hat / (1.0 + rho * eigenvalues)
    return np.fft.ifft2(solution_hat).real


def _shrink_isotropic(field: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = np.sqrt(np.sum(field * field, axis=0))
    scale = np.maximum(1.0, magnitude / threshold)
    return field / scale


def denoise_with_admm(
    noisy: np.ndarray,
    lam: float = 0.08,
    rho: float = 1.0,
    max_iter: int = 150,
    tol: float = 1e-4,
) -> SolverResult:
    x = np.asarray(noisy, dtype=float).copy()
    z = gradient(x)
    u = np.zeros_like(z)
    eigenvalues = _laplacian_eigenvalues(x.shape)

    objective_history: list[float] = []
    residual_history: list[float] = []
    step_history: list[float] = []
    converged = False

    start = perf_counter()
    for iteration in range(max_iter):
        rhs = noisy + rho * divergence(z - u)
        x = _solve_fft(rhs, rho, eigenvalues)

        gx = gradient(x)
        z = _shrink_isotropic(gx + u, lam / rho)
        u = u + gx - z

        primal_residual = float(np.linalg.norm(gx - z))
        objective_value = rof_objective(x, noisy, lam)

        objective_history.append(objective_value)
        residual_history.append(primal_residual)
        step_history.append(rho)

        if primal_residual <= tol * (1.0 + np.linalg.norm(gx)):
            converged = True
            break

    runtime_seconds = perf_counter() - start
    return SolverResult(
        method="admm",
        image=x,
        objective_history=objective_history,
        residual_history=residual_history,
        step_history=step_history,
        iterations=len(objective_history),
        converged=converged,
        runtime_seconds=runtime_seconds,
    )
