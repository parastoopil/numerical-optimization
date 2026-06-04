"""Optimization lab utilities for TV denoising."""

from .metrics import mse, psnr
from .operators import divergence, gradient, tv_norm
from .problems import add_noise, make_phantom
from .solvers import SolverResult, denoise_with_accelerated_gradient, denoise_with_admm, denoise_with_gradient_descent

__all__ = [
    "SolverResult",
    "add_noise",
    "denoise_with_accelerated_gradient",
    "denoise_with_admm",
    "denoise_with_gradient_descent",
    "divergence",
    "gradient",
    "make_phantom",
    "mse",
    "psnr",
    "tv_norm",
]
