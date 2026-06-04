from __future__ import annotations

from optlab.metrics import mse
from optlab.problems import add_noise, make_phantom
from optlab.solvers import denoise_with_admm, denoise_with_gradient_descent


def test_denoisers_improve_over_noisy_input():
    phantom = make_phantom(48)
    noisy = add_noise(phantom, sigma=0.12, seed=1)

    gd = denoise_with_gradient_descent(noisy, max_iter=30)
    admm = denoise_with_admm(noisy, max_iter=30)

    assert gd.objective_history[-1] <= gd.objective_history[0]

    noisy_error = mse(phantom, noisy)
    assert mse(phantom, admm.image) <= noisy_error
