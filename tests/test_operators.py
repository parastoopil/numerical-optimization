from __future__ import annotations

import numpy as np

from optlab.operators import adjoint_gap, divergence, gradient, tv_norm


def test_gradient_and_divergence_are_adjoint():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((8, 9))
    field = rng.standard_normal((2, 8, 9))
    assert adjoint_gap(image, field) < 1e-10


def test_tv_norm_is_zero_on_constant_image():
    image = np.ones((6, 6))
    assert tv_norm(image) == 0.0


def test_divergence_shape_matches_image_shape():
    image = np.zeros((5, 7))
    field = gradient(image)
    div = divergence(field)
    assert div.shape == image.shape
