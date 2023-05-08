from itertools import product
from typing import Optional

import numpy as np
import pytest
import xarray as xr

from srs_tools import BackgroundEstimator


@pytest.fixture
def dataset(
    S: int = 3,
    T: int = 5,
    im_shape: tuple[int, int] = (512, 512),
    seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed=seed)

    full_size = (S, T, *im_shape)
    yy, xx = np.meshgrid(np.arange(im_shape[0]), np.arange(im_shape[1]))

    noise = rng.normal(scale=10, size=full_size).astype("f4")
    noise = xr.DataArray(noise, dims=list("STYX"))

    r = 5
    A = 40
    labels = xr.DataArray(np.zeros(full_size, dtype="u2"), dims=list("STYX"))
    signal = xr.zeros_like(noise)
    # Question: maybe the objects should be fixed across T?
    for s, t in product(range(S), range(T)):
        single_labels = BackgroundEstimator._make_cv_labels(np.zeros(im_shape))
        labels.data[s, t] = single_labels

        for val in np.unique(single_labels)[1:]:
            idx = np.nonzero(single_labels == val)
            dy = idx[0] - np.median(idx[0])
            dx = idx[1] - np.median(idx[1])
            signal.data[(s, t, *idx)] = A * np.exp(-(dy * dy + dx * dx) / (2 * r * r))

    lscale = rng.normal(loc=im_shape[0] / 4, scale=im_shape[0] / 4, size=2)
    bkgd_scale = 10
    bkgd_offset = 20
    bkgd = bkgd_offset + bkgd_scale * np.sin(xx / lscale[0] + yy / lscale[1])
    bkgd = xr.DataArray(bkgd, dims=list("YX"))

    test = signal + bkgd + noise

    return test, labels, signal, bkgd, noise


def test_constructor(dataset):
    test, labels, signal, bkgd, noise = dataset

    be = BackgroundEstimator(test, labels)

    assert isinstance(be, BackgroundEstimator)
