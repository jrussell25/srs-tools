import warnings
from typing import Optional

import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from scipy.spatial import cKDTree

# from skimage.segmentation import expand_labels
from skimage.draw import disk as draw_disk
from skimage.morphology import disk

__all__ = ["BackgroundEstimator"]


class BackgroundEstimator:
    """
    Dataset subclass for estimating the background in a segmented
    SRS imaging dataset.
    """

    def __init__(
        self,
        images: xr.DataArray,
        labels: xr.DataArray,
        dilation_radius: int = 3,
        dilation_iters: int = 3,
    ):
        super().__init__()

        self.images = images
        self.labels = labels

        self.cv_labels = None
        self.initial_estimate = None
        self.background_estimate = None

        self.dilation_structure = disk(dilation_radius)
        self.dilation_iters = dilation_iters

    def check_dims(self) -> None:
        # TODO
        raise NotImplementedError

    def to_dataset(self):
        # TODO
        pass

    def run(self, sigma: np.ndarray[int] = None, use_cv_labels=False) -> None:
        """
        Run the entire background estimation pipeline
        """

        mask = self.labels > 0

        if use_cv_labels:
            if self.cv_labels is None:
                raise ValueError(
                    "CV labels not found. To estimate background"
                    "without CV labels set use_cv_labels=False."
                    "Otherwise initialize CV labels with"
                    "BackgroundEstimator.make_cv_labels."
                )
            else:
                mask = mask | (self.cv_labels > 0)

    def make_cv_labels(
        self, recompute=False, N: int = 7, r: int = 6, seed: Optional[int] = None
    ):
        if self.cv_labels is None or recompute:
            cv_labels = xr.apply_ufunc(
                self._make_cv_labels,
                self.labels,
                N,
                r,
                vectorize=True,
                input_core_dims=[["Y", "X"], [], []],
                output_core_dims=[["Y", "X"]],
                dask="parallelized",
                output_dtypes=["u2"],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )

            self.cv_labels = cv_labels
        else:
            pass

    @staticmethod
    def _make_cv_labels(
        labels: np.ndarray, N: int = 7, r: int = 6, seed: Optional[int] = None
    ):
        """
        Draw a quasi-uniform grid of disks skipping regions that
        already contain objects as defined in labels.
        """
        shape = labels.shape
        M = np.min(shape)
        rng = np.random.default_rng(seed)

        if np.unique(shape).shape[0] != 1:
            warnings.warn(
                f"Images are not square, deteriming cv spacing based on"
                f"smaller dimension with size {np.max(shape)}"
            )
        spacing = M / (N + 1)
        points = np.arange(spacing // 2, 512, spacing).astype(int)
        expanded_labels = ndi.binary_dilation(labels, disk(3), iterations=2)
        cv_grid_centers = np.stack(np.meshgrid(points, points)).reshape(2, -1).T
        cv_offsets = (spacing / 2 - r) * (
            2 * rng.uniform(size=cv_grid_centers.shape) - 1
        )
        cv_centers = cv_grid_centers + cv_offsets
        cv_labels = np.zeros_like(labels, dtype="u2")
        for i, (y, x) in enumerate(cv_centers):
            inds = draw_disk((y, x), r, shape=shape)
            if expanded_labels[inds[0], inds[1]].sum() == 0:
                cv_labels[inds[0], inds[1]] = i + 1
        return cv_labels

    def make_initial_estimate(
        self, use_cv_labels: bool = False, recompute: bool = False
    ):
        if self.initial_estimate is None or recompute:

            if use_cv_labels:
                if self.cv_labels is None:
                    raise ValueError(
                        "CV labels not found. To estimate background"
                        "without CV labels set use_cv_labels=False."
                        "Otherwise initialize CV labels with"
                        "BackgroundEstimator.make_cv_labels."
                    )
                mask = (self.labels > 0) | (self.cv_labels > 0)
            else:
                mask = self.labels > 0
            bkgd_init = xr.apply_ufunc(
                self._make_initial_estimate,
                self.images,
                mask,
                vectorize=True,
                dask="parallelized",
                input_core_dims=[list("SYX"), list("SYX")],
                output_core_dims=[list("SYX")],
                output_dtypes=["f4"],
                dask_gufunc_kwargs={"allow_rechunk": True},
            )
            self.initial_estimate = bkgd_init

    @staticmethod
    def _make_initial_estimate(images: np.ndarray, labels: np.ndarray):
        """
        imgs: np.ndarray with dims corresponding to SYX
        """
        assert images.ndim == 3
        dilated_labels = np.array(
            [
                ndi.binary_dilation(arr > 0, structure=disk(3), iterations=3)
                for arr in labels
            ]
        )
        nan_masked = np.array(images)
        nan_masked[dilated_labels] = np.nan
        scales = np.nanmedian(nan_masked, axis=(-1, -2), keepdims=True)
        nan_masked = nan_masked / scales
        bkgd_init_est = np.nanmedian(nan_masked, axis=0) * scales

        if np.any(np.isnan(bkgd_init_est)):
            isnan = np.all(np.isnan(bkgd_init_est), axis=0)
            positions = np.array(np.nonzero(~isnan)).T
            nan_inds = np.array(np.nonzero(isnan)).T

            kdt = cKDTree(positions)
            dists, neighbors = kdt.query(nan_inds, k=7)
            neighbor_inds = positions[neighbors]
            bkgd_init_est[:, nan_inds[:, 0], nan_inds[:, 1]] = bkgd_init_est[
                :, neighbor_inds[..., 0], neighbor_inds[..., 1]
            ].mean(-1)

        return bkgd_init_est

    def lpf_estimate(self, recompute=False):
        if self.background_estimate is None or recompute:
            bkgd = xr.apply_ufunc(
                self._lpf,
                self.images,
                self.labels,
                self.initial_estimate,
                vectorize=True,
                input_core_dims=[
                    list("YX"),
                    list("YX"),
                    list("YX"),
                ],
                output_core_dims=[list("YX")],
                dask="parallelized",
                dask_gufunc_kwargs={"allow_rechunk": True},
            )
            self.background_estimate = bkgd

    @staticmethod
    def _lpf(im: np.ndarray, labels: np.ndarray, init: np.ndarray, sigma=(3, 24)):
        mask2d = ndi.binary_dilation(labels > 0, structure=disk(3), iterations=(3))
        arr = np.array(im)
        arr[mask2d] = init[mask2d]  # np.median(yin[mask])
        out = ndi.gaussian_filter(arr, sigma=sigma)
        return out
