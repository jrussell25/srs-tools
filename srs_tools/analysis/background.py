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

        self._cv_labels = None
        self._dilated_mask = None
        self._initial_estimate = None
        self._background_estimate = None

        self.dilation_structure = disk(dilation_radius)
        self.dilation_iters = dilation_iters

        self.sigma_opt = None

    def check_dims(self) -> None:
        # TODO
        raise NotImplementedError

    def to_dataset(self):
        """
        Repackage all currently computed arrays into an xarray dataset for
        saving or downstream use.
        """
        ds = xr.Dataset()
        ds["images"] = self.images
        ds["labels"] = self.labels
        ds["cv_labels"] = self.cv_labels
        if self.initial_estimate is not None:
            ds["initial_estimate"] = self.initial_estimate
        if self.background_estimate is not None:
            ds["background_estimate"] = self.background_estimate

        return ds

    def run(self, sigma: Optional[np.ndarray] = None, use_cv_labels=False) -> None:
        """
        Run the entire background estimation pipeline
        """

        if use_cv_labels:
            self.make_cv_labels()

        if self.sigma_opt is None and sigma is None:
            warnings.warn(
                "No sigma provided and no sigma_opt found. Running "
                "grid search to determine optimal kernel parameters."
            )

        self.sigma_scan()

        self.lpf(self.sigma_opt)

    def sigma_scan(
        self,
        sigma_list: Optional[list[int]] = None,
        n_samples: Optional[int] = None,
        return_mse_array=False,
        force_dask=True,
    ) -> None:

        # set up sigma grid to search
        if sigma_list is None:
            sigma_list = np.arange(2, 128, 8)

        sigma_grid = np.array(np.meshgrid(sigma_list, sigma_list))
        sigmas = xr.DataArray(sigma_grid.reshape(2, -1).T, dims=["sigma", "k"])
        if force_dask:
            sigmas = sigmas.chunk({"sigma": 1, "k": -1})

        # setup input data to test
        if n_samples is None:
            warnings.warn(
                "n_samples not provided. Running grid search on full "
                "dataset which can be quite slow. Using 10 or fewer "
                "samples is recommended "
            )

            imgs = self.images
            mask = self.dilated_mask
            initial = self.initial_estimate

        else:
            idx = np.random.default_rng().integers(
                self.images.sizes["T"], size=n_samples
            )
            imgs = self.images.isel(T=idx)
            mask = self.dilated_mask.isel(T=idx)
            initial = self.initial_estimate.isel(T=idx)

        # lazily estimate background for all sigma values
        cv_bkgd = xr.apply_ufunc(
            self._lpf,
            imgs,
            mask,
            initial,
            sigmas,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[list("YX"), list("YX"), list("YX"), ["k"]],
            output_core_dims=[list("YX")],
            output_dtypes=["f4"],
        )
        diff_imgs = cv_bkgd - imgs
        sq_errs = (diff_imgs * diff_imgs).transpose("sigma", ...)
        mse = sq_errs.where(self.cv_labels > 0).mean(self.images.dims)

        mse.load()

        self.sigma_opt = sigmas[mse.data.argmin()]

        if return_mse_array:
            return mse.data.reshape(sigma_grid.shape[1:]).T

    # CV labels
    @property
    def cv_labels(self):
        if self._cv_labels is None:
            self.make_cv_labels()
        return self._cv_labels

    def make_cv_labels(self, N: int = 7, r: int = 6, seed: Optional[int] = None):
        self._cv_labels = xr.apply_ufunc(
            self._make_cv_labels,
            self.labels,
            N,
            r,
            vectorize=True,
            input_core_dims=[list("YX"), [], []],
            output_core_dims=[list("YX")],
            dask="parallelized",
            output_dtypes=["u2"],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

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

    # Initial estimate
    @property
    def initial_estimate(self):
        if self._initial_estimate is None:
            if self._cv_labels is None:
                self.make_initial_estimate(use_cv_labels=False, recompute=True)
            else:
                self.make_initial_estimate(use_cv_labels=True, recompute=True)
        return self._initial_estimate

    def make_initial_estimate(
        self, use_cv_labels: bool = False, recompute: bool = False
    ):
        self._initial_estimate = xr.apply_ufunc(
            self._make_initial_estimate,
            self.images,
            self.dilated_mask,
            vectorize=True,
            dask="parallelized",
            input_core_dims=[list("SYX"), list("SYX")],
            output_core_dims=[list("SYX")],
            output_dtypes=["f4"],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

    @staticmethod
    def _make_initial_estimate(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        imgs: np.ndarray with dims corresponding to SYX
        """
        assert images.ndim == 3
        nan_masked = np.array(images)
        nan_masked[mask] = np.nan
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

    @property
    def dilated_mask(self):
        if self._dilated_mask is None:
            use_cv_labels = self._cv_labels is not None
            self.make_dilated_mask(use_cv_labels)
        return self._dilated_mask

    def make_dilated_mask(self, use_cv_labels=False) -> xr.DataArray:
        inpt = self.labels > 0
        if use_cv_labels:
            inpt = inpt | (self.cv_labels > 0)

        self._dilated_mask = xr.apply_ufunc(
            self._make_dilated_mask,
            inpt,
            input_core_dims=[list("YX")],
            output_core_dims=[list("YX")],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[bool],
            dask_gufunc_kwargs={"allow_rechunk": True},
            kwargs={"structure": self.dilation_structure, "iters": self.dilation_iters},
        )

    @staticmethod
    def _make_dilated_mask(
        im: np.ndarray, structure: np.ndarray, iters: int
    ) -> np.ndarray:

        return ndi.binary_dilation(im, structure=structure, iterations=iters)

    # LPF
    @property
    def background_estimate(self):
        if self._background_estimate is None:
            self.lpf()
        return self._background_estimate

    def lpf(self, sigma: xr.DataArray):

        self._background_estimate = xr.apply_ufunc(
            self._lpf,
            self.images,
            self.labels,
            self.initial_estimate,
            sigma,
            vectorize=True,
            input_core_dims=[
                list("YX"),
                list("YX"),
                list("YX"),
                list("k"),
            ],
            output_core_dims=[list("YX")],
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=["f4"],
        )

    @staticmethod
    def _lpf(im: np.ndarray, labels: np.ndarray, init: np.ndarray, sigma=(3, 24)):
        mask2d = ndi.binary_dilation(labels > 0, structure=disk(3), iterations=(3))
        arr = np.array(im)
        arr[mask2d] = init[mask2d]  # np.median(yin[mask])
        out = ndi.gaussian_filter(arr, sigma=sigma)
        return out