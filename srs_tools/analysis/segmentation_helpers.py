import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr
from fast_overlap import overlap
from skimage.morphology import disk
from skimage.util import map_array


def correct_cellpose_with_thresh(
    cp_masks: xr.DataArray, thresh_masks: xr.DataArray
) -> xr.DataArray:
    """
    Add segmentation masks for regions missed by cellpose but found via thresholding.

    Parameters
    ----------
    cp_masks: xr.DataArray (dims=TYX)
        Segmentation masks predicted by cellpose
    thresh_masks: xr.DataArray (dim=TYX)
        Thresholded image.

    Returns
    -------
    labels: xr.DataArray
        Updated segmentation masks
    """
    labels = cp_masks.copy()
    for i, (m_cp, m_thresh) in enumerate(zip(cp_masks.data, thresh_masks.data)):
        offset = np.max(m_cp)
        missing = ndi.binary_opening(m_thresh & (m_cp == 0), structure=disk(2))
        missing_labels, _ = ndi.label(missing, output="u2")
        missing_labels[missing_labels > 0] += offset
        labels.data[i] += missing_labels

    return labels


def squash_3d_segmentation(
    cyto_masks_3d: xr.DataArray, nuc_labels: xr.DataArray, nuc_coms: pd.DataFrame
) -> xr.DataArray:
    """
    Parameters
    ----------
    cyto_masks_3d: xr.DataArray dims=TZYX
    nuc_labels : xr.DataArray dims=TYX
    nuc_coms: pd.DataFrame
        Must have MultiIndex levels (T, CellID) and columns X and Y

    Returns
    -------
    cyto_labels: xr.DataArray dims=TYX
    """
    # compute naive squash of masks - picking z plane with most segmented area
    z_slices = (cyto_masks_3d > 0).sum(list("YX")).argmax("Z")
    cyto_labels = cyto_masks_3d.isel(Z=z_slices).copy()

    # regions where a cell was called in one z plane but not the max plane
    missed = (cyto_masks_3d > 0).any("Z").where(cyto_labels == 0, other=0).astype("u2")

    for t in range(cyto_masks_3d.sizes["T"]):
        mask = missed.data[t]
        nuc_idx = nuc_coms.loc[t][["Y", "X"]].round().astype(int)

        nuc_pts_in_missing = nuc_idx.loc[mask[tuple(nuc_idx.values.T)] == 1]

        for nuc_id, yx in nuc_pts_in_missing.iterrows():
            nuc_column_cyto_labels = cyto_masks_3d.data[
                t, :, nuc_labels.data[t] == (nuc_id + 1)
            ].T

            best_zi, best_id, max_overlap = (-1, -1, -1)
            for zi, x in enumerate(nuc_column_cyto_labels):
                vals, counts = np.unique(x, return_counts=True)
                nz_ovlps = counts[vals > 0]
                if nz_ovlps.size > 0:
                    if nz_ovlps.max() > max_overlap:
                        best_id = vals[vals > 0][nz_ovlps.argmax()]
                        max_overlap = nz_ovlps.max()
                        best_zi = zi
            fill_mask = cyto_masks_3d.data[t, best_zi] == best_id

            # we dont want to mess up existing labels too much
            # only fill in the new one if it mostly corresponds to a missing region
            if (cyto_labels.data[t, fill_mask] > 0).mean() < 0.5:
                cyto_labels.data[t, fill_mask] = cyto_labels.data[t].max() + 1
    return cyto_labels


def _match_labels(nuc_labels: np.ndarray, cyto_labels: np.ndarray) -> np.ndarray:
    """
    Map between nuclear and cytoplasmic segmentation regions by finding the nuclei
    which maximally overlap with each cell.

    Parameters
    ----------
    nuc_labels: np.ndarray
        Single segmented image of nuclei
    cyto_labels: np.ndarray
        Single segmented image of cells

    Returns
    --------
    matched: np.ndarray
        Array containing cellular segmentations relabeled to match
        nuclear labels. NOTE cells that do not contain a segmented nucleus
        will be eliminated.
    """
    cyto_ids = np.unique(cyto_labels)
    o = overlap(cyto_labels.copy(), nuc_labels.copy())[cyto_ids]
    return map_array(cyto_labels.copy(), cyto_ids.copy(), o.argmax(-1).copy())


def match_labels(cyto_labels: xr.DataArray, nuc_labels: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(
        _match_labels,
        cyto_labels,
        nuc_labels,
        input_core_dims=[list("YX"), list("YX")],
        output_core_dims=[list("YX")],
        vectorize=True,
        dask="parallelized",
    )
