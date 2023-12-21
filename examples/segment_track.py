import sys
from time import perf_counter

import microutil as mu
import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from cellpose.models import Cellpose
from fast_overlap import overlap
from skimage.morphology import disk
from skimage.util import map_array

if __name__ == "__main__":
    t_start = perf_counter()

    #############
    # READ DATA #
    #############

    t0 = perf_counter()

    dataset_path = sys.argv[1]
    fov = int(sys.argv[2])

    position_slice = slice(fov, fov + 1)

    imgs = xr.open_zarr(dataset_path)["images"].isel(S=fov)

    missing_data = (imgs == 0).all(list("YX"))

    if np.any(missing_data.data):
        print("Missing image data in frames:", flush=True)
        missing_coords = np.nonzero(missing_data.data)
        for x, c in zip("TCZ", missing_coords):
            print(f"{x}: {c}", flush=True)

    t1 = perf_counter()
    print(f"Data loaded -- {t1-t0:0.2f} seconds", flush=True)

    ############
    # CELLPOSE #
    ############

    print("Starting Cellpose cyto segmentation", flush=True)

    t0 = perf_counter()

    model = Cellpose(model_type="cyto2", gpu=True)
    channels = [[0, 0]]

    seg_imgs = imgs.isel(C=1)
    seg_imgs = (seg_imgs - seg_imgs.min(list("YX"))) / (
        seg_imgs.max(list("YX")) - seg_imgs.min(list("YX"))
    )
    seg_imgs.load()

    image_list = [im for im in seg_imgs.data.reshape(-1, *seg_imgs.shape[-2:])]

    masks, flows, styles = model.cp.eval(
        image_list,
        batch_size=1024,  # believe this does not actually do anything?
        channels=channels,
        diameter=15,
        flow_threshold=0.6,
        cellprob_threshold=-2,
        normalize=False,
        tile=False,
    )

    cp_ds0 = mu.cellpose.make_cellpose_dataset(masks, flows, styles)

    # mu function only handles TYX images (plus the cellpose style dimension)
    # reshape and rebuild the dataset to include the Z dimension
    cp_ds = xr.Dataset()
    for var in cp_ds0:
        arr = cp_ds0[var]
        cp_ds["cyto_" + var] = xr.DataArray(
            cp_ds0[var].data.reshape(
                seg_imgs.sizes["T"], seg_imgs.sizes["Z"], *arr.shape[1:]
            ),
            dims=["T", "Z", *arr.dims[1:]],
        )
    # mu.save_dataset(cp_ds, dataset_path, position_slice)
    cp_ds[["cyto_cp_masks"]].expand_dims("S").to_zarr(
        dataset_path, region={"S": position_slice}
    )
    cyto_cp_masks = cp_ds["cyto_cp_masks"]  # need this later

    t1 = perf_counter()
    print(f"Cellpose cyto segmentation complete -- {t1-t0:0.2f} seconds", flush=True)

    print("Starting Cellpose nuclear segmentation", flush=True)

    t0 = perf_counter()

    model = Cellpose(model_type="nuclei", gpu=True)
    channels = [[0, 0]]

    seg_imgs = xr.open_zarr(dataset_path)["nuc"].isel(S=fov).load()

    image_list = [im.data for im in seg_imgs]

    masks, flows, styles = model.cp.eval(
        image_list,
        batch_size=512,  # believe this does not actually do anything?
        channels=channels,
        diameter=8,
        flow_threshold=0.75,
        cellprob_threshold=-10,
        normalize=False,
        tile=False,
    )

    cp_ds = mu.cellpose.make_cellpose_dataset(masks, flows, styles)
    cp_ds = cp_ds.rename({x: "nuc_" + x for x in list(cp_ds)})
    cp_ds[["nuc_cp_masks"]].expand_dims("S").to_zarr(
        dataset_path, region={"S": position_slice}
    )

    t1 = perf_counter()
    print(f"Cellpose nuc segmentation complete -- {t1-t0:0.2f} seconds", flush=True)

    ########################
    # CLASSICAL CORRECTION #
    ########################

    print("Correcting cellpose nuclear segmenation")
    t0 = perf_counter()
    thresh_masks = xr.open_zarr(dataset_path)["thresh_mask"].isel(S=fov).load()

    labels = np.copy(cp_ds["nuc_cp_masks"].data)
    for i, (m_cp, m_thresh) in enumerate(
        zip(cp_ds["nuc_cp_masks"].data, thresh_masks.data)
    ):
        offset = np.max(m_cp)
        missing = ndi.binary_opening(m_thresh & (m_cp == 0), structure=disk(2))
        missing_labels, _ = ndi.label(missing, output="u2")
        missing_labels[missing_labels > 0] += offset
        labels[i] += missing_labels

    mask_ds = (
        xr.DataArray(labels, dims=list("TYX"))
        .to_dataset(name="combo_masks")
        .expand_dims("S")
    )
    mask_ds.to_zarr(dataset_path, region={"S": position_slice})

    t1 = perf_counter()
    print(f"Finished post-processing nuclear masks -- {t1-t0:0.2f}", flush=True)

    ##########
    # BTRACK #
    ##########

    print("Starting btrack", flush=True)
    t0 = perf_counter()

    config_file = "particle_config.json"
    updated_masks = mu.btrack.gogogo_btrack(
        labels,
        config_file,
        15,
        intensity_image=seg_imgs.data,
        properties=("intensity_mean", "intensity_max", "area"),
    )

    bt_ds = xr.Dataset(
        {"labels": xr.DataArray(updated_masks, dims=list("TYX"))}
    ).expand_dims("S")
    bt_ds.to_zarr(dataset_path, region={"S": position_slice})

    t1 = perf_counter()
    print(f"Cell tracking complete -- {t1-t0:0.2f} seconds", flush=True)

    ######################
    # CYTO-NUC ALIGNMENT #
    ######################

    print("Starting cyto-nuc alignment", flush=True)
    t0 = perf_counter()

    z_slices = (cyto_cp_masks > 0).sum(list("YX")).argmax("Z")
    cyto_labels = cyto_cp_masks.isel(Z=z_slices).copy().data
    for t in range(imgs.sizes["T"]):
        nuc = bt_ds.labels.data[t]
        cyto = cyto_labels[t]
        cyto_ids = np.unique(cyto)

        # compute overlapse and pare the array down to eliminate rows from missing cells
        o = overlap(cyto_labels[t], nuc)[cyto_ids]
        map_array(cyto_labels[t], cyto_ids, o.argmax(-1), out=cyto_labels[t])

    cyto_labels_ds = xr.zeros_like(bt_ds[["labels"]]).rename({"labels": "cyto_labels"})
    cyto_labels_ds["cyto_labels"].data = cyto_labels
    cyto_labels_ds.expand_dims("S").to_zarr(dataset_path, region={"S": position_slice})

    t1 = perf_counter()
    print(f"Alignment complete -- {t1-t0:0.2f} seconds", flush=True)

    print(f"Preprocessing complete -- Total time {t1-t_start:0.2f} seconds", flush=True)
