import sys
from time import perf_counter

import microutil as mu
import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from aicsimageio.readers import TiffGlobReader
from cellpose.models import Cellpose
from dask.distributed import Client
from skimage.morphology import disk

from srs_tools import io

if __name__ == "__main__":
    t_start = perf_counter()

    #############
    # READ DATA #
    #############
    t0 = perf_counter()

    tiff_path, dataset_path, fov = sys.argv[1:]
    print(f"Reading data from disk - Processing FOV {fov}", flush=True)

    position_slice = slice(int(fov), int(fov) + 1)

    client = Client(processes=False)

    indexer = io.get_srs_indexer(tiff_path)

    reader = TiffGlobReader(indexer["filenames"], indexer[list("STCZ")])
    reader.set_scene(int(fov))

    n_scenes = len(reader.scenes)

    imgs = reader.xarray_dask_data.drop_vars("C")
    # imgs["C"] = ["fluo", "bf", "srs"] # cant have coords if we want to write regions

    mu.save_dataset(
        imgs.to_dataset(name="images"),
        dataset_path,
        position_slice,
        scene_size=n_scenes,
    )

    t1 = perf_counter()
    print(f"Data loaded -- {t1-t0:0.2f} seconds", flush=True)

    ############
    # CELLPOSE #
    ############

    print("Starting Cellpose cyto segmentation", flush=True)

    t0 = perf_counter()

    model = Cellpose(model_type="cyto2", gpu=True)
    channels = [[0, 0]]

    seg_imgs = imgs.isel(C=1).mean("Z")
    seg_imgs = (seg_imgs - seg_imgs.min(list("YX"))) / (
        seg_imgs.max(list("YX")) - seg_imgs.min(list("YX"))
    )
    seg_imgs.load()

    image_list = [im.data for im in seg_imgs]

    masks, flows, styles = model.cp.eval(
        image_list,
        batch_size=512,  # believe this does not actually do anything?
        channels=channels,
        diameter=15,
        flow_threshold=0.6,
        cellprob_threshold=-2,
        normalize=False,
        tile=False,
    )

    cp_ds = mu.cellpose.make_cellpose_dataset(masks, flows, styles)
    cp_ds = cp_ds.rename({x: "cyto_" + x for x in list(cp_ds)})
    mu.save_dataset(cp_ds, dataset_path, position_slice)

    t1 = perf_counter()
    print(f"Cellpose cyto segmentation complete -- {t1-t0:0.2f} seconds", flush=True)

    print("Starting Cellpose nuclear segmentation", flush=True)

    t0 = perf_counter()

    model = Cellpose(model_type="nuclei", gpu=True)
    channels = [[0, 0]]

    seg_imgs = imgs.isel(C=0).max("Z")
    seg_imgs = mu.normalize_fluo(seg_imgs, mode_cutoff_ratio=1e9)

    seg_imgs.load()

    mu.save_dataset(
        seg_imgs.to_dataset(name="nuc"), dataset_path, position_slice=position_slice
    )

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
    mu.save_dataset(cp_ds, dataset_path, position_slice)

    t1 = perf_counter()
    print(f"Cellpose nuc segmentation complete -- {t1-t0:0.2f} seconds", flush=True)

    ########################
    # CLASSICAL CORRECTION #
    ########################

    print("Correcting cellpose nuclear segmenation")
    t0 = perf_counter()
    thresh = mu.calc_thresholds(seg_imgs)
    thresh_masks = seg_imgs > (0.5 * thresh)

    missing_nuc = xr.zeros_like(cp_ds["nuc_cp_masks"])
    labels = np.copy(cp_ds["nuc_cp_masks"].data)
    for i, (m_cp, m_thresh) in enumerate(
        zip(cp_ds["nuc_cp_masks"].data, thresh_masks.data)
    ):
        offset = np.max(m_cp)
        missing = ndi.binary_opening(m_thresh & (m_cp == 0), structure=disk(2))
        missing_labels, _ = ndi.label(missing, output="u2")
        missing_nuc.data[i] = missing_labels
        missing_labels[missing_labels > 0] += offset
        labels[i] += missing_labels

    mask_ds = xr.DataArray(labels, dims=list("TYX")).to_dataset(name="combo_masks")
    mask_ds["thresh_masks"] = thresh_masks
    mask_ds["missing_nuc"] = missing_nuc
    mu.save_dataset(mask_ds, dataset_path, position_slice)

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

    bt_ds = xr.Dataset({"labels": xr.DataArray(updated_masks, dims=list("TYX"))})
    mu.save_dataset(bt_ds, dataset_path, position_slice)

    t1 = perf_counter()
    print(f"Cell tracking complete -- {t1-t0:0.2f} seconds", flush=True)

    # TODO - Match cyto labels up with cell labels
    # TODO - Single cell quantities (avg, com, area)

    print(f"Preprocessing complete -- Total time {t1-t_start:0.2f} seconds", flush=True)
