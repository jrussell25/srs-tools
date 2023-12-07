import sys
from time import perf_counter

import microutil as mu
import xarray as xr
from aicsimageio.readers import TiffGlobReader
from cellpose.models import Cellpose
from dask.distributed import Client

from srs_tools import io

if __name__ == "__main__":
    # parse arguments - tiff_path, output_dataset_path, fov_id
    tiff_path, dataset_path, fov = sys.argv[1:]

    position_slice = slice(int(fov), int(fov) + 1)

    # start a dask thread-based cluster
    client = Client(processes=False)

    # validate output_path - initialize from process 0
    # out_path = validate_output_path(out_path)

    # build indexer
    indexer = io.get_srs_indexer(tiff_path)

    # construct readers -> zarr dataset
    reader = TiffGlobReader(indexer["filenames"], indexer[list("STCZ")])
    reader.set_scene(int(fov))

    n_scenes = len(reader.scenes)

    imgs = reader.xarray_dask_data.drop_vars("C")
    # imgs["C"] = ["fluo", "bf", "srs"] # cant have coords if we want to write regions
    # separately as we do below

    mu.save_dataset(
        imgs.to_dataset(name="images"),
        dataset_path,
        position_slice,
        scene_size=n_scenes,
    )

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
        batch_size=128,  # believe this does not actually do anything?
        channels=channels,
        diameter=15,
        flow_threshold=0.6,
        cellprob_threshold=-2,
        normalize=False,
        tile=False,
    )

    print("Saving Cellpose outputs", flush=True)

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

    mu.save_dataset(seg_imgs.to_dataset(name="nuc"), position_slice=position_slice)

    image_list = [im.data for im in seg_imgs]

    masks, flows, styles = model.cp.eval(
        image_list,
        batch_size=128,  # believe this does not actually do anything?
        channels=channels,
        diameter=8,
        flow_threshold=0.75,
        cellprob_threshold=-10,
        normalize=False,
        tile=False,
    )
    print("Saving Cellpose outputs", flush=True)

    cp_ds = mu.cellpose.make_cellpose_dataset(masks, flows, styles)
    cp_ds = cp_ds.rename({x: "nuc_" + x for x in list(cp_ds)})
    mu.save_dataset(cp_ds, dataset_path, position_slice)

    t1 = perf_counter()
    print(f"Cellpose nuc segmentation complete -- {t1-t0:0.2f} seconds", flush=True)

    ##########
    # BTRACK #
    ##########

    print("Starting btrack", flush=True)
    t0 = perf_counter()

    config_file = "particle_config.json"
    updated_masks = mu.btrack.gogogo_btrack(
        cp_ds["cp_masks"].data,
        config_file,
        15,
        intensity_image=seg_imgs.data,
        properties=("intensity_mean", "intensity_max", "area"),
    )

    bt_ds = xr.Dataset({"labels": xr.DataArray(updated_masks, dims=list("TYX"))})
    mu.save_dataset(bt_ds, dataset_path, position_slice)

    t1 = perf_counter()
    print(f"Cell tracking complete -- {t1-t0:0.2f} seconds", flush=True)
    print("Done", flush=True)
