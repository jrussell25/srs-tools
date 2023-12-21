import sys

import dask.array as da
import xarray as xr
from aicsimageio.readers import TiffGlobReader
from dask.distributed import Client
from microutil import calc_thresholds, normalize_fluo

from srs_tools.io import get_srs_indexer

if __name__ == "__main__":
    tiff_path, dataset_path = sys.argv[1:]
    print(f"{tiff_path=}", flush=True)
    print(f"{dataset_path=}", flush=True)

    client = Client(processes=False)

    indexer = get_srs_indexer(tiff_path)

    t_cutoff = indexer.groupby("S")["T"].max().min()

    indexer = indexer.loc[indexer["T"] <= t_cutoff]

    reader = TiffGlobReader(indexer["filenames"], indexer[list("STCZ")])

    imgs = reader.get_xarray_dask_stack(scene_character="S").drop_vars("C")

    _ = imgs.to_dataset(name="images").to_zarr(
        dataset_path, consolidated=True, mode="a"
    )

    nuc = normalize_fluo(imgs.isel(C=0).max("Z").astype("f4"), mode_cutoff_ratio=1e9)

    nuc.to_dataset(name="nuc").to_zarr(dataset_path, consolidated=True, mode="a")

    thresh_masks = nuc > (0.5 * calc_thresholds(nuc))

    thresh_masks.to_dataset(name="thresh_mask").to_zarr(dataset_path, mode="a")

    # setup the dataset from this single process

    z4d = xr.DataArray(
        da.zeros(imgs.isel(C=0, Z=0).shape, dtype="u2", chunks=(1, -1, -1, -1)),
        dims=list("STYX"),
    )
    dummy_ds = xr.Dataset(
        {
            "cyto_cp_masks": xr.DataArray(
                da.zeros(imgs.isel(C=0).shape, dtype="u2"), dims=list("STZYX")
            ),
            "combo_masks": z4d,
            "nuc_cp_masks": z4d,
            "labels": z4d,
            "cyto_labels": z4d,
        }
    )

    _ = dummy_ds.to_zarr(dataset_path, mode="a", compute=True)

    indexer.to_csv(dataset_path + "/indexer.csv", index=False)
