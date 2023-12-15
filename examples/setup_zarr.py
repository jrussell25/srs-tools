import sys
from time import perf_counter

from aicsimageio.readers import TiffGlobReader
from dask.distributed import Client
from srs_tools.io import get_srs_indexer
import microutil as mu


if __name__=="__main__":

    tiff_path, dataset_path = sys.argv[1:]
    print(f"{tiff_path=}", flush=True)
    print(f"{dataset_path=}", flush=True)

    client = Client(processes=False)

    indexer = get_srs_indexer(tiff_path)

    t_cutoff = indexer.groupby('S')['T'].max().min()

    indexer = indexer.loc[indexer['T']<=t_cutoff]

    reader = TiffGlobReader(indexer["filenames"], indexer[list("STCZ")])

    imgs = reader.get_xarray_dask_stack(scene_character='S').drop_vars("C")

    _ = imgs.to_dataset(name="images").to_zarr(dataset_path, consolidated=True, mode='a')

    nuc = mu.normalize_fluo(imgs.isel(C=0).max('Z').astype('f4'), mode_cutoff_ratio=1e9)

    nuc.to_dataset(name='nuc').to_zarr(dataset_path, consolidated=True, mode='a')

