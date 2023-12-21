import glob
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from aicsimageio.readers import TiffGlobReader


def indexer_check(idxr: pd.DataFrame) -> None:
    """
    Check indexer dataframe to ensure that columns have the correct consecutive and
    unique values to populate a full array.
    """
    for col in idxr:
        consecutive = np.arange(idxr[col].nunique())
        found = np.sort(idxr[col].unique())
        assert np.allclose(consecutive, found), f"Found invalid values in column {col}"


def get_srs_indexer(expt_path: str | Path) -> pd.DataFrame:
    """
    Prepare an indexer dataframe for a typical srs experiment with a single fluorescence
    channel, broghtfield, and single-wavenumber SRS images. Returns a dataframe with
    columns STCZ for dimensions as well as filenames and LDM indices ("L")

    Parameters
    ----------
    expt_path: str|Path
        Path to experiment tiff files. Annoying leica "LUT" files will be filtered out.
    """
    files = [x for x in glob.glob(str(Path(expt_path) / "*.tif")) if "LUT" not in x]
    indexer = pd.DataFrame(
        [re.findall(r"(\d+)", Path(x).name) for x in files], columns=list("SLZC")
    ).astype(int)
    indexer["Q"] = pd.Series(
        [re.findall("fluo|srs", Path(x).name)[0] for x in files], dtype="category"
    ).cat.codes
    indexer["filenames"] = pd.Series(files, copy=True)

    scale = indexer.groupby("S")["Q"].nunique().sum()

    indexer["T"] = indexer.groupby(list("SQ"))["L"].transform(
        lambda x: (x - x.min()) // scale
    )
    indexer_check(indexer[list("SQTCZ")])
    indexer["C"] = indexer["C"] + indexer["Q"]
    indexer.drop(columns="Q", inplace=True)

    return indexer


def zarrify_tiffs(tiff_path: str | Path, zarr_path: str | Path) -> xr.Dataset:
    indexer = get_srs_indexer(Path(tiff_path))

    reader = TiffGlobReader(indexer["filenames"], indexer[list("STCZ")])

    ds = reader.get_xarray_dask_stack(scene_character="S").to_dataset(name="images")
    ds["C"] = ["fluo", "bf", "srs"]

    _ = ds.to_zarr(zarr_path)

    ds = xr.open_zarr(zarr_path)
    return ds


def get_metadata_tables(expt_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = glob.glob(str(Path(expt_path) / "*_Properties.xml"))

    timestamps = []
    dim_data = []
    for s, file in enumerate(files):
        parsed = ET.parse(file)
        s, ldm = (int(x) for x in re.findall(r"(\d+)", Path(file).name))
        mode = re.findall("fluo|srs", Path(file).name)[0]

        for x in parsed.iter("TimeStamp"):
            d = x.attrib  # type : ignore
            d["L"] = ldm
            d["S"] = s
            d["mode"] = mode
            timestamps.append(d)

        for x in parsed.iter("DimensionDescription"):
            d = x.attrib  # type : ignore
            d["L"] = ldm
            dim_data.append(d)

    timestamps = pd.DataFrame(timestamps)
    timestamps["T"] = (
        timestamps.groupby(["S", "mode"])["L"].rank(method="dense").astype(int) - 1
    )

    dim_data = pd.DataFrame(dim_data).set_index(["L", "DimID"]).unstack("DimID")

    timestamps["datetime"] = timestamps.apply(
        lambda df: pd.to_datetime(df["Date"] + " " + df["Time"])
        + pd.to_timedelta(int(df["MiliSeconds"]), unit="ms"),
        axis=1,
    )

    timestamps = timestamps.sort_values("datetime")

    return timestamps, dim_data
