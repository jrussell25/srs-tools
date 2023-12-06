from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from tifffile import imwrite

from srs_tools.io import get_srs_indexer, zarrify_tiffs


@pytest.fixture
def tiff_dataset(tmp_path: Path) -> tuple[pd.DataFrame, xr.Dataset]:
    idxr = pd.read_csv("indexer_ref.csv")

    S = 5
    T = 4
    C = 3
    Z = 7
    XY = 8
    shape = (S, T, C, Z, XY, XY)

    data = np.arange(np.prod(shape)).reshape(shape)
    for i, (s, t, c, z) in enumerate(product(*[range(x) for x in shape[:-2]])):
        filename = idxr["filenames"].iloc[i]
        imwrite(tmp_path / filename, data[s, t, c, z])

    ds_ref = xr.DataArray(
        data, dims=list("STCZYX"), coords={"C": ["fluo", "bf", "srs"]}
    ).to_dataset(name="images")

    return idxr, ds_ref


def test_indexer(tmp_path: Path, tiff_dataset: tuple[pd.DataFrame, xr.Dataset]) -> None:
    idxr_ref, ds_ref = tiff_dataset

    idxr = get_srs_indexer(tmp_path)
    idxr["filenames"] = idxr["filenames"].apply(lambda x: str(Path(x).name))

    idxr = idxr.sort_values(list("STCZ")).reset_index(drop=True)

    pd.testing.assert_frame_equal(idxr, idxr_ref)


def test_zarrify(tmp_path: Path, tiff_dataset: tuple[pd.DataFrame, xr.Dataset]) -> None:
    idxr_ref, ds_ref = tiff_dataset

    ds = zarrify_tiffs(tmp_path, tmp_path / "dataset.zarr")

    xr.testing.assert_allclose(ds, ds_ref)
