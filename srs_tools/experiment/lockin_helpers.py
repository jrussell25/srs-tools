import warnings
from typing import Optional

import dask.dataframe as dd
import pandas as pd


def get_sample_key(keys: list[str]) -> Optional[str]:
    count = 0
    out = None
    for k in keys:
        if "sample" in k:
            out = k
            count += 1

    if count == 0:
        raise ValueError("Did not find a key containing 'sample'")
    elif count > 1:
        warnings.warn("Found multiple sample keys in data. Return last one")

    return out


def parse_lockin_data(data: dict) -> tuple[pd.DataFrame, bool]:
    # for now assume there is only one demodulator that we are subscribed to
    keys = list(data.keys())
    s_key = get_sample_key(keys)
    sample_dict = dict(data[s_key])
    checks = sample_dict.pop("time")
    missing_data = checks["dataloss"]

    sample = pd.DataFrame(sample_dict)
    sample["timestamp"] = sample["timestamp"].astype(
        "int64"
    )  # need this for writing to h5
    sample[["auxin0", "auxin1"]] = (sample[["auxin0", "auxin1"]] > 1).astype("int8")
    sample["x"] = sample["x"].astype("float32")
    sample = sample.set_index("timestamp")
    return sample, missing_data


def dump_data(data: pd.DataFrame, filename: str, key: str) -> None:
    data.to_hdf(filename, key, complevel=9, complib="blosc:lz4", format="table")


def load_lockin_data(
    filename: Optional[str] = None, ddf: Optional[dd.DataFrame] = None
) -> dd.DataFrame:
    if ddf is None:
        ddf = dd.read_hdf(
            filename, key="*", sorted_index=True
        )  # worth whatever write speed cost come from format='table'

    if filename is None and ddf is None:
        raise ValueError("Must provide h5 file or dask dataframe to parse")

    ddf["frame_diff"] = ddf["auxin0"].diff()
    ddf["line_diff"] = ddf["auxin1"].diff()
    ddf = ddf.dropna(how="any")

    ddf[["frame_count", "line_count"]] = (
        ddf[["frame_diff", "line_diff"]] == 1
    ).cumsum()

    pixels = ddf.loc[
        (ddf["auxin0"] == 1) & (ddf["auxin1"] == 1), ["x", "frame_count", "line_count"]
    ]

    return pixels
    # pixels['framewise_line'] = pixels.groupby('frame_count',
    #  group_keys=False).line_count.apply(lambda s: s-s.min, meta=('line_count,'int64'))
