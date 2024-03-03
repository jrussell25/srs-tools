import sys
from time import perf_counter

import microutil as mu
import pandas as pd
import xarray as xr
from dask.distributed import Client

from srs_tools import BackgroundEstimator, trace_lineages

if __name__ == "__main__":
    tstart = perf_counter()
    dataset_path = sys.argv[1]

    client = Client(processes=False)

    ################
    # DATA LOADING #
    ################

    t0 = perf_counter()
    print("Loading data", flush=True)

    ds = xr.open_zarr(dataset_path)
    ds.labels.load()
    srs = ds.images.isel(C=-1).mean("Z", dtype="f4").load()
    labels = ds["cyto_labels"].load().to_dataset(name="labels")

    t1 = perf_counter()
    print(f"Data loaded -- {t1-t0:0.2f} seconds", flush=True)

    #########################
    # BACKGROUND ESTIMATION #
    #########################

    t0 = perf_counter()
    print("Starting background estimation", flush=True)

    if "srs_bsub" in xr.open_zarr(dataset_path):
        print("Found existing background estimates")
        srs_bsub = xr.open_zarr(dataset_path)["srs_bsub"].load()

    else:
        be = BackgroundEstimator(srs, labels["labels"])
        be.cv_labels
        be.sigma_scan(n_samples=10)
        be.sigma_opt.load()  # type: ignore
        print(f"--> Found sigma_opt={be.sigma_opt.data}", flush=True)  # type: ignore

        _ = be.background_estimate.to_dataset(name="bkgd_est").to_zarr(
            dataset_path, mode="a"
        )

        srs_bsub = srs - be.background_estimate

        _ = srs_bsub.to_dataset(name="srs_bsub").to_zarr(dataset_path, mode="a")

    t1 = perf_counter()
    print(f"Background estimation complete -- {t1-t0:0.2f} seconds", flush=True)

    ########################
    # SINGLE CELL ANALYSIS #
    ########################

    t0 = perf_counter()
    print("Starting single-cell quantification", flush=True)

    avgs = (
        mu.single_cell.average(labels, srs_bsub).to_series().dropna().rename("avg")
    )  # .set_index(['S','T','CellID'])
    coms = (
        mu.single_cell.center_of_mass(labels)
        .to_series()
        .dropna()
        .unstack("com")
        .rename(lambda x: f"cyto_com_{x.lower()}", axis=1)
    )
    nuc_coms = (
        mu.single_cell.center_of_mass(ds)
        .to_series()
        .dropna()
        .unstack("com")
        .rename(lambda x: f"nuc_com_{x.lower()}", axis=1)
    )
    areas = (
        mu.single_cell.area(labels)
        .to_series()
        .where(lambda x: x > 0)
        .dropna()
        .rename("area")
    )

    df = pd.concat([avgs, coms, nuc_coms, areas], axis=1, join="inner")
    df.to_hdf(dataset_path + "/sc_table.h5", key="properties")

    lineage_info = pd.concat(
        {s: trace_lineages(nuc_coms.loc[s]) for s in nuc_coms.index.unique("S")},
        names=["S"],
    )

    lineage_info.to_hdf(dataset_path + "/sc_table.h5", key="lineages")

    t1 = perf_counter()
    print(f"Single-cell quantification complete -- {t1-t0:0.2f} seconds", flush=True)

    print(f"Preprocessing complete -- Total time {t1-tstart:0.2f} seconds", flush=True)
