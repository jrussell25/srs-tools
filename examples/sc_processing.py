import sys
from time import perf_counter

import microutil as mu
import xarray as xr
import pandas as pd

from srs_tools import BackgroundEstimator

if __name__=="__main__":
    tstart = perf_count()

    dataset_path = sys.argv[1]

    ######################### 
    # BACKGROUND ESTIMATION #
    ######################### 

    ds = xr.open_zarr(dataset_path)
    srs = ds.images.isel(C=-1).mean('Z', dtype='f4').load()
    labels = ds['cyto_labels'].load()
    
    be = BackgroundEstimator(srs, labels)
    be.cv_labels;
    be.sigma_scan(n_samples=5)
    be.sigma_opt.load()

    srs_bsub = (srs-be.background_estimate)

    labels_ds = labels.to_dataset(name='labels')

    avgs = mu.single_cell.average(labels_ds, srs_bsub).to_series().dropna().rename('avg')#.set_index(['S','T','CellID'])
    coms = mu.single_cell.center_of_mass(labels_ds).to_series().dropna().unstack('com').rename(lambda x: f"cyto_com_{x.lower()}", axis=1)
    nuc_coms = mu.single_cell.center_of_mass(ds).to_series().dropna().unstack('com').rename(lambda x: f"nuc_com_{x.lower()}", axis=1)
    areas = mu.single_cell.area(labels_ds).to_series().where(lambda x: x>0).dropna().rename('area')
