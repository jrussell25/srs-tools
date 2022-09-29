import numpy as np
import pandas as pd

def get_sample_key(keys):
    count = 0
    out = None
    for k in keys:
        if "sample" in k:
                out = k
                count += 1
    
    if count == 0:
        raise ValueError("Did not find a key containing 'sample'")
    elif count>1:
        warnings.warn("Found multiple sample keys in data. Return last one")
    return out


def parse_lockin_data(data):
    # for now assume there is only one demodulator that we are subscribed to
    s_key = get_sample_key(data.keys())
    sample_dict = dict(data[s_key])
    checks = sample_dict.pop('time')
    missing_data = checks['dataloss']
    
    sample = pd.DataFrame(sample_dict)
    sample['timestamp'] = sample['timestamp'].astype('int64') #need this for writing to h5
    sample[['auxin0', 'auxin1']] = (sample[['auxin0','auxin1']]>1).astype('int8')
    sample['x'] = sample['x'].astype('float32')
    sample = sample.set_index('timestamp')
    return sample, missing_data

def dump_data(data, filename, key):
    data.to_hdf(filename, key, complevel=9, complib="blosc:lz4", format='table')

def load_lockin_data(new_file):

    ddf = dd.read_hdf(new_file, key='*',sorted_index=True) #worth whatever write speed cost come from format='table'

    ddf['frame_diff'] = ddf.auxin0.diff().astype('int8')
    ddf['line_diff'] = ddf.auxin1.diff().astype('int8')

    # this is the only section that requires computing 
    # (implicit in the .values)
    starts = ddf.loc[ddf.frame_diff==1].index.values
    stops = ddf.loc[ddf.frame_diff==-1].index.values

    starts = ddf.loc[ddf.line_diff==1].index.values
    stops = ddf.loc[ddf.line_diff==-1].index.values


    frame_idx = pd.IntervalIndex.from_arrays(starts,stops, closed='both')
    line_idx = pd.IntervalIndex.from_arrays(starts, stops, closed='both')

    tss = ddf.index.to_series()

    frame_cat = tss.map_partitions(pd.cut, frame_idx)#.compute()

    frame_codes = frame_cat.dropna().cat.codes.to_frame(name='frame')

    line_cat = tss.map_partitions(pd.cut, line_idx)

    line_codes = line_cat.dropna().cat.codes.to_frame(name='line')

    fl = dd.merge(frame_codes, line_codes)

    subset = ddf.merge(fl)

    subset['framewise_line'] = subset.groupby('frame', group_keys=False).line.apply(lambda s: s-s.min())

    return subset

