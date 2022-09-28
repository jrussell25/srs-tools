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