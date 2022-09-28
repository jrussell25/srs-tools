import pandas as pd
import time
from .locking_helpers import dump_data, parse_lockin_data
import numpy as np
from tqdm.autonotebook import tqdm
from dask.distributed import fire_and_forget

# deepsee = davy_jones.DeepSee.instance()
# daq = ziPython.ziDAQServer(...)

poll_int = 1 # seconds
timeout = 100

n_slices = 7 # Z
n_frames = 51 # T (R)
n_bands = 2 # CH/CD
n_loops = 5 #S 
frame_time = 2.8 # YX
wait_time = 10

change_at_frames = n_slices*n_frames

total_time = n_loops*n_bands*(frame_time*n_frames*n_slices+wait_time)
iterations = int(total_time/poll_int)

data_file = f"{time.strftime('%Y-%m-%d')}.h5"
time.sleep(2)
daq.sync()
frame_count = np.zeros(2, dtype='int')
loop_count = 0
for i in tqdm(range(iterations)): 
    out = daq.poll(poll_int, timeout, flat=True) # block until results come in
    df, err = parse_lockin_data(out)
    f = client.scatter(df[['x', 'auxin0', 'auxin1']])
    fire_and_forget(client.submit(dump_data, f, data_file, f"iter{i}")) # save
    frame_diff = np.diff(df.auxin0.values)
    d = dict(zip(*np.unique(frame_diff, return_counts=True)))
    
    for j, val in enumerate([-1,1]):
        if val in d:
            frame_count[j] += d[val]
    
    print(f"{frame_count}", end='\r')
    
    if np.all(frame_count==change_at_frames):
        print("Changing laser wavelength")
        if loop_count % 2 == 0:
            prepare_cd(daq, deepsee)
        if loop_count % 2 == 1:
            prepare_ch(daq, deepsee)
        loop_count += 1
        frame_count = np.zeros(2, dtype='int')
        daq.sync()
        while deepsee.get_status() != 50: time.sleep(0.1)