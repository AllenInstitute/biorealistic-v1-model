# %% a script for geneating spike trains for the background

import numpy as np
import h5py
import pathlib
import sys


def write_bkg(output_filename, duration=3.0, binsize=2.5e-4, rate=1000, seed=0):
    # time units are seconds, (inverse: Hz)

    nbins = int(duration / binsize)
    np.random.seed(seed)
    spike_bools = np.random.random(nbins) < (rate * binsize)
    spikes_time = (np.where(spike_bools)[0] + 1) * binsize  # to avoid 0, still second
    nids = np.zeros_like(spikes_time, dtype=np.uint)

    # save
    out_file = h5py.File(output_filename, "w")
    out_file["spikes/gids"] = nids
    out_file["spikes/timestamps"] = spikes_time * 1000  # in ms
    out_file.close()
    return 0


if __name__ == "__main__":
    # try to write the bkg (let's make all of them)
    # basedir = "small"
    basedir = sys.argv[1]
    pathlib.Path(f"{basedir}/bkg").mkdir(parents=True, exist_ok=True)
    bkg_name = f"{basedir}/bkg/bkg_spikes_1kHz_100s.h5"
    write_bkg(bkg_name, duration=100.0)
    bkg_name = f"{basedir}/bkg/bkg_spikes_1kHz_10s.h5"
    write_bkg(bkg_name, duration=10.0)
    bkg_name = f"{basedir}/bkg/bkg_spikes_1kHz_3s.h5"
    write_bkg(bkg_name)
    
    # for the 8 direction stimuli (10 repetition)
    start_seed = 381583
    for i in range(8):
        for j in range(10):
            seed = start_seed + i * 10 + j
            dirname = f"{basedir}/bkg_8dir_10trials/angle{i*45}_trial{j}"
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            write_bkg(f"{dirname}/bkg_spikes_1kHz_3s.h5", seed=seed)

    
    
    

