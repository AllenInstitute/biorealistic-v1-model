# %% a script for geneating spike trains for the background

import numpy as np
import h5py
import pathlib
import sys


def write_bkg(
    output_filename, n_neu=1, duration=3.0, binsize=2.5e-4, rate=1000, seed=0
):
    # time units are seconds, (inverse: Hz)

    nbins = int(duration / binsize)
    np.random.seed(seed)
    spike_bools = np.random.random([n_neu, nbins]) < (rate * binsize)
    where = np.where(spike_bools)
    # spikes_time = (np.where(spike_bools)[0] + 1) * binsize  # to avoid 0, still second
    timestamps = (where[1] + 1) * binsize
    # nids = np.zeros_like(spikes_time, dtype=np.uint)
    nids = where[0]

    # save
    out_file = h5py.File(output_filename, "w")
    # out_file["spikes/gids"] = nids
    # add some random value to avoid the bad time stamps.
    # out_file["spikes/timestamps"] = timestamps * 1000 + 0.01  # in ms
    # let's use gzip level 6.
    out_file.create_dataset(
        "spikes/gids",
        data=nids,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
    )
    out_file.create_dataset(
        "spikes/timestamps",
        data=timestamps * 1000 + 0.01,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
    )
    out_file.close()
    return 0


def write_regular_bkg(output_filename, duration=1.0, interval=0.1):
    spikes_time = np.linspace(interval, duration, int(duration / interval))
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
    # read the bkg file and check the number of bkg units.
    bkg_file = f"{basedir}/network/bkg_nodes.h5"
    # open the file and get the number of bkg units
    with h5py.File(bkg_file, "r") as f:
        n_neu = f["nodes"]["bkg"]["node_id"].shape[0]

    # bkg_name = f"{basedir}/bkg/bkg_spikes_1kHz_100s.h5"
    # write_bkg(bkg_name, n_neu, duration=100.0)
    bkg_name = f"{basedir}/bkg/bkg_spikes_1kHz_10s.h5"
    write_bkg(bkg_name, n_neu, duration=10.0)
    bkg_name = f"{basedir}/bkg/bkg_spikes_1kHz_3s.h5"
    write_bkg(bkg_name, n_neu)

    # 250 Hz (new default for 100 background units)
    bkg_name = f"{basedir}/bkg/bkg_spikes_250Hz_10s.h5"
    write_bkg(bkg_name, n_neu, rate=250, duration=10.0)
    bkg_name = f"{basedir}/bkg/bkg_spikes_250Hz_3s.h5"
    write_bkg(bkg_name, n_neu, rate=250, duration=3.0)
    # bkg_name = f"{basedir}/bkg/bkg_spikes_100Hz_10s.h5"
    # write_bkg(bkg_name, n_neu, rate=100, duration=10.0)
    # bkg_name = f"{basedir}/bkg/bkg_spikes_100Hz_3s.h5"
    # write_bkg(bkg_name, n_neu, rate=100, duration=3.0)
    # bkg_name = f"{basedir}/bkg/bkg_spikes_2kHz_10s.h5"
    # write_bkg(bkg_name, n_neu, rate=2000, duration=10.0)
    # bkg_name = f"{basedir}/bkg/bkg_spikes_2kHz_3s.h5"
    # write_bkg(bkg_name, n_neu, rate=2000, duration=3.0)

    # with 'full', all the bins are filled with spikes. i.e. no fluctuations.
    # bkg_name = f"{basedir}/bkg/bkg_spikes_full_100s.h5"
    # write_bkg(bkg_name, n_neu, rate=4000, duration=100.0)
    # bkg_name = f"{basedir}/bkg/bkg_spikes_full_30s.h5"
    # write_bkg(bkg_name, n_neu, rate=4000, duration=30.0)
    bkg_name = f"{basedir}/bkg/bkg_spikes_full_10s.h5"
    write_bkg(bkg_name, n_neu, rate=4000, duration=10.0)
    bkg_name = f"{basedir}/bkg/bkg_spikes_full_3s.h5"
    write_bkg(bkg_name, n_neu, rate=4000, duration=3.0)
    # these require high resolution simulations
    # bkg_name = f"{basedir}/bkg/bkg_spikes_5kHz_3s.h5"
    # write_bkg(bkg_name, n_neu, rate=5000, binsize=1.0e-5, duration=3.0)
    # bkg_name = f"{basedir}/bkg/bkg_spikes_5kHz_10s.h5"
    # write_bkg(bkg_name, n_neu, rate=5000, binsize=1.0e-5, duration=10.0)

    # regular bkg
    bkg_name = f"{basedir}/bkg/bkg_spikes_regular_1s.h5"
    write_regular_bkg(bkg_name)

    # for the 8 direction stimuli (10 repetition)
    start_seed = 381583
    print(f"Generating bkg spikes for orientation stimuli...")
    for i in range(8):
        for j in range(10):
            seed = start_seed + i * 10 + j
            dirname = f"{basedir}/bkg_8dir_10trials/angle{i*45}_trial{j}"
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            write_bkg(f"{dirname}/bkg_spikes_250Hz_3s.h5", n_neu, rate=250, seed=seed)

    start_seed2 = 451958
    contrasts = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
    for i in range(8):  # angle
        print(f"Generating bkg spikes for contrast stimuli {i+1}/8...")
        for j in range(6):  # contrast
            for k in range(10):  # trials
                seed = start_seed2 + i * 6 * 10 + j * 10 + k
                dirname = f"{basedir}/bkg_contrasts/angle{i*45}_contrast{contrasts[j]}_trial{k}"
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
                write_bkg(
                    f"{dirname}/bkg_spikes_250Hz_3s.h5", n_neu, rate=250, seed=seed
                )
