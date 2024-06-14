# %% calculate if curves for all the cell models downloaded.
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

# %% first, load the cell types.
network = "small"
df = pd.read_csv(f"{network}/network/v1_node_types.csv", sep=" ")
# choose a cell type that contains e4 in the name
e4df = df[df["pop_name"].str.contains("e4")]
row = df.iloc[0]


# another thought.
# I can also just use the glif_models folder to get all the cells.
# that way, I don't have to care about network generation.

# %% try loading one model and determine the I-F curve
config = json_utilities.read(f'glif_models/cell_models/{row["dynamics_params"]}')
neuron = GlifNeuron.from_dict(config)

stim_amps = np.linspace(0, 500, 251)


def get_spike_count(neuron, stim_amp, duration=2.0):
    neuron.dt = 2.5e-4  # 4 kHz matching with NEST simulation
    steps = int(duration / neuron.dt)
    stimulus = [1.0e-12 * stim_amp] * steps

    output = neuron.run(stimulus)
    return output["spike_time_steps"].size / duration


def get_if_curve(row, stim_amps):
    config = json_utilities.read(f'glif_models/cell_models/{row["dynamics_params"]}')
    neuron = GlifNeuron.from_dict(config)
    return [get_spike_count(neuron, stim_amp) for stim_amp in stim_amps]


# if_curves = [get_if_curve(row, stim_amps) for _, row in e4df.iterrows()]
# if_curves_np = np.array(if_curves)


def save_if_curve_all(df, stim_amps, filename):
    # if_curves = [get_if_curve(row, stim_amps) for _, row in df.iterrows()]
    if_curves = df.parallel_apply(lambda row: get_if_curve(row, stim_amps), axis=1)
    if_curves_np = np.array(if_curves)
    np.save(filename, if_curves_np)
    return


# if_curves_all = [get_if_curve(row, stim_amps) for _, row in df.iterrows()]
# if_curves_all_np = np.array(if_curves_all)
# # save the variable
# np.save('if_curves_all', if_curves_all_np)

# do this only once (takes about 25 minutes)
save_if_curve_all(df, stim_amps, "glif_models/if_curves_all")


# %time get_if_curve(df.iloc[0], stim_amps)
# if_curve = get_if_curve(df.iloc[0], stim_amps)


# let's plot if curve
# plt.plot(stim_amps, if_curve)
