# %% A script for experimenting basis function conversion

# let's try to replicate the shape of a particular function.
import sys
import matplotlib.pyplot as plt
import nest
import numpy as np
import pandas as pd
from iminuit import minimize

# nest.Install("glif_psc_double_alpha_module")


# %%
def get_trace(tau_syn, tau_syn_slow, amp_slow):
    nest.ResetKernel()
    # Define the neuron parameters
    neuron = nest.Create("glif_psc_double_alpha")
    spikes = nest.Create("spike_generator", 1, {"spike_times": [1.0]})
    params = {
        "tau_syn_fast": [tau_syn],
        "tau_syn_slow": [tau_syn_slow],
        "amp_slow": [amp_slow],
    }
    neuron.set(params)
    multimeter = nest.Create("multimeter")
    multimeter.set({"record_from": ["I_syn"], "interval": 0.1})

    nest.Connect(
        spikes, neuron, syn_spec={"weight": 1.0, "receptor_type": 1, "delay": 1.0}
    )
    nest.Connect(multimeter, neuron)
    nest.Simulate(33.0)

    # return the traces
    times = multimeter.get(["events"])["events"]["times"]
    current_trace = multimeter.get(["events"])["events"]["I_syn"]

    # format the traces to be the same as the generated function
    times = times - 2
    current_trace = current_trace[times > 0.0]
    times = times[times > 0.0]
    # plt.plot(times, current)

    # Return the traces
    return times, current_trace


# %% make a basis function approximation of this double alpha function.


def generate_basis_functions(n_basis, da_params, save_tau=False):
    # define tau_basis between the minimum and maximum of the data.
    # min_tau is minimum of tau_syn, and max_tau is max of tau_syn_slow.
    min_tau = da_params["tau_syn_fast"].min()
    max_tau = da_params["tau_syn_slow"].max()
    tau_basis = np.logspace(np.log10(min_tau), np.log10(max_tau), n_basis)
    if save_tau:
        np.save("tf_props/tau_basis.npy", tau_basis)

    # define the basis alpha functions
    times = np.arange(0, 30, 0.1)
    basis_functions = np.zeros((n_basis, len(times)))
    for i, tau in enumerate(tau_basis):
        basis_functions[i, :] = times / tau * np.exp(1 - times / tau)

    return basis_functions, times


# define a function that adds up the basis functions with given weights


def basis_shape(weights, basis_functions):
    # just multiply weights to each basis function and add them up.
    # return np.sum(weights * basis_functions, axis=0)
    return np.dot(basis_functions.T, weights).flatten()


def loss(weights, basis_functions, target):
    return np.mean((basis_shape(weights, basis_functions) - target) ** 2)


# %%

n_basis = 5
da_params = pd.read_csv("tf_props/double_alpha_params.csv", sep=" ")

basis_functions, times = generate_basis_functions(n_basis, da_params)

# %% plot the basis functions
for i in range(n_basis):
    plt.plot(times, basis_functions[i, :])
plt.xlabel("Time (ms)")
plt.savefig("tf_props/basis_functions.png")

# %%


init_weights = np.ones(n_basis) / n_basis
times, current = get_trace(*da_params.iloc[0])

# show one example
plt.plot(times, current)
result = minimize(loss, init_weights, args=(basis_functions, current))
plt.plot(times, basis_shape(result.x, basis_functions))


# do it for every row of the da_params
# %%
# do it for a bulk
def optimize_params(n_basis, da_params):
    # Load the double alpha parameters
    # da_params = pd.read_csv(da_param_file)

    # Generate the basis functions
    basis_functions, times = generate_basis_functions(n_basis, da_params)

    # Optimize the parameters for each trace
    func_vals = []
    params = []
    for i in range(da_params.shape[0]):
        times, current = get_trace(*da_params.iloc[i])
        init_weights = np.random.rand(n_basis) / 3
        result = minimize(loss, init_weights, args=(basis_functions, current))
        func_vals.append(result.fun)
        params.append(result.x)

    func_vals = np.array(func_vals)
    params = np.array(params)

    return func_vals, params


# %% do it for n_basis 2 to 6
n_bases = np.arange(2, 8)
da_params = pd.read_csv("tf_props/double_alpha_params.csv", sep=" ")
func_vals = []
params = []
for n_basis in n_bases:
    print("n_basis:", n_basis)
    func_vals_tmp, params_tmp = optimize_params(n_basis, da_params)
    func_vals.append(func_vals_tmp)
    params.append(params_tmp)

func_vals = np.array(func_vals)
# params = np.array(params)

# %% func_vals is now 5, 61 matrix, for each n_basis, plot a boxplot of the func_vals
plt.boxplot(func_vals.T)
plt.yscale("log")
plt.xticks(np.arange(1, 7), np.arange(2, 8))
plt.xlabel("Number of basis functions")
plt.ylabel("Loss function value")
plt.savefig("tf_props/loss_vs_n_basis.png")

# %% from each group, pick the worst example and plot the trace along with the true trace.
# pick the worst example
worst_idx = np.argmax(func_vals, axis=1)
# plot the trace and the fitted trace
for i, idx in enumerate(worst_idx):
    basis_functions, times = generate_basis_functions(n_bases[i], da_params)
    print("n_basis:", n_bases[i])
    print("worst_idx:", idx)
    times, current = get_trace(*da_params.iloc[idx])
    plt.figure()
    plt.plot(times, current)
    plt.plot(times, basis_shape(params[i][idx], basis_functions))
    plt.title(
        "n_basis: " + str(n_bases[i]) + ", loss: {:.2e}".format(func_vals[i, idx])
    )
    plt.savefig("tf_props/n_basis_" + str(n_bases[i]) + ".png")

# %% 5 basis function might be a reasonable starting point.
# now optimize full list of the basis functions.
da_params_full = pd.read_csv(
    "tf_props/double_alpha_params_full.csv", sep=" ", index_col=0
)
fvals, params = optimize_params(5, da_params_full)


# now params contains each row of the optimized weights. I'd like to save it to
# another csv file, specifying the weights for each basis function.
# %% save the optimized weights to a csv file
df = pd.DataFrame(params, columns=["w0", "w1", "w2", "w3", "w4"])
df.index = da_params_full.index
df.to_csv("tf_props/basis_function_weights.csv", sep=" ")


# finally, save tau.
_ = generate_basis_functions(5, da_params_full, save_tau=True)

print("Done!")
