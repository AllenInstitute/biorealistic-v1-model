
# Running a biorealistic model with TensorFlow

## Get the code

Fork the following repo.
[https://github.com/JavierGalvan9/V1_GLIF_model/tree/master](https://github.com/JavierGalvan9/V1_GLIF_model/tree/master)

Clone it to your local directory.
`git clone <forked repo>`

Checkout the "master" branch.
`git checkout master`

## Create environment with TensorFlow
Create an environment using the `environment_tf.yml` file, or install all the requirements in it.

The command to create the env should be:
`conda env create -f environment_tf.yml --name <env name>`


## Set environmental variable for the CUDA library
This is necessary if you are using the conda version of cuda.

In your conda environment directory, under "etc/conda/activate.d/", create a script that sets the following environmental variables
XLA_FLAGS=--xla_gpu_cuda_data_dir=\<conda home\>/envs/tf5
LD_LIBRARY_PATH=/lib64:\<conda home\>/envs/tf5/lib

In my case, I have a script named:
`/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/envs/tf5/etc/conda/activate.d/env_vars.sh`

which content is
`export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/shinya.ito/realistic-model/miniconda3/envs/tf5`
`export LD_LIBRARY_PATH=/lib64:/home/shinya.ito/realistic-model/miniconda3/envs/tf5/lib`

If you don't do this, it will try to load the system's CUDA library, which may not be setup.


## Copy SONATA network to the TF directory
Copy the network directory as `GLIF_network` in the base directory of the TF repository.
If the directory structure looks like:
`V1_GLIF_model/GLIF_network/network/v1_nodes.h5`
it's good.

The `GLIF_network` directory should contain the following subdirctories\
`network`: all the SONATA network files\
`components`: should contain `cell_models` and `synaptic_models` subdirectories. Copy all the model files from the original network directory.

The final file structure should look like this.

```
V1_GLIF_model/
    GLIF_network/
        components/
            cell_models/
                ...
            synaptic_models/
                ...
        network/
            bkg_node_types.csv
            bkg_nodes.h5
            ...
```


## Training on the cluster
Example: 'multi_training_slurm.sh' (Please change sbatch fields for your need.)

Before submitting the job, please activate the environment for TF in the hpc-login. (or write that in the script, which might be better.)



## Synaptic data for TensorFlow models
This step is not necessary for running the model, but if you are interested in how the parameters of the files under `synaptic_data` directory in the TensorFlow model repo, you can run the following command to generate them.

`$ snakemake tf_basis_functions`

This will generate `tf_props/tau_basis.npy` and `tf_props/basis_function_weights.csv`.