#!/bin/bash
#SBATCH -N1 -c1 -n56
#SBATCH --partition=braintv
#SBATCH --mem=500G
#SBATCH -t15:00:00
#SBATCH --qos=braintv
#SBATCH --output=full_build_log.out
#SBATCH --error=full_build_log.err

# using 56 core on 1 node ensure that we are using a node with 2 28-core Xeons.
# Those nodes have 512GB of RAM, so we can submit with 500GB of memory usage.
# To build the model, we need about ~400GB of RAM.

# module use /allen/programs/braintv/workgroups/modelingsdk/modulefiles
# module load mpich/3.4.1-slurm
# module load conda/4.5.4
# source activate v1_glif_modeling
source /home/shinya.ito/realistic-model/activate_custom_nest_sdk.sh
# module load nest/2.20.1-py37-slurm
# srun --mpi=pmi2 python build_network.py -f -o full/network
mpirun -np 56 python build_network.py -f -o full/network
