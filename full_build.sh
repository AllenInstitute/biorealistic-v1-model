#!/bin/bash
#SBATCH -N1 -c1 -n56
###SBATCH -c1 -n8
#SBATCH --partition=d3
#SBATCH --mem=800G
###SBATCH --mem-per-cpu=200G
#SBATCH -t30:00:00
#SBATCH --qos=d3
#SBATCH --output=full_build_log.out
#SBATCH --error=full_build_log.err

# using 56 core on 1 node ensure that we are using a node with 2 28-core Xeons.
# Those nodes have 512GB of RAM, so we can submit with 500GB of memory usage.
# To build the full model, we need about ~800GB of RAM.

srun --mpi=pmi2 python -u build_network.py -f -o full/network --radius 700.0 --core-radius 400.0
