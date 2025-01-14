#!/bin/bash
#SBATCH -N1 -c1 -n8
#SBATCH --gpus-per-node=a100:1
###SBATCH --gpus-per-node=1
#SBATCH --partition=d3
#SBATCH --mem=100G
#SBATCH -t48:00:00
#SBATCH --qos=d3
###SBATCH --output=gpu_run.out
###SBATCH --error=gpu_run.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shinya.ito@alleninstitute.org


# module load cuda/11.1
# python custom_run_script.py
# XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/shinya.ito/realistic-model/miniconda3/envs/tf5 \
# LD_LIBRARY_PATH=/lib64:/home/shinya.ito/realistic-model/miniconda3/envs/tf5/lib \


       # --restore_from '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model_Jan2024/results_indv_rossicore_npfr/multi_training/b_ef8e/ckpt-74' \
       # --restore_from '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model_Jan2024/results_em_indv/multi_training/b_hbc2/ckpt-671' \
       # --restore_from '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model_Jan2024/results_em_indv/v1_65871/b_x57m' \
       # --ckpt_dir '/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_new/V1_GLIF_model_Jan2024/results_em_indv/multi_training/b_4dns/small_metrics' \
       # --restore_from 'results_em_indv/v1_65871/b_hvo1' \
       # --restore_from 'results_rate_adjusted/v1_65871/b_olmd' \
       # --restore_from 'results_odsi/v1_65871/b_4mmv/latest' \

python -u multi_training.py --n_epochs 1000 --steps_per_epoch 50 --val_steps 2 --neurons 65871 \
       --rate_cost 20000.0 \
       --voltage_cost 0.1 \
       --osi_cost 2.0 \
       --learning_rate 0.001 \
       --all_neuron_rate_loss \
       --loss_core_radius 200.0 \
       --plot_core_radius 200.0 \
       --nobmtk_compat_lgn \
       --osi_loss_method 'crowd_osi' \
       --rotation 'ccw' \
       --data_dir 'GLIF_network' \
       --results_dir 'results_high_spont2'
