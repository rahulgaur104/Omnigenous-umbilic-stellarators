#!/bin/bash

#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH --account=m4505_g
#SBATCH --time=02:25:00

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load python/3.11;\
module unload cudatoolkit;\

conda activate desc-env2;\


python3 -u driver_OP.py

