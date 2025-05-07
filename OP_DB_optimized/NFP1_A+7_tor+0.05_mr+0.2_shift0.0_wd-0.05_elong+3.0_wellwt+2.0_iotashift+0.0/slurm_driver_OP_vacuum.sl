#!/bin/bash

#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH --account=m4505_g
#SBATCH --time=01:20:00

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load python/3.11;\
module unload cudatoolkit;\

conda activate desc-env;\



python3 -u driver_OP.py 0 
