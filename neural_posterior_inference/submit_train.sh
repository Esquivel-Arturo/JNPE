#!/bin/bash
#SBATCH --job-name=JNPE_Train
#SBATCH --account=<cluster_account>   
#SBATCH --time=05:00:00      
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G            
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1   

# Load modules
module purge
module load <cluster_base_environment_2023>
module load python/3.11
module load cuda/12.2 cudnn

# Activate the environment
source setup_env.sh

python train_M1_M3_JNPE.py  