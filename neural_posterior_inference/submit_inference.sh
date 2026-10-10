#!/bin/bash
#SBATCH --job-name=bf_inference
#SBATCH --account=<cluster_account>  
#SBATCH --time=03:00:00             
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4            
#SBATCH --mem=16G                    
#SBATCH --output=bf_infer_%j.out     
#SBATCH --error=bf_infer_%j.err      

# Load modules
module purge
module load <cluster_base_environment_2023>
module load openmpi
module load python/3.11

# Activate environment
source setup_env.sh

python infer_M1_M3_JNPE.py 