#!/bin/bash
#SBATCH --job-name=gen_mpi_5k
#SBATCH --account=<cluster_account>   
#SBATCH --time=32:00:00 
#SBATCH --nodes=1                    
#SBATCH --ntasks=30 total cores
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G             
#SBATCH --output=sim_%j.out     
#SBATCH --error=sim_%j.err      

module purge
module load <cluster_base_environment_2023>
module load openmpi
module load python/3.11

# Activate your environment
source setup_env.sh
unset SLURM_MEM_PER_NODE

# Execute the script
mpiexec -n $SLURM_NTASKS python static_sim.py --model M1