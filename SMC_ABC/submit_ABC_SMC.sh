#!/bin/bash
#SBATCH --account=<cluster_account>
#SBATCH --time=65:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output=smc_%j.out
#SBATCH --error=smc_%j.err

# Load modules 
module purge
module load <cluster_base_environment_2023>
module load openmpi
module load python/3.11

# Activate environment
source setup_env.sh
unset SLURM_MEM_PER_NODE

# Import and pre-complie Julia
echo "Precompiling Julia environment..."
python -c "from diffeqpy import de"
echo "Precompilation complete."

# Execute the script using mpiexec
mpiexec -n $SLURM_NTASKS python ABC-SMC.py