module purge
# Load required base modules (Anonymized for review)
# Ensure Python 3.11, Julia 1.10, OpenMPI, CUDA 12.2, and cuDNN are loaded in your environment.
module load python/3.11 julia/1.10 openmpi mpi4py cuda/12.2 cudnn

# Activate local python environment (e.g., venv or Conda)
# source /path/to/your/venv/bin/activate

export JULIA_DEPOT_PATH="$HOME/.julia"
export JULIA_NUM_THREADS=1

# Allow Julia to download required packages during initial setup
export PYTHON_JULIAPKG_OFFLINE=no
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

# Force Keras to use the JAX engine 
export KERAS_BACKEND="jax"

# Ensure standard local python paths are included
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH