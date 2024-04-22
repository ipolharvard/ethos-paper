#!/bin/bash -l
#SBATCH --job-name=ethos_tokenize
#SBATCH --time=6:00:00
#SBATCH --partition=defq
#SBATCH --output=ethos_tokenize.log

# you might need to create setup.py in the ethos_deploy directory

script_body="
cd /ethos/ethos_deploy
pip install --no-deps --no-index --no-build-isolation --user -e .
export PATH=\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/

ethos tokenize- $*
"

module load singularity
singularity exec \
  --contain \
  --nv \
  --writable-tmpfs \
  --bind "$SCRATCH":/ethos \
  ethos_latest.sif \
  bash -c "${script_body}"
