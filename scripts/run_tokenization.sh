#!/bin/bash -l
#SBATCH --job-name=ethos_tokenization::proj=IRB2023P002279,
#SBATCH --time=2:00:00
#SBATCH --partition=defq
#SBATCH --output=ethos_tokenize.log

script_body="
clear
cd /ethos

export PATH=\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/

pip install --no-deps --no-index --no-build-isolation --user -e .

ethos tokenize- $*
"

module load singularity
singularity exec \
  --contain \
  --nv \
  --writable-tmpfs \
  --bind "$(pwd)":/ethos \
  ethos.sif \
  bash -c "${script_body}"
