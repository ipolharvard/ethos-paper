#!/bin/bash -l
#SBATCH --job-name=ethos_infer
#SBATCH --time=1-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:8
#SBATCH --output=ethos_infer.log

gpu_num=8
rep_start=1
rep_stop=20
model_variant="best_model.pt"

dataset_dir="tokenized_datasets"
test_name=$1
shift 1
additional_arg="$*"

# Set dataset specific variables
test_data="${dataset_dir}/mimic_test_timelines_p26758.hdf5"
vocab="${dataset_dir}/mimic_vocab_t4367.pkl"
model_folder="mimic_layer_6_batch_32_do_0.3"

# Set test_name specific variables
case $test_name in
"sofa" | "icu_mortality" | "drg" | "icu_readmission" | "admission_mortality" | "readmission" | "mortality") ;;
*)
  echo "Wrong experiment name: '$test_name', available are: 'sofa', 'icu_mortality', 'drg', \
'admission_mortality', 'readmission', 'mortality'"
  exit 1
  ;;
esac

script_body="
cd /ethos/ethos_deploy
pip install --no-deps --no-index --no-build-isolation --user -e .
export PATH=\$HOME/.local/bin:\$PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/

for i in \$(seq ${rep_start} ${rep_stop}); do
    clear
    echo [\${i}/${rep_stop}]: test=${test_name}, model=${model_variant}, dataset=${dataset}
    ethos infer \\
        --test ${test_name} \\
        --model "out/${model_folder}/${model_variant}" \\
        --data ${test_data} \\
        --vocab ${vocab} \\
        --model_name ${model_folder}_${model_variant%_*} \\
        --n_jobs $(("$gpu_num" * 2)) \\
        --n_gpus ${gpu_num} \\
        ${additional_arg} \\
        --suffix rep\${i} || exit 1
done
"

clear
echo "Running inference for ${dataset} ${test_name}"

module load singularity
singularity exec \
  --contain \
  --nv \
  --writable-tmpfs \
  --bind "$SCRATCH":/ethos \
  ethos_latest.sif \
  bash -c "${script_body}"
