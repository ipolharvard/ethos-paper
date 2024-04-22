#!/bin/bash -l
#SBATCH --job-name=ethos_train
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:8
#SBATCH --output=ethos_train.log

export OMP_NUM_THREADS=20

case $1 in
mimic | 1)
  dataset=mimic
  data_path=mimic_train_timelines_p241015.hdf5
  vocab_path=mimic_vocab_t4367.pkl
  val_frac=0.04
  ;;
*)
  echo "Wrong experiment number: '$1', available are: 'mimic'"
  exit 1
  ;;
esac

datasets_dir=ethos/data/tokenized_datasets
data_path=${datasets_dir}/${data_path}
vocab_path=${datasets_dir}/${vocab_path}

BATCH_SIZE=32
BLOCK_SIZE=2048
N_LAYER=10
N_HEAD=12
N_EMBD=768
DROPOUT=0.1
LR=0.0006
MIN_LR=0.00001

model_name="layer_${N_LAYER}_batch_${BATCH_SIZE}_do_${DROPOUT}"

script_body="
cd /ethos/ethos_deploy
pip install --no-deps --no-index --no-build-isolation --user -e .
export PATH=\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/:/.singularity.d/libs/

torchrun --no_python --standalone --nproc_per_node=8 ethos train \
  --data_train $data_path \
  --val_frac $val_frac \
  --vocab $vocab_path \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --n_layer $N_LAYER \
  --n_head $N_HEAD \
  --n_embd $N_EMBD \
  --dropout $DROPOUT \
  --lr $LR \
  --min_lr $MIN_LR \
  --log_interval 5 \
  --eval_interval 1000 \
  --gradient_accumulation_steps 8 \
  --max_iters 1000000 \
  --lr_decay_iters 50000 \
  --eval_iters 50 \
  --ctx_no_grad \
  --wandb_project "ethos-$dataset" \
  --wandb_run_name $model_name \
  --out_dir "out/${dataset}_${model_name}" \
  --wandb_log
"

module load singularity
singularity exec \
  --contain \
  --nv \
  --writable-tmpfs \
  --bind "$SCRATCH":/ethos \
  ethos_latest.sif \
  bash -c "${script_body}"
