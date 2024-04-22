from types import SimpleNamespace

from click import Choice, command, option

from ethos.train import train_ethos


@command()
# data
@option("--data_train", required=True, help="Path to the file with the training data.")
@option(
    "--val_frac",
    required=True,
    type=float,
    help="Fraction of the training data to be the validation set.",
)
@option("--vocab", required=True, help="Path to the vocab file.")
# modes
@option("--out_dir", default="out")
@option("--eval_interval", type=int, default=2000)
@option("--log_interval", type=int, default=1)
@option("--eval_iters", type=int, default=50)
@option("--eval_only", is_flag=True)
@option("--resume", is_flag=True, help="Resume training from the checkpoint.")
# wandb logging
@option("--wandb_log", is_flag=True)
@option("--wandb_project", default="ethos")
@option("--wandb_run_name", default="ethos_run")
# training parameters
@option(
    "--gradient_accumulation_steps",
    type=int,
    default=5 * 8,
    help="used to simulate larger batch sizes",
)
@option(
    "--batch_size",
    type=int,
    default=12,
    help="If gradient_accumulation_steps > 1, this is the micro-batch size.",
)
# model parameters
@option("--block_size", type=int, default=2048)
@option("--n_layer", type=int, default=6)
@option("--n_head", type=int, default=12)
@option("--n_embd", type=int, default=768, help="Defaults to 768 like GPT-2.")
@option("--dropout", type=float, default=0)
@option("--bias", is_flag=True, help="Use bias inside LayerNorm and Linear layers.")
# adamW optimizer
@option("--max_iters", type=int, default=100_000, help="Total number of iterations.")
@option("--lr", "--learning_rate", type=float, default=6e-4, help="Max learning rate.")
@option("--weight_decay", type=float, default=1e-1)
@option("--beta1", type=float, default=0.9)
@option("--beta2", type=float, default=0.95)
@option("--grad_clip", type=float, default=1.0)
# learning rate decay settings
@option("--warmup_iters", type=int, default=2000)
@option("--lr_decay_iters", type=int, default=50_000)
@option("--min_lr", type=float, default=6e-5)
# DDP settings
@option("--backend", default="nccl", type=Choice(["nccl", "gloo"]))
# system
@option("--device", default="cuda", type=Choice(["cuda", "cpu"]))
@option("--dtype", default="bfloat16", type=Choice(["float32", "bfloat16", "float16"]))
@option("--no_compile", is_flag=True, help="Don't compile the model using Triton.")
# optional
@option("--ctx_no_grad", is_flag=True, help="Don't compute gradient for the context tokens.")
def train(**kwargs):
    """This training script can be run both on a single gpu in debug mode, and also in a larger
    training run with distributed data parallel (ddp).

    To run on a single GPU, example:
    $ ethos train [args...]

    To run with DDP on 4 gpus on 1 node, example:

    $ torchrun --standalone --nproc_per_node=4 ethos train [args...]

    To run with DDP on 4 gpus across 2 nodes, example:

    - Run on the first (master) node with example IP 123.456.123.456:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456
     --master_port=1234 ethos train [args...]

    - Run on the worker node:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456
     --master_port=1234 ethos train [args...]

    (If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
    """
    args = SimpleNamespace(**kwargs)

    if not (0 <= args.val_frac <= 1):
        raise ValueError("val_frac must be between 0 and 1")

    train_ethos(args)
