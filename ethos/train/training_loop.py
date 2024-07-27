import os
import time
from pathlib import Path

import numpy as np
import torch as th
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .metrics import estimate_loss
from .utils import get_train_val_data, make_infinite_loader, get_lr
from ..constants import DISCHARGE_STOKEN, ADMISSION_STOKEN
from ..datasets import TimelineDataset
from ..model import ModelConfig, Ethos
from ..tokenize import SpecialToken, Vocabulary
from ..utils import setup_torch, load_model_from_checkpoint


def train_ethos(args):
    device = args.device
    out_dir = Path(args.out_dir)
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        th.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
    tokens_per_iter = args.gradient_accumulation_steps * args.batch_size * args.block_size
    print(f"tokens per iteration per worker is: {tokens_per_iter:,}")

    if master_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    ctx = setup_torch(device, args.dtype, 42 + seed_offset)

    # VOCABULARY
    vocab = Vocabulary(args.vocab)
    vocab_size = (len(vocab) // 64 + 1) * 64
    args.__dict__["vocab_size"] = len(vocab)

    tokens_of_interest = [SpecialToken.DEATH, ADMISSION_STOKEN, DISCHARGE_STOKEN]
    tokens_of_interest = {stoken: vocab.encode(stoken) for stoken in tokens_of_interest}

    # DATASETS
    train_data, val_data = get_train_val_data(args.data_train, args.val_frac)

    train_dataset = TimelineDataset(train_data, encode=vocab.encode, block_size=args.block_size)
    val_dataset = TimelineDataset(val_data, encode=vocab.encode, block_size=args.block_size)

    context_len = train_dataset.context_len if args.ctx_no_grad else 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
    )
    train_dataloader = make_infinite_loader(train_dataloader)

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True
    )
    val_dataloader = make_infinite_loader(val_dataloader)

    if master_process:
        print(
            "Train dataset size: {:,}, Val dataset size: {:,}".format(
                len(train_dataset), len(val_dataset)
            )
        )
        print(
            "Every evaluation will use {:.2%} of validation dataset".format(
                args.batch_size * args.block_size * args.eval_iters / len(val_dataset)
            )
        )

    def get_batch(split):
        data = train_dataloader if split == "train" else val_dataloader
        x, y = next(data)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return x, y

    iter_num, best_val_loss, best_metric_score, optimizer_state = 0, 1e9, 0, None
    # model init
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=vocab_size,
        dropout=args.dropout,
    )
    if args.resume:
        model_path = out_dir / "best_model.pt"
        print(f"Resuming training from {model_path}")

        model, iter_num, best_val_loss, optimizer_state = load_model_from_checkpoint(
            model_path, device
        )
    else:
        print("Initializing a new model from scratch")
        config = ModelConfig(**model_args)
        model = Ethos(config)
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = th.amp.GradScaler(enabled=(args.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), device
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if not args.no_compile:
        print("Compiling the model...")
        model = th.compile(model)

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # logging
    logger = None
    if args.wandb_log and master_process:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
        logger = wandb

    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, get_batch, args.eval_iters, tokens_of_interest)
            print(
                "step {}: train loss {:.4f}, val loss {:.4f}".format(
                    iter_num,
                    losses["loss/train"],
                    losses["loss/val"],
                )
            )
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": losses["loss/val"],
                "model_args": model_args,
                "iter_num": iter_num,
                "config": args,
            }
            th.save(checkpoint, out_dir / "recent_model.pt")
            print(f"Saved the most recent model.")
            if losses["loss/val"] < best_val_loss:
                th.save(checkpoint, out_dir / "best_model.pt")
                print(f"Saved the best model: {best_val_loss} => {losses['loss/val']}")
                best_val_loss = losses["loss/val"]

            metric_score = np.mean(
                [v for k, v in losses.items() if k.startswith("acc_top/") and "/all/" not in k]
            )
            if metric_score > best_metric_score:
                th.save(checkpoint, out_dir / "toi_metric_model.pt")
                print(f"Saved the best toi model: {best_metric_score} => {metric_score}")
                best_metric_score = metric_score

            if logger is not None:
                epochs = iter_num * tokens_per_iter / len(train_dataset)
                logger.log(
                    {
                        "other/iter": iter_num,
                        "other/lr": lr,
                        "other/mfu": running_mfu * 100,  # convert to percentage
                        "other/toi_avg_score": metric_score,
                        "other/epochs": epochs,
                        **losses,
                    }
                )

        if iter_num == 0 and args.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y, context_length=context_len)
                loss = (
                    loss / args.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * args.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"[{iter_num}]: loss={lossf:.4f}, time={dt * 1000:.0f}ms, mfu={running_mfu:.2%}")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > args.max_iters:
            break

    if ddp:
        destroy_process_group()
