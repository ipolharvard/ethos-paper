import json
import multiprocessing as mp

import torch as th
from tqdm import tqdm

from .constants import Test


def get_process_info():
    proc_name = mp.current_process().name
    proc_num = 1 if proc_name == "MainProcess" else int(proc_name.split("-")[-1]) - 1
    return proc_name, proc_num


def run_inference(loader, args, num_gpus: int = 8):
    model, device, vocab, stoi, results_dir, test_name, suffix, no_compile = args

    proc_name, proc_num = get_process_info()
    if device == "cuda":
        device = f"cuda:{proc_num % num_gpus}"
        th.cuda.set_device(device)
    model.to(device)
    if not no_compile:
        model = th.compile(model)

    dataset = loader.dataset.dataset
    context_len = dataset.context_len
    timeline_len = dataset.timeline_len
    max_timeline_size = context_len + timeline_len
    time_limit = 30 / 365.25 if test_name == Test.READMISSION else 2
    toi = th.tensor(vocab.encode(stoi), device=device, dtype=th.long)

    results = []
    for timeline, ground_truth in tqdm(
        loader, proc_name, total=len(loader), position=proc_num, smoothing=0
    ):
        timeline = timeline.to(device)
        gen_token_num = 0
        offset = 0
        while True:
            if test_name == Test.SOFA_PREDICTION and gen_token_num == 1:
                # append a sofa token to the timeline and continue generating
                last_token = th.tensor(
                    vocab.encode(["SOFA"]), device=timeline.device, dtype=th.long
                )
            else:
                last_token, probs = model.get_next_token(timeline[None, ...], return_probs=True)

            if not offset and len(timeline) == max_timeline_size:
                offset = 1

            timeline = th.cat(
                (timeline[:context_len], timeline[context_len + offset :], last_token.view(-1)),
            )
            gen_token_num += 1
            # stop criterion
            if timeline[-1] in toi:
                stop_reason = "token_of_interest"
                break
            elif test_name == Test.READMISSION or gen_token_num > timeline_len:
                timeline_time = vocab.get_timeline_total_time(
                    timeline[-gen_token_num:].cpu(), decode=True
                )
                if timeline_time > time_limit:
                    stop_reason = "time_limit"
                    break
            elif test_name == Test.SOFA_PREDICTION and gen_token_num == 3:
                # if there are 3 tokens generated and none of them is a toi,
                # then sofa experiment failed as the 3rd token should always be a quantile
                stop_reason = "sofa_fail"
                break
            elif test_name == Test.DRG_PREDICTION:
                stop_reason = "drg_fail"
                break

        expected = ground_truth.pop("expected")
        if test_name in (
            Test.ADMISSION_MORTALITY,
            Test.ICU_MORTALITY,
            Test.SINGLE_ADMISSION,
            Test.SOFA_PREDICTION,
            Test.DRG_PREDICTION,
        ):
            expected = vocab.decode(expected)
        actual = vocab.decode(last_token.item())
        actual_prob = probs[0][last_token.item()].item()
        toi_probs = dict(zip(stoi, probs[0][toi].tolist()))
        timeline_time = vocab.get_timeline_total_time(timeline[-gen_token_num:].cpu(), decode=True)

        results.append(
            {
                "expected": expected,
                "actual": actual,
                "stop_reason": stop_reason,
                "actual_prob": actual_prob,
                **toi_probs,
                **ground_truth,
                "token_dist": gen_token_num,
                "token_time": timeline_time,
            }
        )

    res_file = results_dir / f"part_{proc_num}{f'_{suffix}' if suffix is not None else ''}"
    with res_file.with_suffix(".json").open("w") as f:
        json.dump(results, f, indent=4)

    th.cuda.empty_cache()
