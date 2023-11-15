import argparse
import json
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import ctime, time_ns
from typing import Any
from ast import literal_eval

import flax
import wandb
from tqdm import tqdm

from matfusion_jax import vis
from matfusion_jax.data import Generator
from matfusion_jax.model import Model
from matfusion_jax.config import default_modes

# frozendict screws things up in old versions
flax.config.update('flax_return_frozendict', False)


def override_pair(x: str) -> tuple[str, Any]:
    (k, v) = x.split('=', 1)
    v = literal_eval(v)
    return k, v


parser = argparse.ArgumentParser(description='Train an svbrdf model.')
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--mode', type=str, choices=default_modes.keys())
parser.add_argument('--mode_json', type=Path)
parser.add_argument('-O', '--override', type=override_pair, nargs='*', default=[])
parser.add_argument('--epocs', type=int, required=True)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=16)
parser.add_argument('--accumulation', type=int, default=1)
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--test_steps', type=int, default=20)
parser.add_argument('--test_count', type=int, default=10)
parser.add_argument('--test_dataset', type=Path, default=Path('datasets/test_rasterized.yml'))
parser.add_argument('--finetune_checkpoint', type=Path)
parser.add_argument('--resume_checkpoint', type=Path)
parser.add_argument('--run_name', type=str)
parser.add_argument('--output_checkpoint', type=Path)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

if args.mode_json is not None:
    mode = json.loads(args.mode_json.read_text())
elif args.mode is not None:
    mode = deepcopy(default_modes[args.mode])
else:
    assert False, 'must supply --mode or --mode_json'

for (k, v) in args.override:
    mode[k] = v

print(json.dumps(mode, indent=2))

if args.wandb:
    wandb.init(
        project="svbrdf-diffusion",
        config=mode,
    )
    assert wandb.run is not None
    name = wandb.run.name
else:
    name = datetime.utcnow().isoformat()

if args.run_name:
    name = args.run_name

results_path = Path(f'./results/training_{name}')
checkpoint_path = args.output_checkpoint or Path(f'./checkpoints/{name}')

print(f'saving checkpoints to {checkpoint_path}')

if args.wandb:
    print('saving test results to Weights & Biases')
    train_vis = vis.WandbReport()
else:
    print(f'saving test results to {results_path}')
    results_path.mkdir(parents=True, exist_ok=True)
    train_vis = vis.ResultsReport(results_path)

gen = Generator(
    args.dataset,
    seed=args.seed,
    batch_size=mode['batch_size'],
    replicates=1,
    worker_count=args.workers,
)

test_gen = Generator(
    args.test_dataset,
    seed=0,
    batch_size=args.test_batch_size,
    replicates=1,
    worker_count=1,
)

train = Model(
    gen.resolution,
    mode,
    accumulation=args.accumulation,
)
train.init_model()

model_table = train.visulize()
if args.wandb:
    wandb.log({
        'model_summary': wandb.Html(f'<pre>{model_table}</pre>'),
    }, commit=False)
else:
    (results_path / 'model_summary.txt').write_text(model_table)

if args.finetune_checkpoint is not None:
    train.load_finetune_checkpoint(args.finetune_checkpoint)

if args.resume_checkpoint is not None:
    train.load_resume_checkpoint(args.resume_checkpoint)

checkpoint_path.mkdir(parents=True, exist_ok=True)
(checkpoint_path / 'mode.json').write_text(json.dumps(mode, indent=2))
(checkpoint_path / 'model_summary.txt').write_text(model_table)


current_step = 0
training_sum = 0
waiting_sum = 0
perf_start = time_ns()
batched_train_info = defaultdict(list)
pbar = None

for epoc_num in range(1, args.epocs+1):
    if mode['condition'] == 'none' and mode['control_model'] is None:
        train.training_uncon_evaluate_step(
            train_vis,
            steps=args.test_steps,
            count=args.test_count,
        )
    else:
        test_gen.begin()
        while True:
            test_batch = test_gen.take()
            if test_batch is None:
                break
            names = [i['name'] for i in test_batch['id']]
            train.training_evaluate_step(
                test_batch,
                train_vis,
                names=names,
                steps=args.test_steps,
                match_reflectance=True,
            )

    gen.begin()
    if pbar is None:
        pbar = tqdm(total=gen.total_samples*args.epocs)
    while True:
        perf_take = time_ns()
        batch = gen.take()
        waiting_sum += time_ns() - perf_take

        if batch is None:
            break

        if len(batch['id']) < mode['batch_size']:
            current_step += 1
            continue

        perf_train = time_ns()
        train.training_step(batch, train_vis)
        training_sum += time_ns() - perf_train

        if current_step % args.log_every == 0:
            train_vis.submit(
                name=name,
                step=current_step,
                commit=True,
            )
            train_vis.clear()

        pbar.update(len(batch['id']))
        current_step += 1

    perf_epoc = time_ns()
    est_finish = (perf_epoc - perf_start) // 1e9 / epoc_num * args.epocs + perf_start // 1e9
    training_frac = training_sum / (perf_epoc - perf_start)
    waiting_frac = waiting_sum / (perf_epoc - perf_start)

    print(f'Finished epoc {epoc_num} after {current_step} steps ({training_frac*100:.0f}% training, {waiting_frac*100:.0f}% waiting).')
    print(f'Estimated completion {ctime(est_finish)}.')

    train.save_training_checkpoint(checkpoint_path, step=train.current_step)

train.save_training_checkpoint(checkpoint_path, step=train.current_step)
