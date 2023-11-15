import argparse
import json
from pathlib import Path
from typing import Any
from ast import literal_eval

import numpy as np
from tqdm import tqdm

def override_pair(x: str) -> tuple[str, Any]:
    (k, v) = x.split('=', 1)
    v = literal_eval(v)
    return k, v

parser = argparse.ArgumentParser(description='Run svbrdf estimation model.')
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--backbone_checkpoint', type=Path, default=None)
parser.add_argument('-O', '--override', type=override_pair, nargs='*', default=[])
parser.add_argument('-L', '--loader-override', type=override_pair, nargs='*', default=[])
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--replicates', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--sampler', type=str, default='euler_a')
parser.add_argument('--ema', type=lambda x: x in ['1', 'true', 'True'], default=True)
parser.add_argument('--animation', action='store_true')
parser.add_argument('--guidance_scale', type=float, default=None)
parser.add_argument('--match_color', action='store_true')
parser.add_argument('--match_reflectance', action='store_true')
parser.add_argument('--late_match', type=lambda x: x not in ['0', 'false', 'False'], default=True)
parser.add_argument('--full_output', action='store_true')
parser.add_argument('--whitelist', nargs='*')
args = parser.parse_args()

# flake8: noqa: E402
import jax

from matfusion_jax import vis
from matfusion_jax.data import Generator, save_svbrdf
from matfusion_jax.model import Model

# needed to use optax EMA
# flax.config.update('flax_return_frozendict', False)

mode = json.loads((args.checkpoint / 'mode.json').read_text())
print(f'Using {mode}')
for (k, v) in args.override:
    mode[k] = v

args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'mode.json').write_text(json.dumps(mode, indent=2))

gen = Generator(
    args.dataset,
    seed=args.seed,
    batch_size=args.batch_size,
    replicates=args.replicates,
    worker_count=args.workers,
    whitelist=args.whitelist,
    **dict(args.loader_override),
)
gen.begin()

gen.loader.save_config(args.output / 'eval_dataset.yml')
results = gen.loader.with_svbrdfs(args.output)

train = Model(gen.resolution, mode)
train.load_eval_checkpoint(args.checkpoint, args.backbone_checkpoint, args.ema)

model_table = train.visulize()
with (args.output / 'model_table.txt').open('w') as f:
    print(model_table, file=f)

if args.full_output:
    report = vis.ResultsReport()
else:
    report = 'minimal'

pbar = tqdm(total=gen.total_samples)
index = 0
while True:
    batch = gen.take()

    if batch is None:
        break

    output = train.evaluate(
        batch,
        noise_keys=[jax.random.PRNGKey(s) for s in batch['seed']],
        steps=args.steps,
        sampler=args.sampler,
        animation=args.animation,
        guidance_scale=args.guidance_scale,
        match_color=args.match_color,
        match_reflectance=args.match_reflectance,
        late_match=args.late_match,
    )
    svbrdf = np.nan_to_num(output['svbrdf_est'])
    svbrdf = np.clip(svbrdf, 0.0, 1.0)
    save_svbrdf(svbrdf, results, batch['id'])
    for (i, rid) in enumerate(batch['id']):
        results.metadata_path(rid, 'result_id.json').write_text(json.dumps(rid))
        if output.get('svbrdf_animation') is not None:
            vis.save_svbrdf_video(
                results.metadata_path(rid, 'svbrdf_diffusion.webm'),
                output['svbrdf_animation'][i],
                fps=10,
                gamma=2.2,
                horizontal=True,
            )

    pbar.update(len(batch['id']))
    index += 1
