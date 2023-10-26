import argparse
import json
import math
from pathlib import Path
from typing import Any, Tuple, Callable, cast


def override_pair(x: str) -> Tuple[str, Any]:
    (k, v) = x.split('=')
    if v in ['true', 'True']:
        v = True
    elif v in ['false', 'False']:
        v = False
    elif v in ['none', 'None', 'nothing']:
        v = None
    elif v.endswith('i'):
        v = int(v[:-1])
    elif v.endswith('f'):
        v = float(v[:-1])
    else:
        try:
            v = float(v)
        except ValueError:
            pass
    return k, v


parser = argparse.ArgumentParser(description='Score svbrdf estimates.')
parser.add_argument('--output', type=Path, nargs='+')
parser.add_argument('-L', '--loader-override', type=override_pair, nargs='*', default=[])
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--dome_portion', type=float, default=1.0)
parser.add_argument('--topdown_radius', type=float, default=2.0)
parser.add_argument('--sqrt', action='store_true')
parser.add_argument('--rerender_distance', type=float, default=2.41)
parser.add_argument('--light_distance', type=float, default=2.41)
parser.add_argument('--camera_distance', type=float, default=2.41)
parser.add_argument('--dome_render_seed', type=int, default=51234)
parser.add_argument('--num_dome_renders', type=int, default=128)
parser.add_argument('--num_topdown_renders', type=int, default=0)
args = parser.parse_args()


# flake8: noqa: E402
import jax
from jax import Array
import jax.numpy as jnp
import jax.numpy.linalg as jla
from lpips_jax import LPIPSEvaluator
from scipy.stats.qmc import Sobol

from matfusion_jax.data import Loader
from matfusion_jax.nprast import (
    nprast_dome_render_multi,
    nprast_flash_rerender,
    nprast_topdown_render_multi,
)

lpips = cast(
    Callable[[Array, Array], Array],         
    LPIPSEvaluator(replicate=False, net='alexnet'),
)

@jax.jit
def l1_input_error(input, result_svbrdf):
    result = nprast_flash_rerender(result_svbrdf, distance=args.rerender_distance)[0]**(1/args.gamma)
    return jnp.mean(jnp.abs(input - result))


@jax.jit
def l1_rerender_error(test_svbrdf, result_svbrdf):
    test = nprast_flash_rerender(test_svbrdf, distance=args.rerender_distance)[0]**(1/args.gamma)
    result = nprast_flash_rerender(result_svbrdf, distance=args.rerender_distance)[0]**(1/args.gamma)
    return jnp.mean(jnp.abs(test - result))


@jax.jit
def lpips_input_error(input, result_svbrdf):
    result = nprast_flash_rerender(result_svbrdf, distance=args.rerender_distance)[0]**(1/args.gamma)
    err = jnp.mean(lpips(jnp.expand_dims(input, 0)*2-1, jnp.expand_dims(result, 0)*2-1))
    if args.sqrt:
        err = jnp.sqrt(err)
    return err


@jax.jit
def lpips_rerender_error(test_svbrdf, result_svbrdf):
    test = nprast_flash_rerender(test_svbrdf, distance=args.rerender_distance)[0]**(1/args.gamma)
    result = nprast_flash_rerender(result_svbrdf, distance=args.rerender_distance)[0]**(1/args.gamma)
    err = lpips(jnp.expand_dims(test, 0)*2-1, jnp.expand_dims(result, 0)*2-1)
    if args.sqrt:
        err = jnp.sqrt(err)
    return err


@jax.jit
def lpips_dome_error(test_svbrdf, result_svbrdf, angles):
    test = nprast_dome_render_multi(
        test_svbrdf,
        angles,
        camera_distance=args.camera_distance,
        light_distance=args.light_distance,
    )[0]**(1/args.gamma)
    result = nprast_dome_render_multi(
        result_svbrdf,
        angles,
        camera_distance=args.camera_distance,
        light_distance=args.light_distance,
    )[0]**(1/args.gamma)
    err = lpips(test*2-1, result*2-1)
    if args.sqrt:
        err = jnp.sqrt(err)
    return jnp.mean(err)


@jax.jit
def lpips_topdown_error(test_svbrdf, result_svbrdf, h_pos):
    test = nprast_topdown_render_multi(
        test_svbrdf,
        h_pos=h_pos,
        camera_distance=args.camera_distance,
        light_distance=args.light_distance,
    )[0]**(1/args.gamma)
    result = nprast_topdown_render_multi(
        result_svbrdf,
        h_pos=h_pos,
        camera_distance=args.camera_distance,
        light_distance=args.light_distance,
    )[0]**(1/args.gamma)
    err = lpips(test*2-1, result*2-1)
    if args.sqrt:
        err = jnp.sqrt(err)
    return jnp.mean(err)


@jax.jit
def rmse_error(test_svbrdf, result_svbrdf):
    mse = jnp.mean((test_svbrdf - result_svbrdf)**2, axis=(0, 1))
    return {
        'diffuse': jnp.sqrt(jnp.mean(mse[0:3])),
        'specular': jnp.sqrt(jnp.mean(mse[3:6])),
        'roughness': jnp.sqrt(mse[6]),
        'normals': jnp.sqrt(jnp.mean(mse[7:10])),
    }


@jax.jit
def lpips_albedo_error(test_svbrdf, result_svbrdf):
    test_albedo = test_svbrdf[..., 0:3] + test_svbrdf[..., 3:6]
    result_albedo = result_svbrdf[..., 0:3] + result_svbrdf[..., 3:6]
    err = jnp.mean(lpips(
        jnp.expand_dims(test_albedo, 0)*2-1,
        jnp.expand_dims(result_albedo, 0)*2-1,
    ))
    if args.sqrt:
        err = jnp.sqrt(err)
    return err


for output in cast(list[Path], args.output):
    inputs = Loader(output / 'eval_dataset.yml')
    results = inputs.with_svbrdfs(output)
    output.joinpath('score_args.json').write_text(
        json.dumps({
            k: v for (k, v) in vars(args).items()
            if k not in ('output', 'dataset')
        }),
    )
    for result_dir in output.iterdir():
        try:
            rid = json.loads((result_dir / 'result_id.json').read_text())
        except IOError:
            continue

        print(result_dir.name)
        test_batch = inputs.load({ **rid, 'replicate': None })
        result_svbrdf = jnp.array(results.load(rid)['svbrdf'])

        if 'svbrdf' in test_batch:
            test_svbrdf = jnp.array(test_batch['svbrdf'])
            for (k, v) in rmse_error(
                test_svbrdf,
                result_svbrdf,
            ).items():
                results.metadata_path(rid, f'rmse_{k}_error').write_text(
                    str(v.item()),
                )

            results.metadata_path(rid, 'lpips_albedo_error').write_text(
                str(lpips_albedo_error(
                    test_svbrdf,
                    result_svbrdf,
                ).item()),
            )

            results.metadata_path(rid, 'l1_rerender_error').write_text(
                str(l1_rerender_error(
                    test_svbrdf,
                    result_svbrdf,
                ).item()),
            )

            results.metadata_path(rid, 'lpips_rerender_error').write_text(
                str(lpips_rerender_error(
                    test_svbrdf,
                    result_svbrdf,
                ).item()),
            )

            if args.num_dome_renders > 0:
                angles = Sobol(d=2, scramble=True, seed=args.dome_render_seed)\
                    .random_base2(math.ceil(math.log2(args.num_dome_renders)))
                angles[:, 1] = angles[:, 1] * args.dome_portion + 1 * (1 - args.dome_portion)
                results.metadata_path(rid, 'lpips_dome_error').write_text(
                    str(lpips_dome_error(
                        test_svbrdf,
                        result_svbrdf,
                        angles=angles,
                    ).item()),
                )
        else:
            test_input = jnp.array(test_batch['input'])
            results.metadata_path(rid, 'lpips_rerender_error').write_text(
                str(lpips_input_error(
                    test_input,
                    result_svbrdf,
                ).item()),
            )
            results.metadata_path(rid, 'l1_rerender_error').write_text(
                str(l1_input_error(
                    test_input,
                    result_svbrdf,
                ).item()),
            )
