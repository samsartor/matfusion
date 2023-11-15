import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Any, Optional

import flax
from flax.core import FrozenDict
import flax.struct
import jax
import jax.debug
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from flax import serialization
from flax.training import common_utils, train_state
from jax.tree_util import (
    tree_flatten,
    tree_flatten_with_path,
    tree_map,
    tree_map_with_path,
    tree_unflatten,
)

from . import nprast
from .config import (
    center_img,
    center_svbrdf,
    uncenter_img,
    uncenter_svbrdf,
)
from .pipeline import (
    Diffusion,
    DiffusionModel,
    DiffusionSettings,
    distributed,
    func_map,
    func_mean,
    func_replicate,
    func_unreplicate,
)
from .net.mine import MyNoiseModel
from .vis import Report


class Checkpointer:
    """
    A simple checkpointer class shaped like orbax.checkpoint.Checkpointer. Because we
    are locked into JAX 0.4.8, we can't really use tensorstore for the time being but
    Flax's msgpack serializer works well enough.
    """

    def save(self, path: Path, item: Any, step: Optional[int] = None):
        path.mkdir(exist_ok=True, parents=True)
        if step is None:
            tmp_path = path / 'checkpoint_tmp.msgpack'
        else:
            tmp_path = path / f'checkpoint_{step}.msgpack'
        item = serialization.to_state_dict(item)
        tmp_path.write_bytes(serialization.msgpack_serialize(item))
        final_path = path / 'checkpoint.msgpack'
        final_path.unlink(missing_ok=True)
        final_path.symlink_to(tmp_path)

    def restore(self, path: Path, item: Any = None) -> Any:
        path = path / 'checkpoint.msgpack'
        tree = serialization.msgpack_restore(path.read_bytes())
        if item is not None and isinstance(tree, dict):
            return serialization.from_state_dict(item, tree)
        else:
            return tree


ckptr = Checkpointer()


def batched_slide_textures(x, a, b):
    a = a % x.shape[1]
    b = b % x.shape[2]
    x = jnp.concatenate((x[:, a:, :, :], x[:, :a, :, :]), 1)
    x = jnp.concatenate((x[:, :, b:, :], x[:, :, :b, :]), 2)
    return x


def jaxify_batch(batch, allow_scalars=True):
    flat, treedef = tree_flatten(batch)
    flat = [
        ((x if allow_scalars else None) if np.isscalar(x) else jnp.array(x))
        for x in flat
    ]
    return tree_unflatten(treedef, flat)


nprast_animated_render = jax.jit(nprast.nprast_animated_render, static_argnames=['geo'])
nprast_random_render_batch = jax.jit(nprast.nprast_random_render_batch, static_argnames=['geo'])
nprast_ortho_render_batch = jax.jit(nprast.nprast_ortho_render_batch, static_argnames=['geo'])
nprast_flash_rerender_batch = jax.jit(nprast.nprast_flash_rerender_batch, static_argnames=['geo'])


def prepare_batch(batch, mode):
    if 'svbrdf' in batch:
        y = batch['svbrdf']
        y = jnp.nan_to_num(y)
        nprast_input = y
        shape = y.shape[:-1]
    else:
        assert 'input' in batch
        nprast_input = jnp.zeros((*batch['input'].shape[0:3], 10))
        nprast_input = nprast_input.at[:, :, :, 6].set(0.5)
        nprast_input = nprast_input.at[:, :, :, 9].set(1.0)
        y = None
        shape = batch['input'].shape[:-1]

    if 'rast_flash' in mode['inputs'] or 'rast_halfway' in mode['inputs']:
        if jnp.any(batch['view_distance'] == math.inf):
            assert jnp.all(batch['view_distance'] == math.inf)
            rast_flash, rast_halfway = nprast_ortho_render_batch(
                nprast_input,
                jnp.stack((
                    batch['flash_x'],
                    batch['flash_y'],
                    batch['flash_distance'],
                ), axis=1)
            )
        else:
            assert jnp.all(batch['view_distance'] == batch['flash_distance'])
            rast_flash, rast_halfway = nprast_flash_rerender_batch(
                nprast_input,
                distance=batch['flash_distance'],
                geo=mode['svbrdf_geo'],
            )
        batch = {
            'rast_flash': batch['input'] if 'input' in batch else rast_flash,
            'rast_halfway': rast_halfway * 0.5 + 0.5,
            **batch,
        }

    if 'input' in batch and 'render' in mode['inputs']:
        batch = { 'render': batch['input'], **batch }
    if 'other_input' in batch and 'other_render' in mode['inputs']:
        batch = { 'other_render': batch['other_input'], **batch }

    for i in mode['inputs']:
        assert i in batch, f'batch is missing {i}, try a different dataset'

    if len(mode['inputs']) > 0:
        x = jnp.concatenate([batch[i] for i in mode['inputs']], 3)
        x = jnp.nan_to_num(x)
        x = center_img(x, mode)
    else:
        x = None

    if y is not None:
        y = center_svbrdf(y, mode)

    return x, y, shape


@partial(func_map, static_argnames=['dif'])
def generator_training_step_impl(gen_state, x, y, key, dif: Diffusion):
    def loss_fn(gen_params):
        loss, detailed_loss = dif.loss(
            gen_state,
            {'params': gen_params},
            x, y, key
        )
        return loss, detailed_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (eps_loss, detailed_loss), grads = func_mean(grad_fn(gen_state.params))
    return gen_state.apply_gradients(
        grads=grads,
    ), detailed_loss


def generator_step_impl(gen_state, batch, key, mode):
    dif = Diffusion.from_mode(mode)
    x, y, batch_shape = prepare_batch(batch, mode)

    if distributed:
        x = common_utils.shard(x)
        y = common_utils.shard(y)
        loss_keys = jax.random.split(key, num=x.shape[0])
    else:
        loss_keys = key
    gen_state, loss = generator_training_step_impl(gen_state, x, y, loss_keys, dif=dif)
    return gen_state, loss


def evaluate_impl(gen_state, null_state, batch, key, noise_keys, settings, mode):
    x, y, batch_shape = prepare_batch(batch, mode)
    shape = (*batch_shape, mode['channels'])
    dif = Diffusion.from_mode(mode)

    def info_callback(
        svbrdf,
        output,
        sample=None,
        initial=None,
        noise=None,
        velocity=None,
        **kwargs,
    ):
        nonlocal key

        svbrdf = uncenter_svbrdf(svbrdf, mode)
        if output is None:
            output = {}
        else:
            output = {**output}
        output['svbrdf_est'] = svbrdf
        if initial is not None:
            output['noise'] = initial
        elif 'svbrdf_start' not in output:
            output['svbrdf_start'] = svbrdf
        if settings.animation in [True, 'signal']:
            if 'svbrdf_animation' not in output:
                output['svbrdf_animation'] = jnp.expand_dims(svbrdf, 1)
            else:
                output['svbrdf_animation'] = jnp.concatenate([
                    output['svbrdf_animation'],
                    jnp.expand_dims(svbrdf, 1),
                ], axis=1)
        if settings.animation == 'sample' and sample is not None:
            if 'svbrdf_animation' not in output:
                output['svbrdf_animation'] = jnp.expand_dims(sample, 1)
            else:
                output['svbrdf_animation'] = jnp.concatenate([
                    output['svbrdf_animation'],
                    jnp.expand_dims(sample, 1),
                ], axis=1)
        if settings.animation == 'noise' and noise is not None:
            if 'svbrdf_animation' not in output:
                output['svbrdf_animation'] = jnp.expand_dims(noise, 1)
            else:
                output['svbrdf_animation'] = jnp.concatenate([
                    output['svbrdf_animation'],
                    jnp.expand_dims(noise, 1),
                ], axis=1)
        if settings.animation == 'velocity' and velocity is not None:
            if 'svbrdf_animation' not in output:
                output['svbrdf_animation'] = jnp.expand_dims(velocity, 1)
            else:
                output['svbrdf_animation'] = jnp.concatenate([
                    output['svbrdf_animation'],
                    jnp.expand_dims(velocity, 1),
                ], axis=1)
        return output

    def create_noise(key, shape):
        if noise_keys is None:
            return jax.random.normal(key, shape)
        else:
            noise_layers = []
            for i in range(shape[0]):
                key, new_noise_key = jax.random.split(noise_keys[i])
                noise_keys[i] = new_noise_key
                noise_layers.append(jax.random.normal(key, shape[1:]))
            return jnp.stack(noise_layers, axis=0)

    key, noise_key = jax.random.split(key)
    initial = create_noise(noise_key, shape)
    initial = initial * dif.init_noise_scale(settings.sampler)

    if y is not None and y.shape != initial.shape:
        raise Exception(f"shape mismatch: {y.shape} vs {initial.shape}")

    if 'input' in batch:
        settings.true_render = batch['input']
    else:
        settings.true_render = nprast_flash_rerender_batch(batch['svbrdf'])[0]

    key, gen_key = jax.random.split(key)
    info = dif.sample(
        DiffusionModel(
            main_state=gen_state,
            main_vars={'params': gen_state.eval_params},
            null_state=null_state,
            null_vars=None if null_state is None else {'params': null_state.params},
        ),
        x,
        initial,
        gen_key,
        settings=settings,
        callback=info_callback,
    )
    return tree_map(np.array, info)


def training_evaluate_step_impl(gen_state, batch, key, noise_keys, settings, vis, names, mode):
    x, y, batch_shape = prepare_batch(batch, mode)
    shape = (*batch_shape, mode['channels'])
    dif = Diffusion.from_mode(mode)

    def info_callback(svbrdf, output, sample=None, **kwargs):
        nonlocal key

        svbrdf = uncenter_svbrdf(svbrdf, mode)
        if output is None:
            return {
                'svbrdf_est': svbrdf,
                'svbrdf_progress': jnp.expand_dims(svbrdf, 1),
            }
        else:
            return {
                'svbrdf_est': svbrdf,
                'svbrdf_progress': jnp.concatenate((
                    output['svbrdf_progress'],
                    jnp.expand_dims(svbrdf, 1),
                ), 1),
            }

    if noise_keys is None:
        key, noise_key = jax.random.split(key)
        initial = jax.random.normal(noise_key, shape)
    else:
        each_shape = (1, *shape[1:])
        initial = jnp.concatenate([jax.random.normal(noise_key, each_shape) for noise_key in noise_keys], axis=0)

    initial = initial * dif.init_noise_scale(settings.sampler)

    if y is not None and y.shape != initial.shape:
        raise Exception(f"shape mismatch: {y.shape} vs {initial.shape}")

    key, gen_key = jax.random.split(key)
    info = dif.sample(
        DiffusionModel(
            main_state=gen_state,
            main_vars={'params': gen_state.eval_params},
        ),
        x,
        initial,
        gen_key,
        settings=settings,
        callback=info_callback,
    )
    y_est = info['svbrdf_est']

    x = uncenter_img(x, mode)

    if y is None:
        y_loss_channels = None
    else:
        y = uncenter_svbrdf(y, mode)
        y_loss_channels = jnp.mean(
            (y_est - y)**2,
            axis=(0, 1, 2),
        )


    key, r_key = jax.random.split(key)
    r_est = nprast_random_render_batch(y_est, r_key, geo=mode['svbrdf_geo'])[0]

    if y is None:
        r = None
    else:
        r = nprast_random_render_batch(y, r_key, geo=mode['svbrdf_geo'])[0]

    if y_loss_channels is not None:
        vis.scalar('test_loss/diffuse', jnp.mean(y_loss_channels[0:3]), mean=True)
        vis.scalar('test_loss/specular', jnp.mean(y_loss_channels[3:6]), mean=True)
        vis.scalar('test_loss/roughness', jnp.mean(y_loss_channels[6]), mean=True)
        if mode['svbrdf_geo'] == 'normals':
            vis.scalar('test_loss/normals', jnp.mean(y_loss_channels[7:10]), mean=True)
        if mode['svbrdf_geo'] == 'height':
            vis.scalar('test_loss/height', jnp.mean(y_loss_channels[7:8]), mean=True)
    for (i, n) in enumerate(names or range(y_est.shape[0])):
        vis.cond_image('test_vis/{n}/input', x[i], n=n)
        if y is not None:
            vis.svbrdf_image('test_vis/{n}/true_svbrdf', y[i], n=n, geo=mode['svbrdf_geo'])
        vis.svbrdf_image('test_vis/{n}/est_svbrdf', y_est[i], n=n, geo=mode['svbrdf_geo'])
        if mode['svbrdf_geo'] == 'height':
            if y is not None:
                vis.cond_image('test_vis/{n}/true_normals', nprast.height2normals(y[i, :, :, 7:8]), n=n)
            vis.cond_image('test_vis/{n}/est_normals', nprast.height2normals(y_est[i, :, :, 7:8]), n=n)
        if settings.animation in [True, 'signal']:
            vis.svbrdf_video(
                'test_vis/{n}/svbrdf_progress',
                info['svbrdf_progress'][i],
                fps=settings.steps/2,
                n=n,
                geo=mode['svbrdf_geo'],
            )
        if r is not None:
            vis.image('test_vis/{n}/true_relighting', r[i], n=n, gamma=2.2)
        vis.image('test_vis/{n}/est_relighting', r_est[i], n=n, gamma=2.2)
        if settings.animation in [True, 'signal']:
            if y is not None:
                vis.video(
                    'test_vis/{n}/lighting_animation',
                    nprast_animated_render(y[i], geo=mode['svbrdf_geo']),
                    fps=settings.steps/2, n=n, gamma=2.2,
                )
            vis.video(
                'test_vis/{n}/relighting_animation',
                nprast_animated_render(y_est[i], geo=mode['svbrdf_geo']),
                fps=settings.steps/2, n=n, gamma=2.2,
            )


def uncon_evaluate_step_impl(gen_state, shape, key, noise_keys, settings, vis, mode):
    dif = Diffusion.from_mode(mode)

    def info_callback(svbrdf, output, sample=None, **kwargs):
        nonlocal key

        svbrdf = uncenter_svbrdf(svbrdf, mode)
        if output is None:
            return {
                'svbrdf_est': svbrdf,
                'svbrdf_progress': jnp.expand_dims(svbrdf, 1),
            }
        else:
            return {
                'svbrdf_est': svbrdf,
                'svbrdf_progress': jnp.concatenate((
                    output['svbrdf_progress'],
                    jnp.expand_dims(svbrdf, 1),
                ), 1),
            }

    if noise_keys is None:
        key, noise_key = jax.random.split(key)
        initial = jax.random.normal(noise_key, shape)
    else:
        each_shape = (1, *shape[1:])
        initial = jnp.concatenate([jax.random.normal(noise_key, each_shape) for noise_key in noise_keys], axis=0)

    initial = initial * dif.init_noise_scale(settings.sampler)

    key, gen_key = jax.random.split(key)
    info = dif.sample(
        DiffusionModel(
            main_state=gen_state,
            main_vars={'params': gen_state.eval_params},
        ),
        None,
        initial,
        gen_key,
        settings=settings,
        callback=info_callback,
    )
    y_est = info['svbrdf_est']
    r_est = nprast_flash_rerender_batch(y_est, geo=mode['svbrdf_geo'])[0]
    for i in range(y_est.shape[0]):
        vis.svbrdf_image('test_vis/{i}/svbrdf', y_est[i], i=i, geo=mode['svbrdf_geo'])
        if settings.animation:
            vis.svbrdf_video(
                'test_vis/{i}/svbrdf_progress',
                info['svbrdf_progress'][i],
                fps=settings.steps/2,
                i=i,
                geo=mode['svbrdf_geo'],
            )
        vis.image('test_vis/{i}/lighting', r_est[i], i=i)
        if settings.animation:
            vis.video(
                'test_vis/{i}/relighting_animation',
                nprast_animated_render(y_est[i], geo=mode['svbrdf_geo']),
                fps=settings.steps/2, i=i,
            )


def apply_ema(step, ema, params, warmup=False, warmup_power=0.9, decay=0.9999, accumulation=1):
    if warmup:
        cur_decay = 1 - (1 + step) ** -warmup_power
    else:
        cur_decay = (1 + step) / (10 + step)
    cur_decay = jnp.minimum(cur_decay, decay)
    cur_decay = jnp.power(cur_decay, 1/accumulation)

    return tree_map(lambda e, t: e * cur_decay + t * (1 - cur_decay), ema, params)


class TrainState(train_state.TrainState):
    apply_ema: Any = flax.struct.field(pytree_node=False, default=apply_ema)
    ema: Any = flax.struct.field(pytree_node=True, default=None)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        if self.ema is not None:
            new_ema = self.apply_ema(self.step, self.ema, new_params)
        else:
            new_ema = None
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema=new_ema,
            opt_state=new_opt_state,
            **kwargs,
        )

    def zero_ema(self):
        return self.replace(
            ema=tree_map(lambda t: jnp.zeros_like(t), self.params)
        )

    @property
    def eval_params(self):
        if self.ema is None:
            return self.params
        else:
            return self.ema


def pick_optimizer(mode: dict[str, Any], accumulation: int):
    lr = mode['lr']
    if mode['lr_warmup']:
        lrs = optax.linear_schedule(0.0, lr, mode['lr_warmup'])
    else:
        lrs = lr
    if mode['opt'] == 'adam':
        opt = optax.adam(lrs)
    elif mode['opt'] == 'adamw':
        opt = optax.adamw(lrs, b1=0.9, b2=0.999, weight_decay=1e-2)
    else:
        assert False, f'unknown optimizer {mode["opt"]}'
    opt = optax.apply_if_finite(opt, 6)
    if accumulation > 1:
        opt = optax.MultiSteps(opt, every_k_schedule=accumulation)
    return opt


def update_param_keys(params, root=True):
    if isinstance(params, dict):
        for k in list(params.keys()):
            k: str
            if k.startswith('FlaxAttentionBlock_'):
                params[k.replace('Flax', 'Self')] = params.pop(k)
        for v in params.values():
            update_param_keys(v)


class Model:
    gen_state: Optional[TrainState]
    """The diffusion noise model"""

    null_state: Optional[TrainState]
    """Optional unconditional noise model to support classifier free guidance scale"""

    replicated: bool = False

    def __init__(
        self, resolution: int, mode: dict[str, Any],
        init_key=0,
        eval_key=0,
        accumulation=1,
    ):
        self.mode: Any = FrozenDict(mode)
        self.accumulation = accumulation

        input_channels = mode['input_channels']
        self.input_shape = (1, resolution, resolution, input_channels)
        diffusion_channels = mode['channels']
        self.output_shape = (1, resolution, resolution, diffusion_channels)
        self.embedding_channels = mode['timestep_channels']
        self.control_shape = (1, resolution, resolution, diffusion_channels + input_channels)
        if mode['condition'] == 'clip':
            self.embedding_channels += 512
        if mode['condition'] == 'direct':
            diffusion_channels += input_channels
        self.diffusion_shape = (1, resolution, resolution, diffusion_channels)

        self.train_rng = jax.random.PRNGKey(random.randint(0, 10000))
        self.eval_rng = jax.random.PRNGKey(eval_key)
        self.init_rng = jax.random.PRNGKey(init_key)

        self.gen_state = None
        self.null_state = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, resolution=256):
        checkpoint_path = Path(checkpoint_path)
        mode = json.loads((checkpoint_path / 'mode.json').read_text())
        model = cls(resolution, mode)
        model.load_eval_checkpoint(checkpoint_path)
        return model

    def init_model(self):
        """
        Randomly initilize noise model parameters. This is optional if exactly
        restoring a model for inference. But must be called if there will be
        any new trainable parameters, even if those will be zero initilized.
        """
        assert self.gen_state is None, 'noise model is already initilized'

        print('creating noise model')
        gen_init_net = MyNoiseModel(**self.mode['noise_model'], training=False)
        gen_net = MyNoiseModel(**self.mode['noise_model'], training=True)

        print('initing niose model')
        self.init_rng, gen_init_rng = jax.random.split(self.init_rng)
        gen_init = jax.jit(gen_init_net.init)(
            {'params': gen_init_rng},
            jnp.ones(self.diffusion_shape),
            jnp.ones((1, self.embedding_channels)),
        )

        print('creating noise model train state')
        self.gen_state = TrainState.create(
            apply_fn=gen_net.apply,
            params=gen_init['params'],
            tx=pick_optimizer(self.mode, self.accumulation),
            apply_ema=partial(apply_ema, accumulation=self.accumulation),
        )
        self.null_state = None


    def load_finetune_checkpoint(self, path, use_ema=True):
        """
        Load noise model parameters from a checkpoint for finetuning. This
        pads parameters with zeros where needed to match already initilized
        parameter tensors (if any). Do not use when restoring a model
        checkpoint for inference.
        """
        assert self.gen_state is not None, 'must init model before finetuning'

        print('loading backbone checkpoint')
        starting_gen_state = ckptr.restore(path)
        if use_ema:
            starting_gen_params = starting_gen_state['ema']
        else:
            starting_gen_params = starting_gen_state['params']

        print('recreating noise model train state from checkpoint')
        starting_gen_param_dict = dict(tree_flatten_with_path(starting_gen_params)[0])

        def gen_param_mapper(path, x):
            if path in starting_gen_param_dict:
                y = starting_gen_param_dict[path]
                if x.shape == y.shape:
                    return y
                else:
                    return jnp.pad(y, [(0, a-b) for (a,b) in zip(x.shape, y.shape)])
            else:
                print('missing path', path)
                return x

        self.gen_state = TrainState.create(
            apply_fn=self.gen_state.apply_fn,
            params=tree_map_with_path(gen_param_mapper, self.gen_state.params),
            tx=pick_optimizer(self.mode, self.accumulation),
            apply_ema=partial(apply_ema, accumulation=self.accumulation),
        )
        self.gen_state = self.gen_state.zero_ema()
        self.gen_state = func_replicate(self.gen_state)

    def load_resume_checkpoint(self, path: Path):
        assert self.gen_state is not None, 'must init model before restoring'

        loaded_mode = json.loads((path / 'mode.json').read_text())
        assert self.mode == loaded_mode, 'loaded mode does not match'
        self.gen_state = ckptr.restore(path, item=self.gen_state)
        self.replicated = False

    def load_eval_checkpoint(
        self,
        path: Path,
        backbone_path: Optional[Path] = None,
        use_ema: bool = True,
    ):
        base_dict = ckptr.restore(path.absolute())
        if backbone_path is not None:
            backbone_dict = ckptr.restore(backbone_path.absolute())

            null_net = MyNoiseModel(**self.mode['noise_model'], training=False, inputs=0)
            self.null_state = TrainState.create(
                apply_fn=null_net.apply,
                params={} if use_ema else backbone_dict['params'],
                ema=backbone_dict['ema'] if use_ema else None,
                tx=optax.identity(),
                apply_ema=None,
            )
        else:
            self.null_state = None

        gen_net = MyNoiseModel(**self.mode['noise_model'], training=False)
        self.gen_state = TrainState.create(
            apply_fn=gen_net.apply,
            params={} if use_ema else base_dict['params'],
            ema=base_dict['ema'] if use_ema else None,
            tx=optax.identity(),
            apply_ema=None,
        )

    def save_training_checkpoint(self, path: Path, step: Optional[int] = None):
        ckptr.save(path, self.unreplicated_gen_state, step=step)

    def replicate(self):
        print('replicating train state')
        self.gen_state = func_replicate(self.gen_state)
        if self.null_state is not None:
            self.null_state = func_replicate(self.null_state)
        self.replicated = True

    @property
    def unreplicated_gen_state(self):
        if self.gen_state is None:
            return None
        if self.replicated:
            return func_unreplicate(self.gen_state)
        else:
            return self.gen_state

    @property
    def unreplicated_null_state(self):
        if self.null_state is None:
            return None
        if self.replicated:
            return func_unreplicate(self.null_state)
        else:
            return self.null_state

    @property
    def current_step(self):
        assert self.gen_state is not None
        return self.gen_state.step
            

    def visulize(self):
        gen_model = MyNoiseModel(**self.mode['noise_model'], training=False)
        cargs = {'force_terminal': False, 'width': 240}
        text = ''
        text += gen_model.tabulate(
            {'params': self.init_rng, 'gaussian': self.init_rng},
            jnp.ones(self.diffusion_shape),
            jnp.ones((1, self.embedding_channels)),
            console_kwargs=cargs,
        )
        return text

    def training_step(self, batch, vis: Report):
        batch = jaxify_batch(batch, allow_scalars=False)
        # 0=>batches, 1=>width, 2=>height, 3=>channels
        self.train_rng, key = jax.random.split(self.train_rng)
        self.gen_state, loss_info = generator_step_impl(
            self.gen_state,
            batch,
            key,
            self.mode,
        )
        for (k, v) in loss_info.items():
            vis.scalar('train_loss/{k}', jnp.mean(v), mean=True, k=k)


    def training_uncon_evaluate_step(self, vis: Report, count: int = 1, key=None, noise_keys=None, **kwargs):
        if key is None:
            self.eval_rng, key = jax.random.split(self.eval_rng)
        uncon_evaluate_step_impl(
            self.unreplicated_gen_state,
            (count, *self.output_shape[1:]),
            key,
            noise_keys,
            DiffusionSettings(**kwargs),
            vis,
            self.mode,
        )

    def training_evaluate_step(self, batch, vis: Report, names: Optional[list[str]] = None, key=None, noise_keys=None, **kwargs):
        batch = jaxify_batch(batch, allow_scalars=False)
        if key is None:
            self.eval_rng, key = jax.random.split(self.eval_rng)
        training_evaluate_step_impl(
            self.unreplicated_gen_state,
            batch,
            key,
            noise_keys,
            DiffusionSettings(
                true_svbrdf=batch.get('svbrdf'),
                true_render=batch['render'] if 'render' in batch else batch.get('input'),
                **kwargs
            ),
            vis,
            names,
            self.mode,
        )

    def evaluate(self, batch, key=None, noise_keys=None, **kwargs):
        batch = jaxify_batch(batch, allow_scalars=False)
        if key is None:
            self.eval_rng, key = jax.random.split(self.eval_rng)
        return evaluate_impl(
            self.unreplicated_gen_state,
            self.unreplicated_null_state,
            batch,
            key,
            noise_keys,
            settings=DiffusionSettings(
                true_svbrdf=batch.get('svbrdf'),
                true_render=batch['render'] if 'render' in batch else batch.get('input'),
                **kwargs,
            ),
            mode=self.mode,
        )
