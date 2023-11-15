import jax.numpy as jnp
import flax.linen as nn
from functools import partial

default_modes = dict()


def center_svbrdf(arr, mode):
    if mode['svbrdf_geo'] == 'height':
        rest = arr[:, :, :, 0:7]
        height = arr[:, :, :, 7:8]
        return jnp.concatenate(
            (rest * 2 - 1, height * mode['height_mult']),
            axis=3,
        )
    else:
        return arr * 2 - 1


def center_img(arr, mode):
    return arr * 2 - 1


def uncenter_svbrdf(arr, mode):
    if mode['svbrdf_geo'] == 'height':
        rest = arr[:, :, :, 0:7]
        height = arr[:, :, :, 7:8]
        return jnp.concatenate(
            (rest / 2 + 0.5, height / mode['height_mult']),
            axis=3,
        )
    else:
        return arr / 2 + 0.5


def uncenter_img(arr, mode):
    return arr / 2 + 0.5


def model_modes(input_channels, diffusion_channels, condition):
    backbone_inputs = input_channels if condition == 'direct' else 0
    backbone_channels = diffusion_channels
    timestep_channels = 128

    return [
        # this is exactly what was evaluated in the paper, bugs and all
        ('CONVNEXT_V1', {
            'noise_model': {
               'block': 'convnext',
               'inputs': backbone_inputs,
               'channels': backbone_channels,
               'cond_mlp_inputs': timestep_channels,
               'mid_activation': 'gelu',
               'wrong_heads': True,
               'cond_activation': False,
               'features': [128, 256, 512, 512, 1024, 1024],
            },
            'opt': 'adamw',
            'timestep_channels': timestep_channels,
            'timestep_mult': 1/1000,
        }),
        # this is the model with some bug fixes and improvements
        ('CONVNEXT_V1.1', {
            'noise_model': {
                'block': 'convnext',
                'inputs': backbone_inputs,
                'channels': backbone_channels,
                'cond_mlp_inputs': timestep_channels,
                'features': [128, 256, 512, 512, 1024, 1024],
            },
            'opt': 'adamw',
            'timestep_channels': timestep_channels,
        }),
    ]


diffusion_channels = 10

for input_name, input_dict in [
    ('RENDERED_IMAGES', {
        'inputs': ['render'],
        'input_channels': 3,
    }),
    ('RENDERED_OTHER_IMAGES', {
        'inputs': ['render', 'other_render'],
        'input_channels': 6,
    }),
    ('RENDERED_HALFWAY_IMAGES', {
        'inputs': ['render', 'rast_halfway'],
        'input_channels': 6,
    }),
    ('RAST_IMAGES', {
        'inputs': ['rast_flash', 'rast_halfway'],
        'input_channels': 6,
    }),
    ('RAST_NOHALFWAY_IMAGES', {
        'inputs': ['rast_flash'],
        'input_channels': 3,
    }),
]:
    for cond_name, cond_dict in [
        ('DIRECT', {
            'condition': 'direct',
            'control_model': None,
        }),
    ]:
        input_channels = input_dict['input_channels']
        condition = cond_dict['condition']
        for model_name, model_dict in model_modes(input_channels, diffusion_channels, condition):
            default_modes[f'{model_name}_{cond_name}_{input_name}_MODE'] = {
                'svbrdf_geo': 'normals',
                'lr':  2e-5,
                'lr_warmup': 10_000,
                'batch_size': 32,
                'use_ema': True,
                'ema_decay': 0.9999,
                'ema_warmup': 0.9999,
                'timestep_mult': 1,
                **model_dict,
                **cond_dict,
                **input_dict,
                'channels': diffusion_channels,
                'zero_snr': True,
            }

for model_name, model_dict in model_modes(0, diffusion_channels, 'none'):
    default_modes[f'{model_name}_UNCONDITIONAL_MODE'] = {
        'inputs': [],
        'input_channels': 0,
        'svbrdf_geo': 'normals',
        'condition': 'none',
        'lr':  2e-5,
        'lr_warmup': 10_000,
        'batch_size': 32,
        'use_ema': True,
        'ema_decay': 0.9999,
        'timestep_mult': 1,
        **model_dict,
        'channels': diffusion_channels,
        'zero_snr': True,
    }


for name in default_modes.keys():
    default_modes[name]['name'] = name
